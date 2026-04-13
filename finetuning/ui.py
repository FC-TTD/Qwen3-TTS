import json
import logging
import os
import subprocess
import sys
import threading
import gc
import shutil
from pathlib import Path
from typing import Callable, Iterable, List, Tuple, Optional

import gradio as gr
import numpy as np
import soundfile as sf
import torch

try:
    from .asr_client import auto_asr  # type: ignore
except Exception:  # pragma: no cover
    auto_asr = None

logger = logging.getLogger(__name__)

APP_DATA_DIR = Path("/app/data")
APP_LORA_DIR = Path("/app/lora")

# 全局模型缓存
_MODEL_CACHE = {
    "path": None,
    "tts": None
}

def _infer_speaker_name(input_path: str) -> str | None:
    """从 UI 输入路径中推断 speaker 名称。"""
    try:
        if not input_path or not str(input_path).strip():
            return None

        p = Path(str(input_path).strip().replace("\\", "/"))
        parts = p.parts

        # 1. 检查是否在 /app/data/<speaker> 下
        try:
            data_idx = parts.index("data")
            if data_idx > 0 and parts[data_idx-1] == "app" and len(parts) > data_idx + 1:
                spk = parts[data_idx + 1]
                if spk.endswith(".qwen.jsonl"):
                    spk = spk[:-len(".qwen.jsonl")]
                return spk.strip(" .") or None
        except (ValueError, IndexError):
            pass

        # 2. 检查是否在 /app/lora/<speaker> 下
        try:
            lora_idx = parts.index("lora")
            if lora_idx > 0 and parts[lora_idx-1] == "app" and len(parts) > lora_idx + 1:
                return parts[lora_idx + 1].strip(" .") or None
        except (ValueError, IndexError):
            pass

        # 3. 检查文件名后缀
        name = p.name
        spk = None
        if name.endswith(".qwen.jsonl"):
            spk = name[:-len(".qwen.jsonl")]
        elif name.endswith(".prepared.jsonl"):
            spk = name[:-len(".prepared.jsonl")]
        elif name.endswith(".jsonl"):
            spk = p.stem
        
        if spk:
            return spk.strip(" .") or None

        # 4. 兜底：取最后一级目录
        if p.name and p.name not in ["app", "data", "lora", "/"]:
            return p.name.strip(" .") or None
        
        return None
    except Exception:
        return None


def _default_source_jsonl_path(speaker: str) -> str:
    return str((APP_DATA_DIR / f"{speaker}.qwen.jsonl").resolve())


def _default_prepared_jsonl_path(speaker: str) -> str:
    return str((APP_LORA_DIR / speaker / f"{speaker}.prepared.jsonl").resolve())


def _default_train_dir(speaker: str) -> str:
    return str((APP_LORA_DIR / speaker).resolve())


def _is_hidden_name(name: str) -> bool:
    return name.startswith(".") or name.startswith("@")


def iter_wav_files(root_dir: str) -> Iterable[Path]:
    if not root_dir:
        return []
    root_path = Path(root_dir)
    if not root_path.exists():
        return []

    for current_root, dirs, files in os.walk(root_path, topdown=True):
        dirs[:] = [d for d in dirs if not _is_hidden_name(d)]
        for filename in files:
            if _is_hidden_name(filename):
                continue
            if filename.lower().endswith(".wav"):
                yield Path(current_root) / filename


def get_audio_duration(audio_path: Path) -> float:
    with sf.SoundFile(str(audio_path)) as f:
        if f.samplerate <= 0:
            return 0.0
        return float(len(f)) / float(f.samplerate)


def extract_text_from_filename(wav_path: Path) -> str | None:
    stem = wav_path.stem
    parts = stem.split("_")
    if len(parts) >= 5:
        return parts[-1]
    return None


def generate_manifest_jsonl(
    data_dir: str,
    output_jsonl: str,
    asr_func: Callable[[str], str] | None = None,
    overwrite: bool = False,
    use_relative_paths: bool = False,
) -> Tuple[str, int]:
    if not data_dir or not os.path.isdir(data_dir):
        return "错误：数据目录不存在或不可用", 0
    data_dir_path = Path(data_dir).resolve()
    if not output_jsonl or not output_jsonl.strip():
        output_path = (data_dir_path.parent / f"{data_dir_path.name}.qwen.jsonl").resolve()
    else:
        output_path = Path(output_jsonl).expanduser()
    if output_path.exists() and not overwrite:
        return f"错误：输出文件已存在，请勾选覆盖或更换路径：{output_path}", 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries: List[dict] = []
    existing_jsonl = _resolve_jsonl_for_dataset_dir(data_dir_path)
    
    if existing_jsonl is not None:
        logger.info(f"从现有 JSONL 读取数据: {existing_jsonl}")
        with open(existing_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except:
                        continue
    else:
        logger.info(f"扫描目录生成数据: {data_dir_path}")
        for wav_path in iter_wav_files(str(data_dir_path)):
            if "resampled_24k" in wav_path.parts:
                continue
            text = extract_text_from_filename(wav_path) or ""
            entries.append({"audio": str(wav_path), "text": text})

    if not entries:
        return "错误：未找到可用的音频条目", 0

    TARGET_SR = 24000
    resample_dir = data_dir_path / "resampled_24k"
    resample_dir.mkdir(parents=True, exist_ok=True)
    
    import librosa

    processed_entries = []
    asr_runner = asr_func or auto_asr
    
    total = len(entries)
    print(f"[Prep] 开始处理 {total} 条音频数据...")

    for i, entry in enumerate(entries):
        try:
            raw_audio_path = entry.get("audio")
            if not raw_audio_path: continue
            
            p = Path(raw_audio_path)
            if not p.is_absolute(): p = (data_dir_path / p).resolve()
            if not p.exists(): continue

            target_wav_path = resample_dir / p.name
            
            should_resample = True
            if target_wav_path.exists() and target_wav_path.stat().st_size > 0:
                should_resample = False
            
            if should_resample:
                y, sr = librosa.load(str(p), sr=None)
                if sr != TARGET_SR:
                    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
                    sf.write(str(target_wav_path), y_resampled, TARGET_SR)
                else:
                    shutil.copy2(str(p), str(target_wav_path))
            
            text = entry.get("text") or ""
            if not text.strip() and asr_runner:
                text = asr_runner(str(target_wav_path)).strip()
            
            if not text.strip(): continue

            new_entry = dict(entry)
            if use_relative_paths:
                new_entry["audio"] = os.path.relpath(str(target_wav_path), start=str(data_dir_path))
            else:
                new_entry["audio"] = str(target_wav_path)
            
            new_entry["text"] = text
            processed_entries.append(new_entry)
            
            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"[Prep] 进度: {i+1}/{total} (已处理 {len(processed_entries)} 条有效数据)")
                
        except Exception as e:
            logger.warning(f"处理条目失败: {e}")
            continue

    if not processed_entries:
        return "错误：没有成功处理任何音频条目", 0

    best_entry = max(processed_entries, key=lambda x: len(str(x.get("text", ""))), default=None)
    ref_audio = best_entry["audio"] if best_entry else None
    for e in processed_entries: e["ref_audio"] = ref_audio

    with open(output_path, "w", encoding="utf-8") as f:
        for e in processed_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    msg = f"成功：已生成源 JSONL 并在 24kHz 下重采样 {len(processed_entries)} 条样本，输出：{output_path}"
    print(f"[Prep] {msg}")
    return msg, len(processed_entries)


def build_data_prep_tab(asr_func: Callable[[str], str] | None = None) -> Tuple[gr.Textbox, gr.Textbox, gr.TextArea]:
    gr.Markdown("### 🧰 数据准备 (Data Prep)")
    data_dir = gr.Textbox(label="📂 数据目录 (包含 wav)", placeholder="/app/data/afu")
    output_jsonl = gr.Textbox(label="📄 输出 jsonl 路径（可选）", placeholder="例如：/app/data/afu.qwen.jsonl")
    overwrite = gr.Checkbox(label="覆盖已存在的 jsonl", value=False)
    use_relative = gr.Checkbox(label="使用相对路径 (相对数据目录)", value=False)
    run_btn = gr.Button("🧾 生成 jsonl", variant="primary")
    status = gr.TextArea(label="", lines=4, interactive=False, show_label=False, placeholder="等待生成...")

    def _run(d, o, ow, r):
        m, _ = generate_manifest_jsonl(data_dir=d, output_jsonl=o, asr_func=asr_func, overwrite=ow, use_relative_paths=r)
        return m

    def _on_data_dir_change(dir_val, out_val):
        speaker = _infer_speaker_name(dir_val)
        if speaker and (not out_val or not str(out_val).strip()):
            return _default_source_jsonl_path(speaker)
        return out_val

    data_dir.change(_on_data_dir_change, inputs=[data_dir, output_jsonl], outputs=[output_jsonl])
    run_btn.click(_run, inputs=[data_dir, output_jsonl, overwrite, use_relative], outputs=[status])
    return data_dir, output_jsonl, status


training_process: subprocess.Popen | None = None
training_log = ""

def scan_checkpoints(output_dir: str = "finetuning/output") -> list[str]:
    checkpoints: list[str] = []
    if not output_dir or not str(output_dir).strip(): return ["未找到检查点"]
    base_path = Path(output_dir)
    if not base_path.exists(): return ["未找到检查点"]
    for cp_dir in sorted(base_path.glob("checkpoint-epoch-*"), reverse=True):
        if (cp_dir / "model.safetensors").exists():
            checkpoints.append(cp_dir.name)
    return checkpoints if checkpoints else ["未找到检查点"]


def run_prepare_codes(input_jsonl: str, output_jsonl: str, tokenizer_model_path: str, device: str) -> str:
    try:
        if not input_jsonl or not str(input_jsonl).strip(): return "错误：请输入源 JSONL 路径或数据集目录。"
        input_path = Path(input_jsonl).expanduser() if input_jsonl else None
        if input_path is None or not input_path.exists(): return f"错误：输入路径不存在：{input_jsonl}"

        if input_path.is_dir():
            existing_qwen = _resolve_qwen_jsonl_for_dataset_dir(input_path)
            if existing_qwen is None: return "错误：预处理阶段未找到源 JSONL（*.qwen.jsonl）。"
            input_jsonl = existing_qwen
        else:
            input_jsonl = str(input_path)

        speaker = _infer_speaker_name(input_jsonl)
        if not speaker: return f"错误：无法从输入路径推断 speaker：{input_jsonl}"
        if not output_jsonl or not str(output_jsonl).strip():
            output_jsonl = _default_prepared_jsonl_path(speaker)

        print(f"[Prep] 强制检查重采样逻辑 (24kHz)...")
        import librosa
        input_jsonl_path = Path(input_jsonl)
        data_root = input_jsonl_path.parent
        resample_dir = data_root / "resampled_24k"
        resample_dir.mkdir(parents=True, exist_ok=True)
        TARGET_SR = 24000
        resampled_jsonl_path = input_jsonl_path.with_suffix(".resampled_tmp.jsonl")
        new_entries = []
        with open(input_jsonl, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            for i, line in enumerate(lines):
                try:
                    entry = json.loads(line)
                    audio_path = entry.get("audio")
                    if not audio_path: continue
                    p = Path(audio_path)
                    if not p.is_absolute(): p = (data_root / p).resolve()
                    if not p.exists(): continue
                    target_wav = resample_dir / p.name
                    if not target_wav.exists() or target_wav.stat().st_size == 0:
                        y, sr = librosa.load(str(p), sr=None)
                        if sr != TARGET_SR:
                            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
                            sf.write(str(target_wav), y_resampled, TARGET_SR)
                        else:
                            shutil.copy2(str(p), str(target_wav))
                    new_entry = dict(entry)
                    new_entry["audio"] = str(target_wav)
                    ref_audio = entry.get("ref_audio")
                    if ref_audio:
                        rp = Path(ref_audio)
                        if not rp.is_absolute(): rp = (data_root / rp).resolve()
                        if rp.exists():
                            target_ref = resample_dir / rp.name
                            if not target_ref.exists() or target_ref.stat().st_size == 0:
                                ry, rsr = librosa.load(str(rp), sr=None)
                                if rsr != TARGET_SR:
                                    ry_resampled = librosa.resample(ry, orig_sr=rsr, target_sr=TARGET_SR)
                                    sf.write(str(target_ref), ry_resampled, TARGET_SR)
                                else:
                                    shutil.copy2(str(rp), str(target_ref))
                            new_entry["ref_audio"] = str(target_ref)
                    new_entries.append(new_entry)
                except Exception: continue

        with open(resampled_jsonl_path, "w", encoding="utf-8") as f:
            for e in new_entries: f.write(json.dumps(e, ensure_ascii=False) + "\n")
        
        finetuning_dir = Path(__file__).parent
        output_jsonl_path = Path(output_jsonl)
        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(finetuning_dir / "prepare_data.py"), "--device", device, "--tokenizer_model_path", tokenizer_model_path, "--input_jsonl", str(resampled_jsonl_path), "--output_jsonl", output_jsonl]
        subprocess.run(cmd, check=True)
        try: resampled_jsonl_path.unlink()
        except: pass
        if Path(output_jsonl).exists():
            with open(output_jsonl, "r", encoding="utf-8") as f: line_count = sum(1 for _ in f)
            return f"成功！已处理 {line_count} 条样本（自动重采样至 24kHz）。\n输出保存至：{output_jsonl}"
        return "警告：输出文件未创建。"
    except Exception as e:
        logger.exception("预处理失败")
        return f"错误：{type(e).__name__}: {e}"


def start_training(init_model_path, train_jsonl, output_model_path, batch_size, lr, num_epochs, speaker_name) -> str:
    global training_process, training_log
    if training_process is not None and training_process.poll() is None: return "错误：训练正在进行中！"
    if not Path(train_jsonl).exists(): return f"错误：训练数据不存在：{train_jsonl}"
    output_path = Path(output_model_path)
    output_path.mkdir(parents=True, exist_ok=True)
    finetuning_dir = Path(__file__).parent
    cmd = [sys.executable, str(finetuning_dir / "sft_12hz.py"), "--init_model_path", init_model_path, "--output_model_path", output_model_path, "--train_jsonl", train_jsonl, "--batch_size", str(int(batch_size)), "--lr", str(float(lr)), "--num_epochs", str(int(num_epochs)), "--speaker_name", speaker_name]
    training_log = f"开始训练...\n命令：{' '.join(cmd)}\n\n"
    def run_process() -> None:
        global training_process, training_log
        training_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=str(finetuning_dir))
        for line in training_process.stdout:
            training_log += line
            if len(training_log) > 100000: training_log = training_log[-100000:]
        training_process.wait()
        training_log += f"\n\n训练完成，返回码：{training_process.returncode}"
    threading.Thread(target=run_process, daemon=True).start()
    return f"训练已开始！输出将保存至：{output_model_path}"


def get_training_log() -> str: return training_log
def stop_training() -> str:
    global training_process, training_log
    if training_process is not None and training_process.poll() is None:
        training_process.terminate()
        training_log += "\n[用户手动终止训练]"
        return "训练已停止。"
    return "当前没有正在运行的训练。"


def test_finetuned_model(
    checkpoint_name: str,
    test_text: str,
    speaker_name: str,
    checkpoint_base_dir: str,
    cleanup_models: Callable[[], None] | None = None,
) -> Tuple[tuple[int, np.ndarray] | None, str]:
    if not test_text or not test_text.strip(): return None, "错误：请输入文本。"
    if checkpoint_name == "未找到检查点" or not checkpoint_name: return None, "错误：未选择有效的检查点"

    try:
        from qwen_tts import Qwen3TTSModel
        if not checkpoint_base_dir or not str(checkpoint_base_dir).strip():
            return None, "错误：请填写检查点目录"

        checkpoint_dir = Path(checkpoint_base_dir) / checkpoint_name
        if not checkpoint_dir.exists(): return None, f"错误：路径不存在：{checkpoint_dir}"

        # 核心加固：自动纠正说话人名称
        if not speaker_name or not str(speaker_name).strip():
            speaker_name = _infer_speaker_name(checkpoint_base_dir) or "speaker_test"
        
        print(f"[Test] 正在准备生成：Speaker={speaker_name}, Checkpoint={checkpoint_name}")

        # 模型缓存逻辑
        global _MODEL_CACHE
        target_path = str(checkpoint_dir.resolve())
        if _MODEL_CACHE["path"] != target_path:
            print(f"[Test] 正在从硬盘加载模型 (缓存未命中)：{checkpoint_name}")
            if cleanup_models: cleanup_models()
            _MODEL_CACHE["tts"] = Qwen3TTSModel.from_pretrained(
                target_path,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
            )
            _MODEL_CACHE["path"] = target_path
        else:
            print(f"[Test] 使用已加载的模型缓存 (秒开)：{checkpoint_name}")

        tts = _MODEL_CACHE["tts"]
        if tts is None:
            return None, "错误：模型未成功加载，请重试。"
        
        # 核心诊断日志
        supported_spks = tts.get_supported_speakers()
        print(f"[Debug] 模型支持的说话人列表: {supported_spks}")
        print(f"[Debug] 当前请求的说话人名称: {speaker_name}")
        
        # 强制归一化检测
        target_spk = speaker_name
        if supported_spks:
            # 尝试在支持列表中找匹配（忽略大小写）
            lower_map = {s.lower(): s for s in supported_spks}
            if speaker_name.lower() in lower_map:
                target_spk = lower_map[speaker_name.lower()]
                print(f"[Debug] 找到匹配名，重定向为: {target_spk}")
            else:
                print(f"[Warning] 请求的 {speaker_name} 不在支持列表 {supported_spks} 中！")
            print(f"[Debug] Speaker 选择结果: requested={speaker_name}, selected={target_spk}, matched={speaker_name.lower() in lower_map}")
        else:
            print(f"[Debug] Speaker 选择结果: requested={speaker_name}, selected={target_spk}, matched=unknown(模型未提供支持列表)")

        wavs, sr = tts.generate_custom_voice(
            text=test_text.strip(),
            language="Auto",
            speaker=target_spk,
            max_new_tokens=2048,
        )

        return (sr, wavs[0]), f"成功：{speaker_name} (缓存: {'命中' if _MODEL_CACHE['path'] == target_path else '加载'})"
    except Exception as e:
        logger.exception("测试失败")
        return None, f"错误：{type(e).__name__}: {e}"


def build_finetuning_tab(cleanup_models: Callable[[], None] | None = None, asr_func: Callable[[str], str] | None = None) -> None:
    with gr.Tab("语音微调"):
        gr.Markdown("### 自定义语音微调")
        with gr.Tabs():
            with gr.Tab("0. 生成训练数据"):
                build_data_prep_tab(asr_func=asr_func)

            with gr.Tab("1. 数据预处理"):
                gr.Markdown("#### 提取音频编码 + 自动选择参考音频")
                with gr.Row():
                    with gr.Column(scale=2):
                        prep_input_jsonl = gr.Textbox(label="输入路径", placeholder="/app/data/afu")
                        prep_output_jsonl = gr.Textbox(label="输出路径", placeholder="/app/lora/afu/afu.prepared.jsonl")
                        prep_tokenizer_path = gr.Textbox(label="分词器模型", value="Qwen/Qwen3-TTS-Tokenizer-12Hz")
                        prep_device = gr.Dropdown(label="运行设备", choices=["cuda:0", "cuda", "cpu"], value="cuda:0")
                        prep_btn = gr.Button("开始预处理", variant="primary")
                    with gr.Column(scale=2):
                        prep_status = gr.Textbox(label="处理状态", lines=10, interactive=False)
                prep_btn.click(run_prepare_codes, inputs=[prep_input_jsonl, prep_output_jsonl, prep_tokenizer_path, prep_device], outputs=[prep_status])

            with gr.Tab("2. 模型训练"):
                gr.Markdown("#### 训练微调模型")
                with gr.Row():
                    with gr.Column(scale=2):
                        train_init_model = gr.Textbox(label="基础模型", value="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
                        train_data_path = gr.Textbox(label="训练数据路径", placeholder="/app/lora/afu/afu.prepared.jsonl")
                        train_output_path = gr.Textbox(label="输出目录", placeholder="/app/lora/afu")
                        train_speaker_name = gr.Textbox(label="说话人名称", placeholder="afu")
                        with gr.Row():
                            train_batch_size = gr.Number(label="批次大小", value=2, precision=0)
                            train_lr = gr.Number(label="学习力度 (建议 0.00001)", value=0.00001)
                            train_epochs = gr.Number(label="训练轮数", value=3, precision=0)
                        gr.Markdown("💡 **提示**：建议力度 `0.00001`。数据多于 30 条时建议调低至 `0.000005`。")
                        with gr.Row():
                            train_start_btn = gr.Button("开始训练", variant="primary")
                            train_stop_btn = gr.Button("停止训练", variant="stop")
                    with gr.Column(scale=2):
                        train_status = gr.Textbox(label="训练状态", value="等待训练", interactive=False)
                        train_logs = gr.Textbox(label="训练日志", lines=15, interactive=False)
                        refresh_logs_btn = gr.Button("刷新日志", size="sm")
                refresh_logs_btn.click(get_training_log, outputs=[train_logs])
                train_start_btn.click(start_training, inputs=[train_init_model, train_data_path, train_output_path, train_batch_size, train_lr, train_epochs, train_speaker_name], outputs=[train_status])
                train_stop_btn.click(stop_training, outputs=[train_status])

            with gr.Tab("3. 推理测试"):
                gr.Markdown("#### 测试微调后的模型")
                with gr.Row():
                    with gr.Column(scale=2):
                        test_base_dir = gr.Textbox(label="检查点根目录", placeholder="/app/lora/afu")
                        test_checkpoint = gr.Dropdown(label="选择检查点", choices=["未找到检查点"])
                        test_refresh_btn = gr.Button("刷新列表", size="sm")
                        test_text = gr.Textbox(label="测试文本", lines=3, value="你好！这是我微调后的声音测试。")
                        test_speaker = gr.Textbox(label="说话人名称")
                        test_generate_btn = gr.Button("生成语音", variant="primary")
                    with gr.Column(scale=2):
                        test_audio = gr.Audio(label="生成的音频", type="numpy")
                        test_status = gr.Textbox(label="测试状态", lines=2, interactive=False)

                test_refresh_btn.click(lambda d: gr.Dropdown(choices=scan_checkpoints(d)), inputs=[test_base_dir], outputs=[test_checkpoint])
                test_generate_btn.click(test_finetuned_model, inputs=[test_checkpoint, test_text, test_speaker, test_base_dir], outputs=[test_audio, test_status])

            # 联动逻辑
            def _on_prep_input_change(p, o):
                s = _infer_speaker_name(p)
                return _default_prepared_jsonl_path(s) if s and not o else o
            prep_input_jsonl.change(_on_prep_input_change, inputs=[prep_input_jsonl, prep_output_jsonl], outputs=[prep_output_jsonl])

            def _on_train_data_change(p, o, s):
                spk = _infer_speaker_name(p)
                if spk:
                    if not s or s != spk: s = spk
                    if not o or o == _default_train_dir(s): o = _default_train_dir(spk)
                return o, s
            train_data_path.change(_on_train_data_change, inputs=[train_data_path, train_output_path, train_speaker_name], outputs=[train_output_path, train_speaker_name])

            def _on_test_base_dir_change(b, s):
                spk = _infer_speaker_name(b)
                return spk if spk and (not s or s != spk) else s
            test_base_dir.change(_on_test_base_dir_change, inputs=[test_base_dir, test_speaker], outputs=[test_speaker])


def _find_jsonl_in_dir(data_dir: str) -> str | None:
    try:
        root = Path(data_dir)
        if not root.exists() or not root.is_dir(): return None
        candidates = sorted([str(p) for p in root.glob("*.jsonl") if not _is_hidden_name(p.name)])
        return candidates[0] if candidates else None
    except Exception: return None

def _resolve_jsonl_for_dataset_dir(dataset_dir: Path) -> str | None:
    try:
        if not dataset_dir.exists() or not dataset_dir.is_dir(): return None
        sibling = dataset_dir.parent / f"{dataset_dir.name}.jsonl"
        if sibling.exists(): return str(sibling)
        return _find_jsonl_in_dir(str(dataset_dir))
    except Exception: return None

def _resolve_qwen_jsonl_for_dataset_dir(dataset_dir: Path) -> str | None:
    try:
        if not dataset_dir.exists() or not dataset_dir.is_dir(): return None
        sibling = dataset_dir.parent / f"{dataset_dir.name}.qwen.jsonl"
        if sibling.exists(): return str(sibling)
        inner = dataset_dir / f"{dataset_dir.name}.qwen.jsonl"
        if inner.exists(): return str(inner)
        return None
    except Exception: return None

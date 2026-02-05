import json
import logging
import os
import subprocess
import sys
import threading
import gc
import shutil
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

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


def _infer_speaker_name(input_path: str) -> str | None:
    """从 UI 输入路径中推断 speaker 名称。"""
    try:
        if not input_path or not str(input_path).strip():
            return None

        p = Path(str(input_path).strip().replace("\\", "/"))
        parts = p.parts

        # 1. 检查是否在 /app/data/<speaker> 下
        try:
            # parts 可能类似 ('/', 'app', 'data', 'afu')
            data_idx = parts.index("data")
            if data_idx > 0 and parts[data_idx-1] == "app" and len(parts) > data_idx + 1:
                spk = parts[data_idx + 1]
                # 去掉可能存在的扩展名 .qwen.jsonl
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
        # Step0 默认输出：/app/data/<speaker>.qwen.jsonl
        output_path = (data_dir_path.parent / f"{data_dir_path.name}.qwen.jsonl").resolve()
    else:
        output_path = Path(output_jsonl).expanduser()
    if output_path.exists() and not overwrite:
        return f"错误：输出文件已存在，请勾选覆盖或更换路径：{output_path}", 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step0：生成/升级“源 JSONL”（输出必须包含 ref_audio）。
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

    # 重采样目标采样率
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
            if not raw_audio_path:
                continue
            
            p = Path(raw_audio_path)
            if not p.is_absolute():
                p = (data_dir_path / p).resolve()
            
            if not p.exists():
                logger.warning(f"音频文件不存在，跳过: {p}")
                continue

            target_wav_path = resample_dir / p.name
            
            # Cache 检查
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
            
            if not text.strip():
                continue

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

    # 自动选择 ref_audio
    best_entry = max(processed_entries, key=lambda x: len(str(x.get("text", ""))), default=None)
    ref_audio = best_entry["audio"] if best_entry else None
    
    for e in processed_entries:
        e["ref_audio"] = ref_audio

    with open(output_path, "w", encoding="utf-8") as f:
        for e in processed_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    msg = f"成功：已生成源 JSONL 并在 24kHz 下重采样 {len(processed_entries)} 条样本，输出：{output_path}"
    print(f"[Prep] {msg}")
    return msg, len(processed_entries)


def build_data_prep_tab(
    asr_func: Callable[[str], str] | None = None,
) -> Tuple[gr.Textbox, gr.Textbox, gr.TextArea]:
    gr.Markdown(
        """
        ### 🧰 数据准备 (Data Prep)
        扫描指定目录下的 wav 文件，自动 ASR + 时长提取，生成可训练的 jsonl。
        """
    )

    data_dir = gr.Textbox(
        label="📂 数据目录 (包含 wav)",
        placeholder="/app/data/afu",
        value="",
    )
    output_jsonl = gr.Textbox(
        label="📄 输出 jsonl 路径（可选）",
        placeholder="例如：/app/data/afu.qwen.jsonl",
        value="",
    )
    overwrite = gr.Checkbox(
        label="覆盖已存在的 jsonl",
        value=False,
    )
    use_relative = gr.Checkbox(
        label="使用相对路径 (相对数据目录)",
        value=False,
    )

    run_btn = gr.Button("🧾 生成 jsonl", variant="primary")
    status = gr.TextArea(
        label="",
        lines=4,
        interactive=False,
        show_label=False,
        placeholder="等待生成...",
    )

    def _run(
        data_dir_val: str,
        output_jsonl_val: str,
        overwrite_val: bool,
        use_relative_val: bool,
    ) -> str:
        message, _ = generate_manifest_jsonl(
            data_dir=data_dir_val,
            output_jsonl=output_jsonl_val,
            asr_func=asr_func,
            overwrite=overwrite_val,
            use_relative_paths=use_relative_val,
        )
        return message

    def _on_data_dir_change(dir_val: str, out_val: str):
        speaker = _infer_speaker_name(dir_val)
        if speaker and (not out_val or not str(out_val).strip()):
            return _default_source_jsonl_path(speaker)
        return out_val

    data_dir.change(
        _on_data_dir_change,
        inputs=[data_dir, output_jsonl],
        outputs=[output_jsonl],
    )

    run_btn.click(
        _run,
        inputs=[data_dir, output_jsonl, overwrite, use_relative],
        outputs=[status],
    )

    return data_dir, output_jsonl, status


# ===================== Fine-tuning UI (Train/Test) =====================

training_process: subprocess.Popen | None = None
training_log = ""


def scan_checkpoints(output_dir: str = "finetuning/output") -> list[str]:
    checkpoints: list[str] = []
    if not output_dir or not str(output_dir).strip():
        return ["未找到检查点"]

    base_path = Path(output_dir)
    if not base_path.exists():
        return ["未找到检查点"]

    for cp_dir in sorted(base_path.glob("checkpoint-epoch-*"), reverse=True):
        if (cp_dir / "model.safetensors").exists():
            checkpoints.append(cp_dir.name)

    return checkpoints if checkpoints else ["未找到检查点"]


def find_best_ref_audio(input_jsonl: str) -> str | None:
    """从jsonl文件中找到text最长的音频文件作为ref_audio"""
    try:
        input_jsonl_path = Path(input_jsonl)
        if not input_jsonl_path.exists():
            return None
        jsonl_dir = input_jsonl_path.parent
        
        best_entry = None
        max_text_length = 0
        
        with open(input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    text = entry.get("text", "")
                    audio_path = entry.get("audio", "")

                    resolved_audio_path = Path(audio_path)
                    if not resolved_audio_path.is_absolute():
                        resolved_audio_path = (jsonl_dir / resolved_audio_path).resolve()

                    if text and audio_path and resolved_audio_path.exists():
                        text_length = len(text.strip())
                        if text_length > max_text_length:
                            max_text_length = text_length
                            best_entry = dict(entry)
                            best_entry["_resolved_audio"] = str(resolved_audio_path)
                except json.JSONDecodeError:
                    continue
        
        if best_entry:
            ref_audio_path = str(best_entry.get("_resolved_audio") or best_entry["audio"])
            logger.info(f"选择ref_audio: {ref_audio_path} (text长度: {max_text_length})")
            return ref_audio_path
        
        return None
    except Exception as e:
        logger.exception("查找ref_audio失败")
        return None


def add_ref_audio_to_jsonl(input_jsonl: str, output_jsonl: str, ref_audio_path: str) -> Tuple[str, int]:
    """为jsonl文件添加统一的ref_audio字段"""
    try:
        entries = []
        count = 0
        
        with open(input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    entry["ref_audio"] = ref_audio_path
                    entries.append(entry)
                    count += 1
                except json.JSONDecodeError:
                    continue
        
        # 写入新文件
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        return f"成功：为 {count} 条数据添加ref_audio: {ref_audio_path}", count
    except Exception as e:
        return f"错误：添加ref_audio失败: {e}", 0


def _find_jsonl_in_dir(data_dir: str) -> str | None:
    try:
        root = Path(data_dir)
        if not root.exists() or not root.is_dir():
            return None

        candidates: list[Path] = []
        for p in sorted(root.glob("*.jsonl")):
            if _is_hidden_name(p.name):
                continue
            candidates.append(p)

        if not candidates:
            return None

        return str(candidates[0])
    except Exception:
        logger.exception("查找目录下 jsonl 失败：%s", data_dir)
        return None


def _resolve_jsonl_for_dataset_dir(dataset_dir: Path) -> str | None:
    """适配布局：dataset_dir 与 <name>.jsonl 同级（父目录）。"""
    try:
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            return None

        sibling_jsonl = dataset_dir.parent / f"{dataset_dir.name}.jsonl"
        if sibling_jsonl.exists() and sibling_jsonl.is_file():
            return str(sibling_jsonl)

        inner = _find_jsonl_in_dir(str(dataset_dir))
        if inner is not None:
            return inner

        return None
    except Exception:
        logger.exception("解析数据集目录对应的 jsonl 失败：%s", dataset_dir)
        return None


def _resolve_qwen_jsonl_for_dataset_dir(dataset_dir: Path) -> str | None:
    """预处理阶段仅解析“已准备好的源 JSONL”（.qwen.jsonl）。"""
    try:
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            return None

        sibling_jsonl = dataset_dir.parent / f"{dataset_dir.name}.qwen.jsonl"
        if sibling_jsonl.exists() and sibling_jsonl.is_file():
            return str(sibling_jsonl)

        inner_jsonl = dataset_dir / f"{dataset_dir.name}.qwen.jsonl"
        if inner_jsonl.exists() and inner_jsonl.is_file():
            return str(inner_jsonl)

        return None
    except Exception:
        logger.exception("解析数据集目录对应的 qwen jsonl 失败：%s", dataset_dir)
        return None


def run_prepare_codes(
    input_jsonl: str,
    output_jsonl: str,
    tokenizer_model_path: str,
    device: str,
) -> str:
    try:
        if not input_jsonl or not str(input_jsonl).strip():
            return "错误：请输入源 JSONL 路径或数据集目录。"

        input_path = Path(input_jsonl).expanduser() if input_jsonl else None
        if input_path is None or not input_path.exists():
            return f"错误：输入路径不存在：{input_jsonl}"

        if input_path.is_dir():
            existing_qwen = _resolve_qwen_jsonl_for_dataset_dir(input_path)
            if existing_qwen is None:
                return (
                    "错误：预处理阶段未找到源 JSONL（*.qwen.jsonl）。\n"
                    "请先在 Step0 ‘生成 jsonl’ 里对该目录生成/升级源 JSONL。"
                )
            input_jsonl = existing_qwen
        else:
            input_jsonl = str(input_path)

        speaker = _infer_speaker_name(input_jsonl)
        if not speaker:
            return f"错误：无法从输入路径推断 speaker：{input_jsonl}"

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
            total = len(lines)
            
            for i, line in enumerate(lines):
                try:
                    entry = json.loads(line)
                    audio_path = entry.get("audio")
                    if not audio_path: continue
                    
                    p = Path(audio_path)
                    if not p.is_absolute():
                        p = (data_root / p).resolve()
                    
                    if not p.exists():
                        continue
                    
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
                    if (i+1) % 50 == 0 or (i+1) == total:
                        print(f"[Prep] 重采样进度: {i+1}/{total}")
                except Exception:
                    continue

        with open(resampled_jsonl_path, "w", encoding="utf-8") as f:
            for e in new_entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        
        finetuning_dir = Path(__file__).parent
        output_jsonl_path = Path(output_jsonl)
        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            sys.executable,
            str(finetuning_dir / "prepare_data.py"),
            "--device", device,
            "--tokenizer_model_path", tokenizer_model_path,
            "--input_jsonl", str(resampled_jsonl_path),
            "--output_jsonl", output_jsonl,
        ]

        subprocess.run(cmd, check=True)
        
        try: resampled_jsonl_path.unlink()
        except: pass

        if Path(output_jsonl).exists():
            with open(output_jsonl, "r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
            return f"成功！已处理 {line_count} 条样本（自动重采样至 24kHz）。\n输出保存至：{output_jsonl}"

        return "警告：输出文件未创建。"
    except subprocess.CalledProcessError:
        return "数据准备出错，请检查控制台日志。"
    except Exception as e:
        logger.exception("预处理失败")
        return f"错误：{type(e).__name__}: {e}"


def start_training(
    init_model_path: str,
    train_jsonl: str,
    output_model_path: str,
    batch_size: int,
    lr: float,
    num_epochs: int,
    speaker_name: str,
) -> str:
    global training_process, training_log

    if training_process is not None and training_process.poll() is None:
        return "错误：训练正在进行中！"

    if not Path(train_jsonl).exists():
        return f"错误：训练数据不存在：{train_jsonl}"

    output_path = Path(output_model_path)
    output_path.mkdir(parents=True, exist_ok=True)

    finetuning_dir = Path(__file__).parent
    cmd = [
        sys.executable,
        str(finetuning_dir / "sft_12hz.py"),
        "--init_model_path", init_model_path,
        "--output_model_path", output_model_path,
        "--train_jsonl", train_jsonl,
        "--batch_size", str(int(batch_size)),
        "--lr", str(float(lr)),
        "--num_epochs", str(int(num_epochs)),
        "--speaker_name", speaker_name,
    ]

    training_log = f"开始训练...\n命令：{' '.join(cmd)}\n\n"

    def run_process() -> None:
        global training_process, training_log
        training_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(finetuning_dir),
        )
        assert training_process.stdout is not None
        for line in training_process.stdout:
            training_log += line
            if len(training_log) > 100000:
                training_log = training_log[-100000:]

        training_process.wait()
        training_log += f"\n\n训练完成，返回码：{training_process.returncode}"

    threading.Thread(target=run_process, daemon=True).start()
    return f"训练已开始！输出将保存至：{output_model_path}"


def get_training_log() -> str:
    return training_log


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
    if not test_text or not test_text.strip():
        return None, "错误：请输入文本。"
    if checkpoint_name == "未找到检查点" or not checkpoint_name:
        return None, "错误：未选择有效的检查点"

    try:
        if cleanup_models is not None:
            cleanup_models()

        from qwen_tts import Qwen3TTSModel

        if not checkpoint_base_dir or not str(checkpoint_base_dir).strip():
            return None, "错误：请填写训练输出目录（/app/lora/<speaker>）"

        checkpoint_dir = Path(checkpoint_base_dir) / checkpoint_name
        if not checkpoint_dir.exists():
            return None, f"错误：检查点不存在：{checkpoint_dir}"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        attn_impl = None
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except Exception:
            attn_impl = None

        tts = Qwen3TTSModel.from_pretrained(
            str(checkpoint_dir),
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )

        wavs, sr = tts.generate_custom_voice(
            text=test_text.strip(),
            language="Auto",
            speaker=speaker_name,
            max_new_tokens=2048,
        )

        del tts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (sr, wavs[0]), f"使用检查点生成成功：{checkpoint_name}"
    except Exception as e:
        return None, f"错误：{type(e).__name__}: {e}"


def build_finetuning_tab(
    cleanup_models: Callable[[], None] | None = None,
    asr_func: Callable[[str], str] | None = None,
) -> None:
    with gr.Tab("语音微调"):
        gr.Markdown(
            """
            ### 自定义语音微调
            推荐流程：
            1) 自动生成训练数据（扫描音频 + ASR 识别）
            2) 数据预处理（提取音频编码）
            3) 模型训练
            4) 推理测试
            """
        )

        with gr.Tabs():
            with gr.Tab("0. 生成训练数据"):
                build_data_prep_tab(asr_func=asr_func)

            with gr.Tab("1. 数据预处理"):
                gr.Markdown(
                    """
                    #### 提取音频编码 + 自动选择参考音频
                    - 将 jsonl 中的音频文件编码为 `audio_codes`，用于后续训练
                    - **自动选择ref_audio**：从所有数据中选择text最长的音频文件作为统一参考音频
                    - 为所有数据添加统一的 `ref_audio` 字段，确保训练稳定性
                    """
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        prep_input_jsonl = gr.Textbox(
                            label="输入路径（JSONL 或 数据集目录）",
                            placeholder="例如：/app/data/afu 或 /app/data/afu.qwen.jsonl",
                            value="",
                        )
                        prep_output_jsonl = gr.Textbox(
                            label="输出 JSONL 路径",
                            placeholder="例如：/app/lora/afu/afu.prepared.jsonl",
                            value="",
                        )
                        prep_tokenizer_path = gr.Textbox(
                            label="分词器模型路径",
                            value="Qwen/Qwen3-TTS-Tokenizer-12Hz",
                        )
                        prep_device = gr.Dropdown(
                            label="运行设备",
                            choices=["cuda:0", "cuda", "cpu"],
                            value="cuda:0",
                        )
                        prep_btn = gr.Button("开始预处理", variant="primary")
                    with gr.Column(scale=2):
                        prep_status = gr.Textbox(
                            label="处理状态",
                            lines=10,
                            interactive=False,
                        )

                prep_btn.click(
                    run_prepare_codes,
                    inputs=[prep_input_jsonl, prep_output_jsonl, prep_tokenizer_path, prep_device],
                    outputs=[prep_status],
                )

            with gr.Tab("2. 模型训练"):
                gr.Markdown("#### 训练微调模型")

                with gr.Row():
                    with gr.Column(scale=2):
                        train_init_model = gr.Textbox(
                            label="基础模型路径",
                            value="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                            placeholder="例如：Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        )
                        train_data_path = gr.Textbox(
                            label="训练数据路径（已预处理的 JSONL）",
                            placeholder="例如：/app/lora/afu/afu.prepared.jsonl",
                            value="",
                        )
                        train_output_path = gr.Textbox(
                            label="输出目录",
                            value="",
                            placeholder="例如：/app/lora/afu",
                        )
                        train_speaker_name = gr.Textbox(
                            label="说话人名称",
                            placeholder="例如：afu",
                            value="",
                        )

                        with gr.Row():
                            train_batch_size = gr.Number(label="批次大小", value=2, precision=0)
                            train_lr = gr.Number(label="学习率", value=2e-5)
                            train_epochs = gr.Number(label="训练轮数", value=3, precision=0)

                        with gr.Row():
                            train_start_btn = gr.Button("开始训练", variant="primary")
                            train_stop_btn = gr.Button("停止训练", variant="stop")

                    with gr.Column(scale=2):
                        train_status = gr.Textbox(
                            label="训练状态",
                            value="准备就绪，点击开始训练",
                            interactive=False,
                        )
                        train_logs = gr.Textbox(
                            label="训练日志",
                            lines=15,
                            interactive=False,
                        )
                        refresh_logs_btn = gr.Button("刷新日志", size="sm")

                refresh_logs_btn.click(get_training_log, outputs=[train_logs])
                train_start_btn.click(
                    start_training,
                    inputs=[
                        train_init_model, train_data_path, train_output_path,
                        train_batch_size, train_lr, train_epochs, train_speaker_name,
                    ],
                    outputs=[train_status],
                )
                train_stop_btn.click(stop_training, outputs=[train_status])

            with gr.Tab("3. 推理测试"):
                gr.Markdown("#### 测试微调后的模型")

                with gr.Row():
                    with gr.Column(scale=2):
                        test_base_dir = gr.Textbox(
                            label="训练输出目录（检查点目录）",
                            placeholder="例如：/app/lora/afu",
                            value="",
                        )
                        test_checkpoint = gr.Dropdown(
                            label="选择检查点",
                            choices=["未找到检查点"],
                            value=None,
                            interactive=True,
                        )
                        test_refresh_btn = gr.Button("刷新检查点列表", size="sm")
                        test_text = gr.Textbox(
                            label="测试文本",
                            lines=3,
                            value="你好！这是我微调后的声音测试。",
                        )
                        test_speaker = gr.Textbox(
                            label="说话人名称",
                            value="",
                        )
                        test_generate_btn = gr.Button("生成语音", variant="primary")
                    with gr.Column(scale=2):
                        test_audio = gr.Audio(label="生成的音频", type="numpy")
                        test_status = gr.Textbox(label="测试状态", lines=2, interactive=False)

                def _update_checkpoints(base_dir: str) -> gr.Dropdown:
                    return gr.Dropdown(choices=scan_checkpoints(base_dir))

                test_refresh_btn.click(
                    _update_checkpoints,
                    inputs=[test_base_dir],
                    outputs=[test_checkpoint],
                )

                def _test(cp: str, txt: str, spk: str, base_dir: str):
                    return test_finetuned_model(
                        checkpoint_name=cp, test_text=txt,
                        speaker_name=spk, checkpoint_base_dir=base_dir,
                        cleanup_models=cleanup_models,
                    )

                test_generate_btn.click(
                    _test,
                    inputs=[test_checkpoint, test_text, test_speaker, test_base_dir],
                    outputs=[test_audio, test_status],
                )

            # ----------------- UI 自动联动 -----------------
            def _on_prep_input_change(path_val: str, output_val: str):
                speaker = _infer_speaker_name(path_val)
                if not speaker: return output_val
                current = str(output_val).strip()
                if not current or "a/a.prepared.jsonl" in current or current.endswith("/.prepared.jsonl"):
                    return _default_prepared_jsonl_path(speaker)
                return output_val

            prep_input_jsonl.change(
                _on_prep_input_change,
                inputs=[prep_input_jsonl, prep_output_jsonl],
                outputs=[prep_output_jsonl],
            )

            def _on_train_data_change(train_data_val: str, out_dir_val: str, spk_val: str):
                speaker = _infer_speaker_name(train_data_val)
                new_out_dir = out_dir_val
                new_spk = spk_val
                if speaker:
                    # 只要推断出了新说话人，且与当前不一致，就更新联动
                    if not new_spk or not str(new_spk).strip() or speaker != new_spk:
                        new_spk = speaker
                    
                    # 如果输出目录为空，或者看起来是基于旧说话人的默认目录，则更新
                    if not new_out_dir or not str(new_out_dir).strip() or \
                       (spk_val and new_out_dir == _default_train_dir(spk_val)):
                        new_out_dir = _default_train_dir(speaker)
                return new_out_dir, new_spk

            train_data_path.change(
                _on_train_data_change,
                inputs=[train_data_path, train_output_path, train_speaker_name],
                outputs=[train_output_path, train_speaker_name],
            )

            def _on_speaker_change(spk_val: str, train_data_val: str, out_dir_val: str, test_base_val: str):
                spk = spk_val.strip() if spk_val else ""
                new_train_data = train_data_val
                new_out_dir = out_dir_val
                new_test_base = test_base_val
                if spk:
                    # 说话人变了，如果其他字段还是默认值或基于旧说话人，则同步更新
                    if not new_train_data or not str(new_train_data).strip():
                        new_train_data = _default_prepared_jsonl_path(spk)
                    if not new_out_dir or not str(new_out_dir).strip():
                        new_out_dir = _default_train_dir(spk)
                    if not new_test_base or not str(new_test_base).strip():
                        new_test_base = _default_train_dir(spk)
                return new_train_data, new_out_dir, new_test_base

            train_speaker_name.change(
                _on_speaker_change,
                inputs=[train_speaker_name, train_data_path, train_output_path, test_base_dir],
                outputs=[train_data_path, train_output_path, test_base_dir],
            )

            def _on_train_output_path_change(out_dir_val: str, spk_val: str):
                speaker = _infer_speaker_name(out_dir_val)
                # 如果用户手动改了输出路径，尝试同步更新说话人名称
                if speaker and (not spk_val or not str(spk_val).strip() or speaker != spk_val):
                    return speaker
                return spk_val

            train_output_path.change(
                _on_train_output_path_change,
                inputs=[train_output_path, train_speaker_name],
                outputs=[train_speaker_name],
            )

            def _on_test_base_dir_change(base_dir_val: str, spk_val: str):
                speaker = _infer_speaker_name(base_dir_val)
                # 切换测试检查点目录时，自动更新测试说话人
                if speaker and (not spk_val or not str(spk_val).strip() or speaker != spk_val):
                    return speaker
                return spk_val

            test_base_dir.change(
                _on_test_base_dir_change,
                inputs=[test_base_dir, test_speaker],
                outputs=[test_speaker],
            )
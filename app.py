# coding=utf-8
# Qwen3-TTS Gradio Demo with Multi-Model Support
# Supports: Voice Design, Voice Clone (Base), TTS (CustomVoice), Fine-tuning
import gc
import os
import time
from typing import Any, Dict

import gradio as gr
import numpy as np
import torch
from ttd_fastapi_utils import SmartModel

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from finetuning.ui import build_finetuning_tab
from finetuning.asr_client import auto_asr

loaded_models: Dict[tuple, SmartModel] = {}
model_last_used: Dict[tuple, float] = {}
MAX_MODELS = int(os.environ.get("QWEN_TTS_MAX_MODELS", "1"))

# Model size options
MODEL_SIZES = ["0.6B", "1.7B"]

# Speaker and language choices for CustomVoice model
SPEAKERS = [
    "Aiden", "Dylan", "Eric", "Ono_Anna", "Ryan", "Serena", "Sohee", "Uncle_Fu", "Vivian"
]
LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Russian"]


def _repo_id(model_type: str, model_size: str) -> str:
    return f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}"


def _memory_info() -> str:
    if not torch.cuda.is_available():
        return "CUDA not available"
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    return f"GPU Memory: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB"


def cleanup_old_models(keep_key: tuple | None = None) -> None:
    global loaded_models, model_last_used

    while len(loaded_models) > 0 and len(loaded_models) >= MAX_MODELS:
        if keep_key is not None and keep_key in loaded_models:
            return
        oldest_key = min(model_last_used.keys(), key=lambda k: model_last_used[k])
        if keep_key is not None and oldest_key == keep_key:
            return
        print(f"[app] Unloading model: {oldest_key}")
        
        # Stop the auto-unload monitor and force unload
        wrapper = loaded_models[oldest_key]
        wrapper.stop()
        wrapper.unload()
        
        del loaded_models[oldest_key]
        del model_last_used[oldest_key]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[app] {_memory_info()}")


def get_model(model_type: str, model_size: str):
    global loaded_models, model_last_used

    key = (model_type, model_size)
    if key in loaded_models:
        model_last_used[key] = time.time()
        return loaded_models[key].get()

    cleanup_old_models(keep_key=key)

    print(f"[app] Preparing model loader: {key}")
    print(f"[app] {_memory_info()}")

    def loader():
        from qwen_tts import Qwen3TTSModel

        hf_token = os.environ.get("HF_TOKEN") or None
        repo_id = _repo_id(model_type, model_size)

        attn_impl = None
        try:
            import flash_attn  # noqa: F401

            attn_impl = "flash_attention_2"
        except Exception:
            attn_impl = None

        kwargs: Dict[str, Any] = dict(
            device_map="cuda",
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
        if hf_token:
            kwargs["token"] = hf_token

        print(f"[app] Loading model from {repo_id}...")
        model = Qwen3TTSModel.from_pretrained(repo_id, **kwargs)
        print(f"[app] Model loaded: {key}")
        print(f"[app] {_memory_info()}")
        return model

    # Initialize with SmartModel, default 2h timeout
    wrapper = SmartModel(loader, timeout_seconds=7200)
    loaded_models[key] = wrapper
    model_last_used[key] = time.time()
    
    return wrapper.get()


def _normalize_audio(wav, eps=1e-12, clip=True):
    """Normalize audio to float32 in [-1, 1] range."""
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


def _audio_to_tuple(audio):
    """Convert Gradio audio input to (wav, sr) tuple."""
    if audio is None:
        return None

    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr

    return None


def generate_voice_design(text, language, voice_description):
    """Generate speech using Voice Design model (1.7B only)."""
    if not text or not text.strip():
        return None, "错误：请输入文本。"
    if not voice_description or not voice_description.strip():
        return None, "错误：请输入声音描述。"

    try:
        tts = get_model("VoiceDesign", "1.7B")
        wavs, sr = tts.generate_voice_design(
            text=text.strip(),
            language=language,
            instruct=voice_description.strip(),
            max_new_tokens=2048,
        )
        return (sr, wavs[0]), "语音设计生成成功！"
    except Exception as e:
        return None, f"错误：{type(e).__name__}: {e}"


def generate_voice_clone(ref_audio, ref_text, target_text, language, use_xvector_only, model_size):
    """Generate speech using Base (Voice Clone) model."""
    if not target_text or not target_text.strip():
        return None, "错误：请输入目标文本。"

    audio_tuple = _audio_to_tuple(ref_audio)
    if audio_tuple is None:
        return None, "错误：请上传参考音频。"

    if not use_xvector_only and (not ref_text or not ref_text.strip()):
        return None, "错误：未启用“仅使用 x-vector”时，必须提供参考文本。"

    try:
        tts = get_model("Base", model_size)
        wavs, sr = tts.generate_voice_clone(
            text=target_text.strip(),
            language=language,
            ref_audio=audio_tuple,
            ref_text=ref_text.strip() if ref_text else None,
            x_vector_only_mode=use_xvector_only,
            max_new_tokens=2048,
        )
        return (sr, wavs[0]), "声音克隆生成成功！"
    except Exception as e:
        return None, f"错误：{type(e).__name__}: {e}"


def generate_custom_voice(text, language, speaker, instruct, model_size):
    """Generate speech using CustomVoice model."""
    if not text or not text.strip():
        return None, "错误：请输入文本。"
    if not speaker:
        return None, "错误：请选择说话人。"

    try:
        tts = get_model("CustomVoice", model_size)
        wavs, sr = tts.generate_custom_voice(
            text=text.strip(),
            language=language,
            speaker=speaker.lower().replace(" ", "_"),
            instruct=instruct.strip() if instruct else None,
            max_new_tokens=2048,
        )
        return (sr, wavs[0]), "语音生成成功！"
    except Exception as e:
        return None, f"错误：{type(e).__name__}: {e}"


def get_memory_info():
    """获取当前显存使用情况"""
    return _memory_info()


def build_ui():
    """Build Gradio UI"""
    with gr.Blocks(title="Qwen3-TTS Demo") as demo:
        gr.Markdown(
            """
# Qwen3-TTS 语音合成演示

统一的文本转语音演示，包含四种强大模式：
- **语音设计**：使用自然语言描述创建自定义声音
- **声音克隆**：从参考音频克隆任意声音
- **文本转语音**：使用预设说话人和可选风格指令生成语音
- **语音微调**：使用自己的数据训练自定义语音模型

由阿里巴巴 Qwen 团队基于 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 构建。
"""
        )

        with gr.Tabs():
            # Tab 1: Voice Design (Default, 1.7B only)
            with gr.Tab("语音设计"):
                gr.Markdown("### 使用自然语言创建自定义声音")
                with gr.Row():
                    with gr.Column(scale=2):
                        design_text = gr.Textbox(
                            label="要合成的文本",
                            lines=4,
                            placeholder="请输入要转换为语音的文本...",
                            value="它在最上面的抽屉里……等等，是空的？不可能！我肯定把它放在那儿了！"
                        )
                        design_language = gr.Dropdown(
                            label="语言",
                            choices=LANGUAGES,
                            value="Auto",
                            interactive=True,
                        )
                        design_instruct = gr.Textbox(
                            label="声音描述",
                            lines=3,
                            placeholder="描述你想要的声音特征...",
                            value="用难以置信的语气说话，但声音中开始出现一丝恐慌。"
                        )
                        design_btn = gr.Button("生成自定义声音", variant="primary")

                    with gr.Column(scale=2):
                        design_audio_out = gr.Audio(label="生成的音频", type="numpy")
                        design_status = gr.Textbox(label="状态", lines=2, interactive=False)

                design_btn.click(
                    generate_voice_design,
                    inputs=[design_text, design_language, design_instruct],
                    outputs=[design_audio_out, design_status],
                )

            # Tab 2: Voice Clone (Base)
            with gr.Tab("声音克隆"):
                gr.Markdown("### 从参考音频克隆声音")
                with gr.Row():
                    with gr.Column(scale=2):
                        clone_ref_audio = gr.Audio(
                            label="参考音频（上传要克隆的声音样本）",
                            type="numpy",
                        )
                        clone_ref_text = gr.Textbox(
                            label="参考文本（参考音频的转录文本）",
                            lines=2,
                            placeholder="请输入参考音频中说的确切文本...",
                        )
                        clone_xvector = gr.Checkbox(
                            label="仅使用 x-vector（无需参考文本，但质量较低）",
                            value=False,
                        )

                    with gr.Column(scale=2):
                        clone_target_text = gr.Textbox(
                            label="目标文本（用克隆声音合成的文本）",
                            lines=4,
                            placeholder="请输入你希望克隆声音说的文本...",
                        )
                        with gr.Row():
                            clone_language = gr.Dropdown(
                                label="语言",
                                choices=LANGUAGES,
                                value="Auto",
                                interactive=True,
                            )
                            clone_model_size = gr.Dropdown(
                                label="模型大小",
                                choices=MODEL_SIZES,
                                value="1.7B",
                                interactive=True,
                            )
                        clone_btn = gr.Button("克隆并生成", variant="primary")

                with gr.Row():
                    clone_audio_out = gr.Audio(label="生成的音频", type="numpy")
                    clone_status = gr.Textbox(label="状态", lines=2, interactive=False)

            def auto_transcribe_ref_audio(audio_data):
                """当参考音频变化时自动转录文本"""
                if audio_data is None:
                    return ""
                
                try:
                    # 使用统一的音频转换逻辑
                    audio_tuple = _audio_to_tuple(audio_data)
                    if audio_tuple is None:
                        return ""
                    
                    wav, sr = audio_tuple
                    
                    import tempfile
                    import os
                    
                    # 将 numpy 音频数据保存为临时 wav 文件
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                        import soundfile as sf
                        # _audio_to_tuple 返回 (wav, sr)，sf.write 期望 (file, data, samplerate)
                        sf.write(tmp_file.name, wav, sr)
                        tmp_path = tmp_file.name
                    
                    try:
                        # 调用 ASR
                        transcribed_text = auto_asr(tmp_path)
                        return transcribed_text.strip()
                    finally:
                        # 清理临时文件
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                except Exception as e:
                    print(f"ASR 转录失败: {e}")
                    return ""

            clone_btn.click(
                generate_voice_clone,
                inputs=[clone_ref_audio, clone_ref_text, clone_target_text, clone_language, clone_xvector, clone_model_size],
                outputs=[clone_audio_out, clone_status],
            )
            
            # 当参考音频变化时自动转录
            clone_ref_audio.change(
                auto_transcribe_ref_audio,
                inputs=[clone_ref_audio],
                outputs=[clone_ref_text],
            )

            # Tab 3: TTS (CustomVoice)
            with gr.Tab("文本转语音"):
                gr.Markdown("### 使用预设说话人进行文本转语音")
                with gr.Row():
                    with gr.Column(scale=2):
                        tts_text = gr.Textbox(
                            label="要合成的文本",
                            lines=4,
                            placeholder="请输入要转换为语音的文本...",
                            value="你好！欢迎使用文本转语音系统。这是我们 TTS 功能的演示。"
                        )
                        with gr.Row():
                            tts_language = gr.Dropdown(
                                label="语言",
                                choices=LANGUAGES,
                                value="English",
                                interactive=True,
                            )
                            tts_speaker = gr.Dropdown(
                                label="说话人",
                                choices=SPEAKERS,
                                value="Ryan",
                                interactive=True,
                            )
                        with gr.Row():
                            tts_instruct = gr.Textbox(
                                label="风格指令（可选）",
                                lines=2,
                                placeholder="例如：用欢快且充满活力的语气说话",
                            )
                            tts_model_size = gr.Dropdown(
                                label="模型大小",
                                choices=MODEL_SIZES,
                                value="1.7B",
                                interactive=True,
                            )
                        tts_btn = gr.Button("生成语音", variant="primary")

                    with gr.Column(scale=2):
                        tts_audio_out = gr.Audio(label="生成的音频", type="numpy")
                        tts_status = gr.Textbox(label="状态", lines=2, interactive=False)

                tts_btn.click(
                    generate_custom_voice,
                    inputs=[tts_text, tts_language, tts_speaker, tts_instruct, tts_model_size],
                    outputs=[tts_audio_out, tts_status],
                )

            # Tab 4: Fine-tuning (entry point)
            build_finetuning_tab(cleanup_models=cleanup_old_models)

        # 添加显存监控
        with gr.Row():
            memory_info = gr.Textbox(
                label="显存状态",
                value=get_memory_info(),
                interactive=False
            )
            refresh_btn = gr.Button("刷新显存信息")
        
        refresh_btn.click(
            get_memory_info,
            outputs=[memory_info]
        )

        gr.Markdown(
            """
---

**注意**：此演示使用智能内存管理以防止 OOM 错误。
一次只加载一个模型。切换时模型会自动卸载。
"""
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=8000,
        share=False
    )

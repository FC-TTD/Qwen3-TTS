from __future__ import annotations

from io import BytesIO

import soundfile as sf

import app as demo_app
from fusion.base_proxy import BaseProxy
from fusion.config import FusionSettings
from fusion import local_runtime


def _wav_response_to_tuple(content: bytes):
    wav, sr = sf.read(BytesIO(content), dtype="float32")
    return (sr, wav)


def build_ui(proxy: BaseProxy, settings: FusionSettings):
    def generate_clone(audio_data, ref_text, target_text, language, use_xvector_only):
        if not target_text or not target_text.strip():
            return None, "错误：请输入目标文本。"

        audio_tuple = demo_app._audio_to_tuple(audio_data)
        if audio_tuple is None:
            return None, "错误：请上传参考音频。"

        wav, sr = audio_tuple
        buffer = BytesIO()
        sf.write(buffer, wav, sr, format="WAV")
        buffer.seek(0)

        try:
            content = proxy.synthesize_sync(
                text=target_text.strip(),
                language=language,
                audio_bytes=buffer.getvalue(),
                filename="reference.wav",
                content_type="audio/wav",
                ref_text=ref_text.strip() if ref_text else None,
                x_vector_only_mode=use_xvector_only,
                remove_silence=False,
                speed=1.0,
                expected_duration=None,
                postprocess=True,
                lufs=-23.0,
                temperature=0.9,
                top_p=1.0,
                top_k=50,
                repetition_penalty=1.05,
                max_new_tokens=2048,
            )
        except RuntimeError as exc:
            return None, f"错误：{exc}"

        return _wav_response_to_tuple(content), "声音克隆生成成功！"

    def auto_transcribe(_audio_data):
        return ""

    return demo_app.build_ui(
        generate_voice_design_fn=local_runtime.generate_voice_design,
        generate_voice_clone_fn=generate_clone,
        generate_custom_voice_fn=local_runtime.generate_custom_voice,
        auto_transcribe_fn=auto_transcribe,
        get_memory_info_fn=local_runtime.get_memory_info,
        list_checkpoints_fn=local_runtime.list_checkpoints,
        infer_speaker_fn=local_runtime.infer_speaker,
        enable_finetuning=settings.enable_finetuning,
        enable_memory_info=True,
        enable_auto_asr=settings.enable_auto_asr,
        enable_checkpoint_selector=settings.enable_checkpoint_selector,
    )

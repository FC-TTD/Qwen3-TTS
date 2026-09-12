from fastapi import HTTPException
import app as demo_app
from fusion.config import FusionSettings
from fusion.runtime import FusionRuntime


def build_callbacks(runtime: FusionRuntime):
    def generate_clone(audio_data, ref_text, target_text, language, use_xvector_only):
        audio = demo_app._audio_to_tuple(audio_data)
        try:
            result = runtime.synthesize(text=target_text, language=language, mode="voice_clone",
                                        ref_audio=audio, ref_text=ref_text,
                                        x_vector_only_mode=use_xvector_only)
            return result, "声音克隆生成成功！"
        except (ValueError, HTTPException) as exc:
            return None, f"错误：{getattr(exc, 'detail', str(exc))}"

    def generate_design(text, language, instruct):
        try:
            return runtime.synthesize(text=text, language=language, mode="voice_design",
                                      instruct=instruct), "语音设计生成成功！"
        except (ValueError, HTTPException) as exc:
            return None, f"错误：{getattr(exc, 'detail', str(exc))}"

    def generate_custom(text, language, speaker, instruct, model_size, checkpoint_path):
        try:
            return runtime.synthesize(text=text, language=language, mode="custom_voice",
                                      speaker=speaker, instruct=instruct, model_size=model_size,
                                      checkpoint_path=checkpoint_path), "语音生成成功！"
        except (ValueError, HTTPException) as exc:
            return None, f"错误：{getattr(exc, 'detail', str(exc))}"

    return generate_clone, generate_design, generate_custom


def build_ui(runtime: FusionRuntime, settings: FusionSettings):
    clone, design, custom = build_callbacks(runtime)
    # Fine-tuning and ASR remain disabled in the inference-only fusion service.
    return demo_app.build_ui(
        generate_voice_design_fn=design,
        generate_voice_clone_fn=clone,
        generate_custom_voice_fn=custom,
        get_memory_info_fn=demo_app.get_memory_info,
        list_checkpoints_fn=demo_app._list_custom_voice_checkpoints,
        infer_speaker_fn=demo_app._infer_speaker_from_checkpoint,
        enable_finetuning=False, enable_memory_info=True, enable_auto_asr=False,
        enable_checkpoint_selector=settings.enable_checkpoint_selector,
    )

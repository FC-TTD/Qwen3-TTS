from __future__ import annotations

from typing import Optional

import app as demo_app


def generate_voice_design(text: str, language: str, voice_description: str):
    return demo_app.generate_voice_design(text, language, voice_description)


def generate_custom_voice(
    text: str,
    language: str,
    speaker: str,
    instruct: Optional[str],
    model_size: str,
    checkpoint_path: str,
):
    return demo_app.generate_custom_voice(text, language, speaker, instruct, model_size, checkpoint_path)


def get_memory_info() -> str:
    return demo_app.get_memory_info()


def list_checkpoints() -> list[str]:
    return demo_app._list_custom_voice_checkpoints()


def infer_speaker(checkpoint_path: str) -> Optional[str]:
    return demo_app._infer_speaker_from_checkpoint(checkpoint_path)


def shutdown() -> None:
    for key in list(demo_app.loaded_models.keys()):
        wrapper = demo_app.loaded_models.pop(key)
        try:
            wrapper.stop()
        finally:
            wrapper.unload()
            demo_app.model_last_used.pop(key, None)

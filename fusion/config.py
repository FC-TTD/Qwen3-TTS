from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class FusionSettings:
    host: str = os.environ.get("HOST", "0.0.0.0")
    port: int = int(os.environ.get("PORT", "8000"))
    base_api_url: str = os.environ.get("BASE_API_URL", "http://qwen-api-base:8000")
    request_timeout_seconds: float = float(os.environ.get("FUSION_REQUEST_TIMEOUT_SECONDS", "300"))
    enable_auto_asr: bool = os.environ.get("FUSION_ENABLE_AUTO_ASR", "false").lower() == "true"
    enable_finetuning: bool = os.environ.get("FUSION_ENABLE_FINETUNING", "false").lower() == "true"
    enable_checkpoint_selector: bool = os.environ.get("FUSION_ENABLE_CHECKPOINT_SELECTOR", "true").lower() == "true"


settings = FusionSettings()

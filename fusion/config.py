from dataclasses import dataclass
import os


@dataclass(frozen=True)
class FusionSettings:
    host: str = os.environ.get("HOST", "0.0.0.0")
    port: int = int(os.environ.get("PORT", "8000"))
    idle_seconds: int = int(os.environ.get("FUSION_IDLE_SECONDS", "7200"))
    max_pending: int = int(os.environ.get("FUSION_MAX_PENDING", "8"))
    queue_timeout: float = float(os.environ.get("FUSION_QUEUE_TIMEOUT_SECONDS", "300"))
    enable_auto_asr: bool = False
    enable_finetuning: bool = False
    enable_checkpoint_selector: bool = os.environ.get("FUSION_ENABLE_CHECKPOINT_SELECTOR", "true").lower() == "true"


settings = FusionSettings()

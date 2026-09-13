"""Shared legacy Base audio contract for all fusion modes."""
from io import BytesIO
import logging
import numpy as np
import soundfile as sf
import soxr
from ttd_fastapi_utils import eq, loudnorm, trim_silence
from ttd_fastapi_utils.speed_control import time_stretch_wav

logger = logging.getLogger(__name__)


def finish_audio(wav, sr, *, speed=1.0, expected_duration=None,
                 remove_silence=False, postprocess=True, lufs=-23.0):
    final_speed = float(speed)
    if expected_duration is not None:
        try:
            trimmed = trim_silence(wav, sr, min_silence_duration_ms=0)
        except Exception:
            logger.exception("Silence measurement failed")
            trimmed = wav
        duration = len(trimmed) / sr
        if abs(duration - expected_duration) / expected_duration > 0.05:
            factor = min(1.5, max(0.5, duration / expected_duration))
            final_speed = min(2.0, max(0.5, final_speed * factor))
    if final_speed != 1.0:
        # Do not silently return unmodified audio when requested timing fails.
        wav = time_stretch_wav(wav, sr, final_speed, allow_passthrough_on_failure=False)
    if remove_silence:
        wav = trim_silence(wav, sr)
    if postprocess:
        wav, _ = loudnorm(wav, sr, target_loudness=float(lufs))
        wav = eq(wav, sr)
    if sr != 48000:
        wav = soxr.resample(wav, sr, 48000)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.size == 0 or not np.isfinite(wav).all():
        raise RuntimeError("Model returned empty or non-finite audio")
    return 48000, wav


def encode_wav(audio):
    sr, wav = audio
    buffer = BytesIO()
    sf.write(buffer, wav, sr, format="WAV")
    return buffer.getvalue()

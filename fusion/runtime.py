"""One serialized GPU runtime shared by HTTP and Gradio requests."""
from contextlib import contextmanager
import math
import logging
import threading
import time
import traceback
from pathlib import Path

import torch
from fastapi import HTTPException
from ttd_fastapi_utils import SmartModel

from fusion.audio import finish_audio

LANGUAGES = ["auto", "chinese", "english", "japanese", "korean", "french",
             "german", "spanish", "portuguese", "russian", "italian"]
SPEAKERS = ["Aiden", "Dylan", "Eric", "Ono_Anna", "Ryan", "Serena", "Sohee", "Uncle_Fu", "Vivian"]
logger = logging.getLogger(__name__)

MODES = {"voice_clone": "Base", "voice_design": "VoiceDesign", "custom_voice": "CustomVoice"}


def validate_request(*, text, mode, language, model_size, ref_audio, ref_text,
                     x_vector_only_mode, instruct, speaker, speed, expected_duration,
                     lufs, temperature, top_p, top_k, repetition_penalty, max_new_tokens):
    if not text or not text.strip():
        raise ValueError("text must not be blank")
    if mode not in MODES:
        raise ValueError(f"unsupported mode: {mode}")
    if model_size not in {"0.6B", "1.7B"}:
        raise ValueError("model_size must be 0.6B or 1.7B")
    if language.strip().lower() not in LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")
    if mode == "voice_clone":
        if ref_audio is None:
            raise ValueError("voice_clone mode requires ref_audio")
        if not x_vector_only_mode and not (ref_text and ref_text.strip()):
            raise ValueError("ref_text is required unless x_vector_only_mode=true")
    if mode == "voice_design" and not (instruct and instruct.strip()):
        raise ValueError("voice_design mode requires instruct")
    if mode == "custom_voice" and not speaker:
        raise ValueError("custom_voice mode requires speaker")
    for name, value in [("speed", speed), ("temperature", temperature),
                        ("repetition_penalty", repetition_penalty)]:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and > 0")
    if expected_duration is not None and (not math.isfinite(expected_duration) or expected_duration <= 0):
        raise ValueError("expected_duration must be finite and > 0")
    if not math.isfinite(lufs) or not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("lufs must be finite and top_p must be in (0, 1]")
    if top_k < 0 or max_new_tokens <= 0:
        raise ValueError("top_k must be >= 0 and max_new_tokens must be > 0")


def load_model(key):
    from qwen_tts import Qwen3TTSModel
    kind, size, checkpoint = key
    model_path = checkpoint or f"Qwen/Qwen3-TTS-12Hz-{size}-{kind}"
    try:
        import flash_attn  # noqa: F401
        attention = "flash_attention_2"
    except ImportError:
        attention = "sdpa"
    return Qwen3TTSModel.from_pretrained(
        model_path, device_map="cuda:0", dtype=torch.bfloat16,
        attn_implementation=attention,
    )


def detached_error(error):
    """Keep public error semantics without retaining failed CUDA call frames."""
    if isinstance(error, HTTPException):
        replacement = HTTPException(error.status_code, error.detail, headers=error.headers)
    elif isinstance(error, ValueError):
        replacement = ValueError(str(error))
    else:
        replacement = RuntimeError(str(error))
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        pending.extend(e for e in (current.__cause__, current.__context__) if e is not None)
        traceback.clear_frames(current.__traceback__)
        current.__traceback__ = None
        current.__cause__ = None
        current.__context__ = None
    return replacement


class FusionRuntime:
    def __init__(self, *, loader=load_model, idle_seconds=7200, max_pending=8, queue_timeout=300):
        self.loader = loader
        self.idle_seconds = idle_seconds
        self.queue_timeout = queue_timeout
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(max_pending + 1)
        self._state_lock = threading.Lock()
        self._pending = 0
        self._busy = False
        self._manager = None
        self._key = None
        self._last_used = time.monotonic()
        self._stop = threading.Event()
        self._thread = None
        self.ready = False

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self.ready = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._monitor, daemon=True, name="FusionIdleUnload")
        self._thread.start()

    def _monitor(self):
        while not self._stop.wait(30):
            try:
                self.unload_if_idle()
            except Exception:
                logger.exception("Fusion idle unload failed")

    def unload_if_idle(self):
        if self.idle_seconds >= 0 and self._lock.acquire(blocking=False):
            try:
                if time.monotonic() - self._last_used >= self.idle_seconds:
                    self._unload()
            finally:
                self._lock.release()

    def status(self):
        with self._state_lock:
            return {"ready": self.ready, "busy": self._busy, "pending": self._pending,
                    "loaded_model": list(self._key) if self._key else None,
                    "self_contained": True}

    @contextmanager
    def _request(self):
        if not self.ready:
            raise HTTPException(503, "Fusion runtime is not ready")
        if not self._slots.acquire(blocking=False):
            raise HTTPException(429, "Fusion request queue is full", headers={"Retry-After": "5"})
        with self._state_lock:
            self._pending += 1
        acquired = False
        try:
            acquired = self._lock.acquire(timeout=self.queue_timeout)
            if not acquired:
                raise HTTPException(503, "Timed out waiting for the fusion GPU", headers={"Retry-After": "5"})
            if not self.ready:
                raise HTTPException(503, "Fusion runtime is shutting down")
            with self._state_lock:
                self._busy = True
            yield
        finally:
            with self._state_lock:
                self._pending -= 1
                if acquired:
                    self._busy = False
            if acquired:
                self._last_used = time.monotonic()
                self._lock.release()
            self._slots.release()

    def _unload(self):
        if self._manager is not None:
            self._manager.stop()
            self._manager.unload()
            self._manager = None
        with self._state_lock:
            self._key = None

    def unload(self):
        with self._request():
            self._unload()
        return self.status()

    def shutdown(self):
        self.ready = False
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        with self._lock:
            self._unload()

    def synthesize(self, *, text, language="auto", mode="voice_clone", ref_audio=None,
                   ref_text=None, x_vector_only_mode=False, instruct=None, speaker=None,
                   model_size="1.7B", checkpoint_path="default", speed=1.0,
                   expected_duration=None, remove_silence=False, postprocess=True,
                   lufs=-23.0, temperature=0.9, top_p=1.0, top_k=50,
                   repetition_penalty=1.05, max_new_tokens=2048):
        validate_request(text=text, mode=mode, language=language, model_size=model_size,
                         ref_audio=ref_audio, ref_text=ref_text, x_vector_only_mode=x_vector_only_mode,
                         instruct=instruct, speaker=speaker, speed=speed,
                         expected_duration=expected_duration, lufs=lufs, temperature=temperature,
                         top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty,
                         max_new_tokens=max_new_tokens)
        checkpoint = None
        if mode == "custom_voice" and checkpoint_path and checkpoint_path != "default":
            path = Path(checkpoint_path).resolve()
            if not path.is_relative_to(Path("/app/lora")) or not (path / "model.safetensors").is_file():
                raise ValueError("checkpoint_path must be a model checkpoint under /app/lora")
            checkpoint = str(path)
        # VoiceDesign only exists at 1.7B; preserve the prior fusion behavior.
        size = "1.7B" if mode == "voice_design" else model_size
        key = (MODES[mode], size, checkpoint)
        with self._request():
            if key != self._key:
                self._unload()
                self._manager = SmartModel(lambda: self.loader(key), timeout_seconds=-1)
            model = None
            failure = None
            try:
                model = self._manager.get()
                with self._state_lock:
                    self._key = key
                supported = model.get_supported_languages()
                language = language.strip().lower()
                if supported is not None and language not in {str(v).lower() for v in supported}:
                    raise ValueError(f"Unsupported language: {language}")
                params = dict(text=text.strip(), language=language, temperature=temperature,
                              top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty,
                              max_new_tokens=max_new_tokens)
                if mode == "voice_clone":
                    wavs, sr = model.generate_voice_clone(
                        **params, ref_audio=ref_audio, ref_text=ref_text,
                        x_vector_only_mode=x_vector_only_mode)
                elif mode == "voice_design":
                    wavs, sr = model.generate_voice_design(**params, instruct=instruct.strip())
                else:
                    wavs, sr = model.generate_custom_voice(
                        **params, speaker=speaker.strip().lower().replace(" ", "_"), instruct=instruct or None)
                result = finish_audio(wavs[0], sr, speed=speed, expected_duration=expected_duration,
                                      remove_silence=remove_silence, postprocess=postprocess, lufs=lufs)
            except Exception as exc:
                failure = detached_error(exc)
            finally:
                # clear_frames cannot clear this currently executing frame.
                model = None
            if failure is not None:
                self._unload()
                # Raise outside the except block: even `from None` inside it would
                # retain the original exception through implicit __context__.
                raise failure from None
            return result

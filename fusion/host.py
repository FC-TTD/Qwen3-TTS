from contextlib import asynccontextmanager
from io import BytesIO
import logging
from typing import Optional

import gradio as gr
import soundfile as sf
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from ttd_fastapi_utils import setup_cuda_health

from fusion.audio import encode_wav
from fusion.config import settings
from fusion.local_runtime import runtime
from fusion.runtime import LANGUAGES, SPEAKERS
from fusion.webui import build_ui

logger = logging.getLogger("qwen-fusion")


@asynccontextmanager
async def lifespan(app):
    runtime.start()
    try:
        yield
    finally:
        await run_in_threadpool(runtime.shutdown)


app = FastAPI(title="Qwen3-TTS Fusion", version="1.1.0",
              description="Self-contained Base, VoiceDesign and CustomVoice API + WebUI",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
# The CUDA monitor also reports device failures after lazy model loading.
setup_cuda_health(app, path="/health", ready_predicate=lambda: runtime.ready,
                  enable_default_home=False)


@app.get("/health/backends")
async def backend_health():
    return {"local": runtime.status()}


@app.post("/api/unload")
async def unload():
    return await run_in_threadpool(runtime.unload)


@app.get("/api/speakers")
async def speakers():
    return {"speakers": SPEAKERS}


@app.get("/languages")
async def languages():
    return {"languages": LANGUAGES}


@app.post("/api/tts")
async def api_tts(
    text: str = Form(...),
    language: str = Form("auto"),
    mode: str = Form("voice_clone"),
    ref_audio: Optional[UploadFile] = File(None),
    ref_text: Optional[str] = Form(None),
    x_vector_only_mode: bool = Form(False),
    instruct: Optional[str] = Form(None),
    speaker: Optional[str] = Form(None),
    model_size: str = Form("1.7B"),
    checkpoint_path: str = Form("default"),
    remove_silence: bool = Form(False),
    speed: float = Form(1.0),
    expected_duration: Optional[float] = Form(None),
    postprocess: bool = Form(True),
    lufs: float = Form(-23.0),
    temperature: float = Form(0.9),
    top_p: float = Form(1.0),
    top_k: int = Form(50),
    repetition_penalty: float = Form(1.05),
    max_new_tokens: int = Form(2048),
):
    params = dict(text=text, language=language, mode=mode, ref_text=ref_text,
                  x_vector_only_mode=x_vector_only_mode, instruct=instruct, speaker=speaker,
                  model_size=model_size, checkpoint_path=checkpoint_path,
                  remove_silence=remove_silence, speed=speed, expected_duration=expected_duration,
                  postprocess=postprocess, lufs=lufs, temperature=temperature, top_p=top_p,
                  top_k=top_k, repetition_penalty=repetition_penalty, max_new_tokens=max_new_tokens)
    try:
        audio = None
        if mode == "voice_clone" and ref_audio is not None:
            content = await ref_audio.read()
            try:
                wav, sr = await run_in_threadpool(sf.read, BytesIO(content), dtype="float32")
                if wav.size == 0 or not np.isfinite(wav).all():
                    raise ValueError("empty or non-finite reference audio")
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                audio = (wav, sr)
            except Exception as exc:
                raise HTTPException(400, "ref_audio must be a readable, nonempty audio file") from exc
        result = await run_in_threadpool(runtime.synthesize, ref_audio=audio, **params)
        content = await run_in_threadpool(encode_wav, result)
        return Response(content=content, media_type="audio/wav")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Fusion synthesis failed")
        raise HTTPException(500, "Internal server error") from exc


demo = build_ui(runtime, settings)
app = gr.mount_gradio_app(app, demo, path="/")

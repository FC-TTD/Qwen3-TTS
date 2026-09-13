from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from io import BytesIO
from typing import Optional

import gradio as gr
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from fusion.base_proxy import BaseProxy
from fusion.config import settings
from fusion import local_runtime
from fusion.webui import build_ui
import app as demo_app


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("qwen-fusion")

proxy = BaseProxy(settings.base_api_url, settings.request_timeout_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        local_runtime.shutdown()


app = FastAPI(
    title="Qwen3-TTS Fusion",
    description="Unified Gradio and API host for Base + Design + CustomVoice",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    base_status = await proxy.health()
    return {
        "ok": True,
        "base": base_status,
    }


@app.get("/health/backends")
async def backend_health():
    return {"base": await proxy.health()}


@app.get("/api/speakers")
async def get_speakers():
    return {"speakers": demo_app.SPEAKERS}


@app.get("/languages")
async def get_languages():
    return {"languages": [lang.lower() for lang in demo_app.LANGUAGES]}


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
    if mode == "voice_clone":
        return await proxy.synthesize(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
            remove_silence=remove_silence,
            speed=speed,
            expected_duration=expected_duration,
            postprocess=postprocess,
            lufs=lufs,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )

    if mode == "voice_design":
        if not instruct:
            raise HTTPException(status_code=400, detail="voice_design mode requires instruct")
        wav_result, status = local_runtime.generate_voice_design(text, language, instruct)
    elif mode == "custom_voice":
        if not speaker:
            raise HTTPException(status_code=400, detail="custom_voice mode requires speaker")
        wav_result, status = local_runtime.generate_custom_voice(text, language, speaker, instruct, model_size, checkpoint_path)
    else:
        raise HTTPException(status_code=400, detail=f"unsupported mode: {mode}")

    if wav_result is None:
        raise HTTPException(status_code=400, detail=status)

    sr, wav = wav_result
    import soundfile as sf

    buffer = BytesIO()
    sf.write(buffer, wav, sr, format="WAV")
    buffer.seek(0)
    return Response(content=buffer.read(), media_type="audio/wav")


demo = build_ui(proxy, settings)
app = gr.mount_gradio_app(app, demo, path="/")

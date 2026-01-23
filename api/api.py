#!/usr/bin/env python3
from contextlib import asynccontextmanager
from io import BytesIO
import logging
import os
import sys
import tempfile
import time
from typing import Optional, List, Dict

# 添加当前目录到 Python 路径，确保相对导入可用
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 限制 PyTorch 显存碎片化缓存
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import numpy as np
import soundfile as sf
import soxr
import torch
from ttd_fastapi_utils import (
    eq as _eq,
    loudnorm as _loudnorm,
    setup_cuda_health,
    trim_silence as _trim_silence,
)

from qwen_tts import Qwen3TTSModel

from lang import create_router as create_lang_router
from lang import setup_language_openapi
from lang import validate_language_or_400

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("qwen-api")
logging.getLogger("uvicorn.access").addFilter(
    lambda r: "/health" not in r.getMessage() and "/docs" not in r.getMessage()
)

# 全局变量
_model: Optional[Qwen3TTSModel] = None

# 配置参数
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = os.environ.get("DTYPE", "bfloat16")

def _get_torch_dtype(dtype_str: str):
    if dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "float16":
        return torch.float16
    return torch.float32

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    try:
        logger.info(f"Loading Qwen3-TTS model: {MODEL_ID} on {DEVICE}...")
        
        attn_impl = None
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
            logger.info("Using Flash Attention 2")
        except ImportError:
            logger.info("Flash Attention not found, using default attention")

        _model = Qwen3TTSModel.from_pretrained(
            MODEL_ID,
            device_map=DEVICE,
            dtype=_get_torch_dtype(DTYPE),
            attn_implementation=attn_impl
        )
        logger.info("Qwen3-TTS model loaded successfully")
        yield
    except Exception:
        logger.exception("Failed to initialize Qwen3-TTS model")
        raise

app = FastAPI(
    title="Qwen3-TTS API",
    description="Qwen3-TTS Voice Clone API Service",
    version="1.0.0",
    lifespan=lifespan,
)

setup_language_openapi(app, lambda: _model)
app.include_router(create_lang_router(lambda: _model))

# CUDA 健康检查
cuda_monitor = setup_cuda_health(
    app,
    path="/health",
    ready_predicate=lambda: _model is not None,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/tts")
async def api_tts(
    text: str = Form(...),
    ref_audio: UploadFile = File(...),
    ref_text: Optional[str] = Form(None),
    language: str = Form("auto"),
    x_vector_only_mode: bool = Form(False),
    remove_silence: bool = Form(False),
    postprocess: bool = Form(True),
    temperature: float = Form(0.9),
    top_p: float = Form(1.0),
    top_k: int = Form(50),
    repetition_penalty: float = Form(1.05),
    max_new_tokens: int = Form(2048),
):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    t0 = time.perf_counter()
    temp_ref = None
    
    try:
        # 保存参考音频到临时文件
        ref_data = await ref_audio.read()
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(ref_audio.filename or "")[1] or ".wav", delete=False) as f_ref:
            f_ref.write(ref_data)
            temp_ref = f_ref.name

        language_norm = validate_language_or_400(language, lambda: _model)

        logger.info(
            f"Generating TTS for text: {text[:50]}... (lang={language_norm}, xvec={x_vector_only_mode})"
        )

        # 执行推理
        # Qwen3TTSModel.generate_voice_clone 返回 (wavs, sample_rate)
        # wavs 是 List[np.ndarray]
        wavs, sr = await run_in_threadpool(
            _model.generate_voice_clone,
            text=text,
            language=language_norm,
            ref_audio=temp_ref,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )

        wav_np = wavs[0]
        
        # 后处理
        if remove_silence:
            try:
                wav_np = _trim_silence(wav_np, sr)
            except Exception:
                logger.exception("Silence trimming failed")

        if postprocess:
            try:
                wav_np, _ = _loudnorm(wav_np, sr)
                wav_np = _eq(wav_np, sr)
            except Exception:
                logger.exception("Post-processing failed")

        # 统一输出 48k
        if sr != 48000:
            wav_np = soxr.resample(wav_np, sr, 48000)
            sr = 48000

        # 导出为 WAV
        buffer = BytesIO()
        sf.write(buffer, wav_np, sr, format="WAV")
        buffer.seek(0)
        
        logger.info(f"TTS generation completed in {time.perf_counter() - t0:.3f}s")
        return Response(content=buffer.read(), media_type="audio/wav")

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        logger.exception("TTS generation failed")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        if temp_ref and os.path.exists(temp_ref):
            try:
                os.remove(temp_ref)
            except Exception:
                pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

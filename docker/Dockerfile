# syntax=docker/dockerfile:1.7-labs
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PYTHONPATH=/app \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
      git git-lfs libsndfile1 ffmpeg curl ca-certificates sox \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv --system-site-packages /opt/appenv
ENV VIRTUAL_ENV=/opt/appenv
ENV PATH="/opt/appenv/bin:${PATH}"

WORKDIR /app

COPY pyproject.toml LICENSE README.md ./
RUN mkdir -p qwen_tts && touch qwen_tts/__init__.py

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install "gradio>=5.0,<6.0" \
    && pip install .

# Install flash-attn separately as it might take time and requires hardware compatibility
# Removing cache mount to avoid [Errno 18] Invalid cross-device link when moving downloaded wheels
RUN FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE MAX_JOBS=4 pip install -U flash-attn --no-build-isolation

RUN find /opt/appenv -type d -name "__pycache__" -prune -exec rm -rf {} + || true \
    && rm -rf /root/.cache/pip || true

COPY qwen_tts ./qwen_tts
COPY app.py ./

EXPOSE 8000

# Default model is CustomVoice as it's a good balanced choice
CMD ["python", "app.py"]

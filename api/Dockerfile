FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PYTHONPATH=/app \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/root/.cache/uv

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
      git git-lfs libsndfile1 ffmpeg curl ca-certificates sox \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN --mount=type=cache,target=/root/.cache/uv \
    curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

# 复制项目配置文件
COPY pyproject.toml LICENSE README.md ./
RUN mkdir -p qwen_tts && touch qwen_tts/__init__.py

# 创建虚拟环境并安装主项目依赖
RUN --mount=type=cache,target=/root/.cache/uv \
    /root/.local/bin/uv venv /app/venv && \
    . /app/venv/bin/activate && \
    /root/.local/bin/uv pip install -e .

# 安装 API 额外依赖，包括私有仓库的 ttd-fastapi-utils
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,target=/tmp/requirements.txt,source=api/requirements.txt \
    . /app/venv/bin/activate && \
    /root/.local/bin/uv pip install -r /tmp/requirements.txt \
    --index-url http://pypi-server/simple/ \
    --trusted-host pypi-server \
    --extra-index-url https://pypi.org/simple

# 安装 flash-attn
# 使用 MAX_JOBS=4 限制编译并发，避免内存耗尽
RUN --mount=type=cache,target=/root/.cache/uv \
    . /app/venv/bin/activate && \
    FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE MAX_JOBS=4 \
    /root/.local/bin/uv pip install -U flash-attn --no-build-isolation

# 复制源码
COPY qwen_tts ./qwen_tts
COPY api ./api

EXPOSE 8000

# 启动命令 - 使用虚拟环境的 python
CMD ["/app/venv/bin/python", "api/api.py"]
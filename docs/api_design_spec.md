# Qwen3-TTS API 服务设计文档 (实际部署版)

## 1. 概述 (Overview)

本文档描述了 `Qwen3-TTS` API 服务的实际实施情况。该服务将 Qwen3-TTS 的 **声音克隆 (Voice Clone)** 能力封装为 RESTful API，已成功部署在 Docker Swarm 集群 (`ttd-worker`)，集成标准化的音频后处理与健康监控，并使用 **uv** 进行高效的依赖管理。

## 2. 架构 (Architecture)

### 2.1 核心组件

- **框架**: FastAPI (Python)
- **运行时**: Uvicorn (ASGI)
- **依赖管理**: uv (虚拟环境 + 缓存优化)
- **容器化**: Docker (基于 PyTorch 官方镜像)
- **编排**: Docker Swarm (部署于 `ttd-worker` 节点)
- **反向代理**: Caddy (通过 Docker Swarm labels 配置)
- **健康监控**: ttd_fastapi_utils (CUDA 健康检查)

### 2.2 目录结构

```text
Qwen3-TTS/
├── api/
│   ├── api.py           # FastAPI 主程序 (203行)
│   ├── Dockerfile       # API 容器定义 (使用 uv 虚拟环境)
│   ├── stack.yml        # Docker Swarm stack 配置 (双实例)
│   └── requirements.txt # API 额外依赖 (12个包)
├── deploy.sh            # 统一部署入口 (Ansible + CI/CD)
├── ansible/
│   └── site.yml         # 部署配置 (Swarm + Portainer)
├── docs/
│   └── api_design_spec.md # 本设计文档
└── pyproject.toml       # 项目配置 (含可选依赖组)
```

### 2.3 部署架构

- **节点**: `ttd-worker` (GPU 节点)
- **实例**: 2个容器实例 (api-1, api-2)
- **GPU 分配**:
  - api-1: NVIDIA_VISIBLE_DEVICES=1
  - api-2: NVIDIA_VISIBLE_DEVICES=2
- **模型缓存**: `/TTD-Data/qwen3_tts/hf_cache`
- **网络**: Caddy 反向代理 + DNS 配置

## 3. API 规范 (API Specification)

### 3.1 端点 (Endpoints)

#### `GET /languages`

- **用途**: 获取支持的语言列表。
- **响应**: `200 OK` (返回语言代码与名称映射)。

#### `GET /health`

- **用途**: 健康检查，对接负载均衡器。
- **实现**: 使用 `ttd_fastapi_utils.setup_cuda_health`。
- **响应**: `200 OK` + `{"status":"healthy"}`

#### `POST /api/tts` (声音克隆)

- **用途**: 使用参考音频进行零样本语音合成 (Zero-shot Voice Cloning)。
- **请求参数**:
  - `text` (form, required): 要合成的文本
  - `ref_audio` (file, required): 参考音频文件
  - `ref_text` (form, optional): 参考音频对应的文本
  - `language` (form, default="Auto"): 语言代码
  - `x_vector_only_mode` (form, default=false): 仅使用 x-vector 模式
  - `remove_silence` (form, default=false): 移除静音
  - `speed` (form, default=1.0): 语速倍率，>1 更快更短，<1 更慢更长。实现由 `ttd-fastapi-utils==0.3.0` 的 `speed_control` 插件提供（SoX `sox tempo -s` 不变调变速）
  - `expected_duration` (form, optional): 期望有效发音时长(秒)。对齐反馈采用内部临时 trim(B-align)；如果偏差>5%则计算 `final_speed` 并在 API 层使用 `speed_control` 对生成音频做一次 time-stretch（不再二次推理）。与 remove_silence 正交
  - `postprocess` (form, default=true): 音频后处理
  - `temperature` (form, default=0.9): 生成温度
  - `top_p` (form, default=1.0): Top-p 采样
  - `top_k` (form, default=50): Top-k 采样
  - `repetition_penalty` (form, default=1.05): 重复惩罚
  - `max_new_tokens` (form, default=2048): 最大新 token 数
- **响应**: `200 OK` (WAV 音频文件) 或错误状态码
- **后处理**:
  - 静音修剪 (可选)
  - LUFS 标准化
  - 高频增强
  - 统一输出 48kHz

## 4. 技术实现 (Technical Implementation)

### 4.1 依赖管理 (uv)

- **虚拟环境**: 使用 `uv venv /app/venv` 创建独立环境
- **依赖安装**:
  - 主项目: `uv pip install -e .`
  - API 依赖: `uv pip install -r api/requirements.txt`
  - 特殊包: `uv pip install -U flash-attn --no-build-isolation`
- **缓存优化**: `UV_CACHE_DIR=/root/.cache/uv` + Docker 挂载缓存

### 4.2 模型配置

- **模型**: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- **设备**: CUDA (自动检测)
- **数据类型**: bfloat16 (可配置)
- **注意力机制**: Flash Attention 2 (可选，回退到默认)

### 4.3 音频处理

- **输入格式**: 任意音频格式 (通过 soundfile 解码)
- **输出格式**: WAV 48kHz
- **后处理链**:
  1. 静音修剪 (librosa-based)
  2. LUFS 标准化 (-23 LUFS)
  3. 高频增强
  4. 采样率转换 (soxr)

### 4.4 健康监控

- **CUDA 监控**: `ttd_fastapi_utils.setup_cuda_health`
- **就绪检查**: 模型加载完成
- **自动重启**: CUDA 错误时容器自动重启
- **日志抑制**: 过滤 `/health` 和 `/docs` 访问日志

## 5. 部署配置 (Deployment)

### 5.1 Docker Swarm

- **Stack 名称**: `qwen-tts`
- **服务数量**: 2个实例 (api-1, api-2)
- **节点约束**: `node.hostname == ttd-worker`
- **GPU 分配**: 独占 GPU 1 和 GPU 2
- **重启策略**: 失败时最多重启 3 次

### 5.2 网络配置

- **网络**: `caddy` (外部网络)
- **DNS**: 8.8.8.8, 1.1.1.1 (解决 huggingface.co 连接)
- **路由**: Caddy 反向代理
- **健康检查**: 30秒间隔，5秒超时

### 5.3 存储挂载

- **模型缓存**: `/TTD-Data/qwen3_tts/hf_cache:/root/.cache/huggingface`
- **缓存内容**: HuggingFace 模型文件和配置

## 6. 环境变量 (Environment Variables)

| 变量名 | 默认值 | 说明 |
| -------- | -------- | ------ |
| `MODEL_ID` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | HuggingFace 模型 ID |
| `DEVICE` | `cuda` | 计算设备 (cuda/cpu) |
| `DTYPE` | `bfloat16` | 数据类型 |
| `PORT` | `8000` | 服务端口 |
| `TZ` | `Asia/Shanghai` | 时区 |
| `NVIDIA_VISIBLE_DEVICES` | - | GPU 可见性 (1 或 2) |
| `TTD_SPEED_CONTROL_BYPASS_SOX` | - | speed_control 插件旁路开关：设置为非空值时绕过 SoX 并原样透传音频（排障/环境缺少 SoX） |

## 7. 错误处理 (Error Handling)

### 7.1 HTTP 状态码

- `200`: 成功
- `400`: 请求参数错误
- `500`: 内部服务器错误
- `503`: 模型未初始化

### 7.2 异常处理

- **模型加载失败**: 应用启动失败，容器重启
- **推理失败**: 详细错误日志，返回 500
- **CUDA 错误**: 自动重启容器
- **文件处理错误**: 临时文件自动清理

## 8. 性能优化 (Performance)

### 8.1 并发处理

- **异步框架**: FastAPI + Uvicorn
- **线程池**: CPU 密集型操作使用 `run_in_threadpool`
- **GPU 利用率**: 双实例并行服务

### 8.2 缓存策略

- **Docker 层缓存**: 依赖安装缓存
- **uv 缓存**: 包下载和编译缓存
- **模型缓存**: HuggingFace 本地缓存

### 8.3 资源管理

- **内存管理**: 临时文件及时清理
- **GPU 内存**: 模型加载后固定占用
- **文件 I/O**: 内存缓冲区处理

## 9. 监控与日志 (Monitoring & Logging)

### 9.1 日志配置

- **级别**: INFO
- **格式**: 时间戳 + 模块 + 级别 + 消息
- **过滤**: 抑制健康检查访问日志
- **输出**: 控制台 (容器日志)

### 9.2 性能指标

- **请求耗时**: 每个请求的完整处理时间
- **模型加载时间**: 启动时模型加载耗时
- **错误率**: 各类异常的统计

## 10. 部署状态 (Current Status)

### 10.1 服务状态

✅ **已部署**: Docker Swarm 集群正常运行  
✅ **健康检查**: 两个实例均通过健康检查  
✅ **模型加载**: Qwen3-TTS 1.7B 模型成功加载  
✅ **GPU 可用**: CUDA 设备正常工作  
✅ **网络连接**: DNS 配置解决外网访问  

### 10.2 访问信息

- **服务地址**: `http://qwen-api` (Caddy 反向代理)
- **健康检查**: `http://qwen-api/health`
- **API 文档**: `http://qwen-api/docs`
- **直接访问**: 容器内部 `localhost:8000`

### 10.3 已知问题

- **外部访问**: Caddy 路由配置需要手动刷新
- **Flash Attention**: 可选依赖，未安装时自动回退
- **模型下载**: 首次启动需要下载模型文件

---

**文档最后更新**: 2026-01-24 (基于实际部署情况)

# TTS 智能变速控制方案总结

## 背景描述

在 Qwen3-TTS 项目中，我们为 FastAPI `/api/tts` 端点实现了智能变速控制功能，包括：

1. **手动语速调整**：通过 `speed` 参数（float，默认 1.0）直接控制生成语音的播放速度。
2. **自适应时长对齐**：通过 `expected_duration` 参数（可选 float）实现基于 B-align（临时去静音后的时长）的反馈式速度调整，当实际时长与期望偏差超过 5% 时自动重新推理。
3. **后处理兼容**：`remove_silence` 参数独立控制最终输出的静音裁剪，不影响自适应反馈的计算基准。

### 初始实现与问题

- **初始方案**：在模型层对离散 codec codes 做时间维重采样（`speed != 1.0` 时对 `talker_codes_list`/`talker_hidden_states_list` 进行索引采样）。
- **线上反馈**：实测出现“重音和杂音”，音质明显下降。
- **根因分析**：
  - 离散 codes 重采样导致跳变/伪影。
  - 我们尝试的“12Hz v2 连续 latent 插值”绕开了官方 `decoder.chunked_decode`，进一步损失音质。

### 技术演进路径

1. **离散 codes 重采样**（模型层）→ 发现重音/杂音
2. **连续 latent 插值**（12Hz v2 量化后特征）→ 音质下降
3. **原生解码 + 插件化波形层 time-stretch**（ttd-fastapi-utils speed_control）→ 音质恢复

## 最终方案

### 核心设计

1. **保持原生解码音质**：统一使用 `speech_tokenizer.decode()`（内部走 `decoder.chunked_decode`）。
2. **波形层不变调 time-stretch**：在裁掉 prompt 段后，调用 `ttd-fastapi-utils==0.3.0` 提供的 `speed_control` 插件实现变速。
3. **自适应时长对齐**：`expected_duration` 逻辑不变，通过调整 `speed` 参数反馈到 `speed_control` 插件。
4. **兼容性**：仅对生成段做 stretch，保留 prompt 段音质；支持所有 tokenizer 类型（不局限于 12Hz）。

### 实现要点

- 使用插件函数：
  - `time_stretch_wav(wav, sr, speed, allow_passthrough_on_failure=True)`
  - `apply_speed_to_wav_list(wavs, sr, speed, allow_passthrough_on_failure=True)`
- 插件内部使用 **SoX `sox tempo -s`** 做不变调变速。
- 环境变量：
  - `TTD_SPEED_CONTROL_BYPASS_SOX`：设置为任意非空值时，绕过 SoX 并原样透传音频（用于环境缺少 SoX 或排障场景）。

### 关键优势

- **音质保真**：不侵入模型解码，保持官方解码路径。
- **不变调**：插件内部使用 SoX `tempo -s` 保证音高不变。
- **通用性**：适用于所有 tokenizer 类型。
- **可运维**：支持 `TTD_SPEED_CONTROL_BYPASS_SOX` 旁路，便于排障与环境降级。

## 集成到 ttd_fastapi_utils 的建议

### 目标

为无法在模型层实现变速的 TTS 服务提供**通用、高质量、不变调**的变速能力，作为对原生 LLM 模型不足的补充。

### 设计要点

- **通用性**：支持任意 TTS 服务返回的 wav（np.ndarray, sr）。
- **高质量**：优先用 SoX `tempo -s`（音质最佳），备选 FFmpeg `atempo`（更通用）。
- **不变调**：确保 pitch 不变。
- **易用性**：提供装饰器或后处理函数，与现有 FastAPI 端点无缝集成。

### 建议实现

本能力已由 `ttd-fastapi-utils==0.3.0` 原生提供：

- `time_stretch_wav(wav, sr, speed, allow_passthrough_on_failure=True)`
- `apply_speed_to_wav_list(wavs, sr, speed, allow_passthrough_on_failure=True)`

建议在各 TTS 服务中统一复用该插件能力，避免重复实现与音质回归风险。

#### 2. FastAPI 装饰器/中间件（可选）

```python
# ttd_fastapi_utils/fastapi_speed.py
from functools import wraps
from typing import Callable
from fastapi import Request
from .speed_control import apply_speed_to_wav_list

def with_speed_control(
    speed_param_name: str = "speed",
    wav_field_name: str = "wav",
    sr_field_name: str = "sr",
):
    """
    装饰器：对 FastAPI 端点返回的 wav 做变速后处理。
    
    用法：
    @with_speed_control()
    async def tts_endpoint(...):
        return {"wav": wav, "sr": sr}
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 从 request 或 kwargs 取 speed
            speed = 1.0
            if "request" in kwargs:
                request: Request = kwargs["request"]
                speed = float(await request.form().get(speed_param_name, 1.0))
            elif speed_param_name in kwargs:
                speed = float(kwargs[speed_param_name])
            
            result = await func(*args, **kwargs)
            
            if speed != 1.0 and wav_field_name in result:
                wavs = result[wav_field_name]
                sr = result.get(sr_field_name, 48000)
                if isinstance(wavs, list):
                    result[wav_field_name] = apply_speed_to_wav_list(wavs, sr, speed)
                else:
                    result[wav_field_name] = time_stretch_wav(wavs, sr, speed)
            return result
        return wrapper
    return decorator
```

#### 3. 使用示例

```python
# 在任意 FastAPI TTS 服务中
from ttd_fastapi_utils.speed_control import time_stretch_wav

@router.post("/tts")
async def tts(text: str, speed: float = 1.0):
    wav, sr = my_tts_model.synthesize(text)  # 原生模型输出
    if speed != 1.0:
        wav = time_stretch_wav(wav, sr, speed)
    return Response(wav.tobytes(), media_type="audio/wav")
```

### 优势总结

- **音质保真**：不侵入模型解码，使用专业音频工具。
- **通用适配**：任何返回 wav 的 TTS 服务都能直接套用。
- **不变调**：SoX `tempo -s` 保证音高不变。
- **容错**：SoX 不可用时自动回退到原始音频（passthrough）。

### 集成建议

- 将 `speed_control.py` 作为独立模块发布到 `ttd_fastapi_utils`。
- 在文档中强调“模型层变速优先，波形层变速作为补充”。
- 可选提供装饰器以简化现有端点改造。

### 系统依赖

**重要**：`ttd_fastapi_utils.speed_control` 强依赖 SoX 音频处理工具，必须在系统中安装：

```bash
# Ubuntu/Debian
sudo apt-get install sox

# CentOS/RHEL
sudo yum install sox

# Alpine
sudo apk add sox

# macOS
brew install sox
```

- SoX 提供高质量的 `tempo -s` 算法，保证音高不变的 time-stretch
- 若 SoX 不可用，插件会抛出异常并回退到原始音频（passthrough）
- 可通过环境变量 `TTD_SPEED_CONTROL_BYPASS_SOX=1` 强制跳过 SoX 处理

**注意**：当前版本（0.3.0）暂未实现 FFmpeg fallback，仅支持 SoX。若 SoX 不可用，会返回原始音频。

这样，未来遇到无法在模型层实现变速的 TTS 项目时，直接引入 `ttd_fastapi_utils.speed_control` 即可获得高质量变速能力。

## 经验教训

1. **模型层变速风险**：离散 codec 重采样容易引入伪影，需谨慎评估。
2. **保持官方解码路径**：绕过官方 `chunked_decode` 等核心流程可能导致音质下降。
3. **专业音频工具优势**：SoX 的 time-stretch 算法经过长期优化，音质可靠。
4. **分层设计**：将变速能力从模型层分离到后处理层，提高系统可维护性和通用性。

---

*文档生成时间：2026-01-27*  
*适用场景：TTS 服务变速控制、音质优化、架构设计参考*

# Qwen3-TTS API 接口文档

## 概述

Qwen3-TTS API 提供语音克隆和文本转语音服务，支持多语言和高质量的语音合成.

**基础信息**

- **Base URL**: `http://qwen-api` (可配置)
- **API版本**: v1.0.0
- **认证**: 无需认证 (根据实际部署情况调整)
- **响应格式**: 音频文件 (WAV) 或 JSON (错误信息)

## 端点列表

### 1. 健康检查

#### GET /health

检查API服务健康状态。

**请求**

```http
GET /health
```

**响应**

```json
{
  "status": "healthy"
}
```

**状态码**

- `200` - 服务正常
- `503` - 服务不可用

#### TRACE /health

获取详细的健康追踪信息。

**请求**

```http
TRACE /health
```

**响应**

```json
{
  "status": "healthy",
  "details": {
    "timestamp": "2024-01-24T03:45:00Z",
    "version": "1.0.0",
    "uptime": 3600
  }
}
```

### 2. 语言支持

#### GET /languages

获取支持的语言列表。

**请求**

```http
GET /languages
```

**响应**

```json
{
  "languages": [
    "auto",
    "chinese", 
    "english",
    "french",
    "german",
    "italian",
    "japanese",
    "korean"
  ]
}
```

**状态码**

- `200` - 成功返回语言列表

### 3. 语音合成

#### POST /api/tts

执行文本转语音合成。

**请求**

```http
POST /api/tts
Content-Type: multipart/form-data
```

**请求参数 (multipart/form-data)**

| 参数名 | 类型 | 必需 | 默认值 | 描述 |
|--------|------|------|--------|------|
| text | string | ✅ | - | 要合成的文本 |
| ref_audio | binary | ✅ | - | 参考音频文件 (WAV格式) |
| ref_text | string | 条件必需 | - | 参考音频对应的文本 (当x_vector_only_mode=False时必需) |
| language | string | ❌ | "auto" | 目标语言 |
| x_vector_only_mode | boolean | ❌ | false | 是否仅使用X向量模式 |
| remove_silence | boolean | ❌ | true | 是否移除静音片段 |
| postprocess | boolean | ❌ | true | 是否进行后处理 |
| temperature | float | ❌ | 0.9 | 生成温度 (0.1-1.0) |
| top_p | float | ❌ | 1.0 | Top-p采样 (0.1-1.0) |
| top_k | integer | ❌ | 50 | Top-k采样 (1-100) |
| repetition_penalty | float | ❌ | 1.05 | 重复惩罚系数 |
| max_new_tokens | integer | ❌ | 2048 | 最大生成令牌数 |

**支持的语言代码**

- `auto` - 自动检测
- `chinese` - 中文
- `english` - 英文  
- `french` - 法文
- `german` - 德文
- `italian` - 意大利文
- `japanese` - 日文
- `korean` - 韩文

**响应**

- **成功**: WAV音频文件 (Content-Type: audio/wav)
- **失败**: JSON错误信息

**成功响应头**

```http
Content-Type: audio/wav
Content-Length: [音频文件大小]
```

**错误响应**

```json
{
  "detail": "错误描述信息"
}
```

**状态码**

- `200` - 合成成功
- `400` - 请求参数错误
- `422` - 参数验证失败
- `500` - 服务器内部错误

## 使用示例

### cURL 示例

```bash
# 基本语音合成
curl -X POST \
  -F "text=Hello, world!" \
  -F "ref_audio=@reference.wav" \
  -F "ref_text=Hello, world!" \
  http://qwen-api/api/tts \
  --output output.wav

# 指定语言合成
curl -X POST \
  -F "text=Bonjour le monde" \
  -F "ref_audio=@reference.wav" \
  -F "ref_text=Bonjour le monde" \
  -F "language=french" \
  http://qwen-api/api/tts \
  --output french_output.wav

# X向量模式
curl -X POST \
  -F "text=Hello, world!" \
  -F "ref_audio=@reference.wav" \
  -F "x_vector_only_mode=true" \
  http://qwen-api/api/tts \
  --output x_vector_output.wav
```

### Python SDK 示例

```python
import requests

class QwenTTSClient:
    def __init__(self, base_url="http://qwen-api"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def synthesize(self, text, ref_audio_path, ref_text=None, **kwargs):
        """语音合成"""
        files = {
            'text': (None, text),
            'ref_audio': open(ref_audio_path, 'rb')
        }
        
        # 添加可选参数
        if ref_text is not None:
            files['ref_text'] = (None, ref_text)
        
        for key, value in kwargs.items():
            if value is not None:
                files[key] = (None, str(value))
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/tts", 
                files=files
            )
            response.raise_for_status()
            return response.content  # WAV音频数据
        finally:
            files['ref_audio'][1].close()
    
    def health_check(self):
        """健康检查"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_languages(self):
        """获取支持的语言"""
        response = self.session.get(f"{self.base_url}/languages")
        response.raise_for_status()
        return response.json()

# 使用示例
client = QwenTTSClient()

# 健康检查
health = client.health_check()
print(f"API状态: {health['status']}")

# 语音合成
audio_data = client.synthesize(
    text="Hello, world!",
    ref_audio_path="reference.wav",
    ref_text="Hello, world!",
    language="english"
)

# 保存音频文件
with open("output.wav", "wb") as f:
    f.write(audio_data)
```

### JavaScript SDK 示例

```javascript
class QwenTTSClient {
    constructor(baseUrl = 'http://qwen-api') {
        this.baseUrl = baseUrl.replace(/\/$/, '');
    }
    
    async synthesize(text, refAudioFile, options = {}) {
        const formData = new FormData();
        formData.append('text', text);
        formData.append('ref_audio', refAudioFile);
        
        // 添加可选参数
        if (options.refText) {
            formData.append('ref_text', options.refText);
        }
        
        const params = [
            'language', 'x_vector_only_mode', 'remove_silence', 
            'postprocess', 'temperature', 'top_p', 'top_k', 
            'repetition_penalty', 'max_new_tokens'
        ];
        
        params.forEach(param => {
            if (options[param] !== undefined) {
                formData.append(param, options[param]);
            }
        });
        
        const response = await fetch(`${this.baseUrl}/api/tts`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Synthesis failed');
        }
        
        return await response.blob(); // WAV音频数据
    }
    
    async healthCheck() {
        const response = await fetch(`${this.baseUrl}/health`);
        if (!response.ok) throw new Error('Health check failed');
        return await response.json();
    }
    
    async getLanguages() {
        const response = await fetch(`${this.baseUrl}/languages`);
        if (!response.ok) throw new Error('Failed to get languages');
        return await response.json();
    }
}

// 使用示例
const client = new QwenTTSClient();

// 语音合成
document.getElementById('synthesize-btn').onclick = async () => {
    const text = document.getElementById('text-input').value;
    const refAudio = document.getElementById('ref-audio').files[0];
    
    try {
        const audioBlob = await client.synthesize(text, refAudio, {
            refText: text,
            language: 'english'
        });
        
        // 播放音频
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        audio.play();
    } catch (error) {
        console.error('合成失败:', error);
    }
};
```

## 错误处理

### 常见错误码

| 状态码 | 错误类型 | 描述 | 解决方案 |
| -------- | ---------- | ------ | ---------- |
| 400 | Bad Request | 请求参数错误 | 检查请求参数格式 |
| 422 | Validation Error | 参数验证失败 | 检查必需参数是否缺失 |
| 500 | Internal Server Error | 服务器内部错误 | 稍后重试或联系技术支持 |

### 错误响应格式

```json
{
  "detail": "具体的错误描述信息"
}
```

### 常见错误场景

1. **缺少必需参数**

```json
{
  "detail": "ref_text is required when x_vector_only_mode=False"
}
```

1. **不支持的语言**

```json
{
  "detail": {
    "error": "Unsupported language: xyz",
    "supported": ["auto", "chinese", "english", "french", ...]
  }
}
```

1. **音频格式错误**

```json
{
  "detail": "Invalid audio format"
}
```

## 性能考虑

### 请求限制

- 最大文件大小: 50MB (参考音频)
- 最大文本长度: 1000字符
- 请求超时: 60秒

### 优化建议

1. 使用适当的参考音频 (清晰、无噪音)
2. 文本长度控制在合理范围内
3. 合理设置生成参数以平衡质量和速度
4. 实现客户端缓存以减少重复请求

## SDK 开发指南

### 核心功能实现

1. **HTTP客户端封装**
   - 支持multipart/form-data请求
   - 处理二进制音频响应
   - 错误处理和重试机制

2. **参数验证**
   - 必需参数检查
   - 参数类型和范围验证
   - 语言代码验证

3. **音频处理**
   - WAV格式支持
   - 音频数据缓存
   - 流式处理 (如支持)

4. **异步支持**
   - 支持async/await模式
   - 并发请求控制
   - 进度回调

### 推荐的SDK结构

```fs
qwen-tts-sdk/
├── src/
│   ├── client.py          # 主客户端类
│   ├── exceptions.py      # 异常定义
│   ├── models.py          # 数据模型
│   └── utils.py           # 工具函数
├── tests/                 # 单元测试
├── examples/              # 使用示例
└── README.md              # SDK文档
```

### 版本兼容性

- API版本: v1.0.0
- SDK版本: 建议语义化版本控制
- 向后兼容: 保持主版本号内的兼容性

## 更新日志

### v1.0.0 (2024-01-24)

- 初始版本发布
- 支持基本语音合成功能
- 多语言支持
- 健康检查端点

---

**技术支持**: 如有问题，请联系开发团队或查看API文档更新。

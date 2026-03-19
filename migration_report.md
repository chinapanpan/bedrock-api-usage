# Anthropic Claude API -> AWS Bedrock Claude API 迁移差异报告

> 生成日期: 2026-03-19
> 覆盖模型: Claude Opus 4.6, Sonnet 4.6, Opus 4.5, Sonnet 4.5, Haiku 4.5

---

## 目录

1. [概览与迁移路径选择](#1-概览与迁移路径选择)
2. [认证与连接方式](#2-认证与连接方式)
3. [模型 ID 映射](#3-模型-id-映射)
4. [Messages API 核心参数差异](#4-messages-api-核心参数差异)
5. [Tool Use 差异](#5-tool-use-差异)
6. [Extended Thinking 差异](#6-extended-thinking-差异)
7. [Prompt Caching 差异](#7-prompt-caching-差异)
8. [Streaming 差异](#8-streaming-差异)
9. [Vision (图像) 差异](#9-vision-图像-差异)
10. [PDF 支持差异](#10-pdf-支持差异)
11. [Citations 差异](#11-citations-差异)
12. [Batch Processing 差异](#12-batch-processing-差异)
13. [Structured Outputs 差异](#13-structured-outputs-差异)
14. [Built-in Tools 差异](#14-built-in-tools-差异)
15. [Anthropic 独有功能 (Bedrock 不支持)](#15-anthropic-独有功能-bedrock-不支持)
16. [Bedrock 独有功能](#16-bedrock-独有功能)
17. [Claude Agent SDK 迁移](#17-claude-agent-sdk-迁移)
18. [完整迁移 Checklist](#18-完整迁移-checklist)

---

## 1. 概览与迁移路径选择

### Bedrock 提供三种 API 调用方式

| API | 说明 | 推荐场景 |
|-----|------|----------|
| **InvokeModel** | 使用 Anthropic 原生 Messages 格式 | **从 Anthropic API 迁移时首选**，参数格式最接近 |
| **Converse** | AWS 统一跨模型接口，参数命名不同 | 需要跨模型供应商统一调用时 |
| **Responses** | OpenAI 兼容格式 | 需要 OpenAI 兼容或服务端工具调用时 |

**迁移建议**: 使用 **Anthropic Bedrock SDK** (`pip install "anthropic[bedrock]"`) + InvokeModel，可以保持代码改动最小。SDK 内部调用 InvokeModel API，参数格式与 Anthropic 直接 API 几乎一致。

### 两种 SDK 选择

```python
# 方式1: Anthropic Bedrock SDK (推荐 - 改动最小)
from anthropic import AnthropicBedrock
client = AnthropicBedrock(aws_region="us-west-2")

# 方式2: boto3 (需要手动处理 JSON)
import boto3
client = boto3.client('bedrock-runtime', region_name='us-west-2')
```

---

## 2. 认证与连接方式

### 差异对比

| 方面 | Anthropic 直接 API | AWS Bedrock |
|------|-------------------|-------------|
| 认证方式 | `x-api-key` HTTP Header | AWS IAM SigV4 签名 / Bearer Token |
| API 版本 | `anthropic-version` HTTP Header: `2023-06-01` | `anthropic_version` 在 request body: `bedrock-2023-05-31` |
| Endpoint | `https://api.anthropic.com/v1/messages` | `bedrock-runtime.<region>.amazonaws.com` |
| Beta 功能 | `anthropic-beta` HTTP Header | `anthropic_beta` 在 request body (数组) |

### 迁移代码示例

**Anthropic 直接 API (迁移前)**:
```python
from anthropic import Anthropic

client = Anthropic(api_key="sk-xxx")

message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)
```

**Bedrock - Anthropic SDK (迁移后)**:
```python
from anthropic import AnthropicBedrock

# 在 EC2 上有 IAM Role，无需显式传凭证
client = AnthropicBedrock(aws_region="us-west-2")

message = client.messages.create(
    model="us.anthropic.claude-sonnet-4-6",   # 变更: 模型 ID 格式不同
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)
```

**Bedrock - boto3 原生调用**:
```python
import boto3, json

client = boto3.client('bedrock-runtime', region_name='us-west-2')

body = json.dumps({
    "anthropic_version": "bedrock-2023-05-31",  # 变更: 必须在 body 中指定
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello"}]
})

response = client.invoke_model(
    body=body,
    modelId="us.anthropic.claude-sonnet-4-6"    # 变更: 模型 ID 在 API 参数中
)
result = json.loads(response['body'].read())
```

> **注意**: 使用 Anthropic Bedrock SDK 时，`anthropic_version` 由 SDK 自动处理，无需手动设置。

---

## 3. 模型 ID 映射

| Anthropic Model ID | Bedrock Base Model ID | Global ID (推荐) | Regional ID (US) |
|--------------------|----------------------|-------------------|------------------|
| `claude-opus-4-6` | `anthropic.claude-opus-4-6-v1` | `global.anthropic.claude-opus-4-6-v1` | `us.anthropic.claude-opus-4-6-v1` |
| `claude-sonnet-4-6` | `anthropic.claude-sonnet-4-6` | `global.anthropic.claude-sonnet-4-6` | `us.anthropic.claude-sonnet-4-6` |
| `claude-opus-4-5` / `claude-opus-4-5-20251101` | `anthropic.claude-opus-4-5-20251101-v1:0` | `global.anthropic.claude-opus-4-5-20251101-v1:0` | `us.anthropic.claude-opus-4-5-20251101-v1:0` |
| `claude-sonnet-4-5` / `claude-sonnet-4-5-20250929` | `anthropic.claude-sonnet-4-5-20250929-v1:0` | `global.anthropic.claude-sonnet-4-5-20250929-v1:0` | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` |
| `claude-haiku-4-5` / `claude-haiku-4-5-20251001` | `anthropic.claude-haiku-4-5-20251001-v1:0` | `global.anthropic.claude-haiku-4-5-20251001-v1:0` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` |

### 端点类型说明

- **Global** (`global.`): 动态路由到可用区域，**无溢价**，推荐用于大多数场景
- **Regional** (`us.`/`eu.`/`jp.`/`apac.`): 指定区域路由，**+10% 费用**，用于数据驻留合规
- **Base ID** (无前缀): 直接路由到当前 region

### 模型 ID 映射辅助函数

```python
MODEL_ID_MAP = {
    "claude-opus-4-6":     "global.anthropic.claude-opus-4-6-v1",
    "claude-sonnet-4-6":   "global.anthropic.claude-sonnet-4-6",
    "claude-opus-4-5":     "global.anthropic.claude-opus-4-5-20251101-v1:0",
    "claude-sonnet-4-5":   "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-haiku-4-5":    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
}

def to_bedrock_model_id(anthropic_model: str) -> str:
    """将 Anthropic 模型 ID 转换为 Bedrock 模型 ID"""
    # 处理带日期版本的 ID
    for key, value in MODEL_ID_MAP.items():
        if anthropic_model.startswith(key):
            return value
    raise ValueError(f"Unknown model: {anthropic_model}")
```

---

## 4. Messages API 核心参数差异

### 4.1 使用 Anthropic Bedrock SDK 时的参数差异

使用 `AnthropicBedrock` SDK 时，**大部分参数保持不变**，主要差异是：

| 参数 | Anthropic | Bedrock (Anthropic SDK) | 说明 |
|------|-----------|------------------------|------|
| `model` | `claude-sonnet-4-6` | `us.anthropic.claude-sonnet-4-6` | 必须使用 Bedrock 模型 ID |
| 其他参数 | - | - | SDK 自动处理格式转换 |

### 4.2 使用 boto3 InvokeModel 时的参数差异

| 参数 | Anthropic 直接 API | Bedrock InvokeModel | 差异说明 |
|------|-------------------|---------------------|----------|
| `model` | 在 request body 中 | 在 `invoke_model(modelId=...)` 参数中，**不在 body 中** | 位置不同 |
| `anthropic_version` | HTTP Header `anthropic-version: 2023-06-01` | Body 字段 `"anthropic_version": "bedrock-2023-05-31"` | **值和位置都不同** |
| `anthropic_beta` | HTTP Header `anthropic-beta: xxx` | Body 字段 `"anthropic_beta": ["xxx"]` | 位置不同，格式为数组 |
| `max_tokens` | body 中 | body 中 | **相同** |
| `messages` | body 中 | body 中 | **相同** |
| `system` | body 中 | body 中 | **相同** |
| `temperature` | body 中，默认 1.0 | body 中，默认 1 | **相同** |
| `top_p` | body 中 | body 中 | **相同**，但 Sonnet 4.5/Haiku 4.5 不能同时设 temperature 和 top_p |
| `top_k` | body 中 | body 中 | **相同** |
| `stop_sequences` | body 中 | body 中 | **相同** |
| `stream` | body 中 `"stream": true` | 使用不同的 API: `invoke_model_with_response_stream()` | **方式不同** |
| `tools` | body 中 | body 中 | **相同** |
| `tool_choice` | body 中 | body 中 | **相同** |
| `thinking` | body 中 | body 中 | **相同** (详见第6节) |
| `metadata` | body 中 | body 中 | **相同** |
| `output_config` | body 中 | body 中 (需 beta header) | effort 需要 `effort-2025-11-24` beta |
| `service_tier` | body 中 `"auto"/"standard_only"` | **不支持此参数** | Bedrock 有自己的 serviceTier 机制 |
| `container` | body 中 | **不支持** | Bedrock 无此功能 |
| `inference_geo` | body 中 | **不支持** | Bedrock 用模型 ID 前缀控制 |
| `cache_control` | body 中 | body 中 | **相同** (InvokeModel 时) |

### 4.3 使用 Converse API 时的参数映射

如果选择 Converse API，参数命名完全不同：

| Anthropic 参数 | Converse API 参数 | 说明 |
|---------------|-------------------|------|
| `model` (body) | `modelId` (API 参数) | 位置和命名不同 |
| `max_tokens` | `inferenceConfig.maxTokens` | camelCase |
| `temperature` | `inferenceConfig.temperature` | 嵌套在 inferenceConfig |
| `top_p` | `inferenceConfig.topP` | camelCase |
| `top_k` | `additionalModelRequestFields.top_k` | 需要通过额外字段传递 |
| `stop_sequences` | `inferenceConfig.stopSequences` | camelCase |
| `system` (string) | `system` (ContentBlock 数组) | 格式不同 |
| `tools` | `toolConfig.tools` | 嵌套结构完全不同 |
| `tool_choice` | `toolConfig.toolChoice` | 结构不同 |
| `cache_control` | `cachePoint` ContentBlock | **语法完全不同** |

```python
# Anthropic 直接 API
{
    "system": "You are helpful.",
    "max_tokens": 1024,
    "temperature": 0.7,
    "top_k": 250
}

# Bedrock Converse API 等效写法
{
    "system": [{"text": "You are helpful."}],
    "inferenceConfig": {
        "maxTokens": 1024,
        "temperature": 0.7
    },
    "additionalModelRequestFields": {
        "top_k": 250
    }
}
```

### 4.4 Response 差异

| 字段 | Anthropic | Bedrock InvokeModel | Bedrock Converse |
|------|-----------|---------------------|-----------------|
| `stop_reason` | `end_turn`/`max_tokens`/`stop_sequence`/`tool_use` | 同左 + `refusal` + `model_context_window_exceeded` | `stopReason` (camelCase) |
| `usage.input_tokens` | 有 | 有 | `usage.inputTokens` |
| `usage.output_tokens` | 有 | 有 | `usage.outputTokens` |
| `usage.cache_creation_input_tokens` | 有 | 有 | `usage.cacheWriteInputTokensCount` |
| `usage.cache_read_input_tokens` | 有 | 有 | `usage.cacheReadInputTokensCount` |
| `content` | 数组 | 数组 (相同) | `output.message.content` (嵌套更深) |

---

## 5. Tool Use 差异

### 5.1 InvokeModel API (差异最小)

使用 InvokeModel/Anthropic Bedrock SDK 时，Tool Use 格式与 Anthropic 直接 API **基本相同**：

```python
# 两个平台的 tool 定义格式完全相同
tools = [{
    "name": "get_weather",
    "description": "Get current weather for a location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"}
        },
        "required": ["location"]
    }
}]

# Anthropic 直接 API
client = Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-6",
    tools=tools,
    tool_choice={"type": "auto"},
    max_tokens=1024,
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}]
)

# Bedrock (Anthropic SDK) - 只需改 client 和 model
client = AnthropicBedrock(aws_region="us-west-2")
response = client.messages.create(
    model="us.anthropic.claude-sonnet-4-6",  # 只需改这里
    tools=tools,                              # 完全相同
    tool_choice={"type": "auto"},             # 完全相同
    max_tokens=1024,
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}]
)
```

**差异点**:
- `tool_choice` 选项 `"none"`: Anthropic API 支持，**Bedrock 不支持** (Bedrock 仅支持 `auto`/`any`/`tool`)
- `disable_parallel_tool_use`: Anthropic API 支持，**Bedrock 文档中未提及**，可能不受支持
- `strict: true` (Structured Outputs for tools): Anthropic API 和 Bedrock 均支持，Vertex 不支持
- Tool 定义中 `"type": "custom"` 在 Bedrock 上是**可选字段** (Anthropic API 上也是可选)

### 5.2 Converse API Tool Use (差异较大)

```python
# Anthropic 格式
{
    "tools": [{
        "name": "get_weather",
        "description": "Get weather",
        "input_schema": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    }]
}

# Converse API 格式 - 结构完全不同
{
    "toolConfig": {
        "tools": [{
            "toolSpec": {
                "name": "get_weather",
                "description": "Get weather",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"]
                    }
                }
            }
        }]
    }
}
```

**Tool Result 格式差异**:

```python
# Anthropic 格式
{
    "role": "user",
    "content": [{
        "type": "tool_result",
        "tool_use_id": "toolu_xxx",
        "content": "Sunny, 25°C",
        "is_error": False
    }]
}

# Converse API 格式
{
    "role": "user",
    "content": [{
        "toolResult": {
            "toolUseId": "toolu_xxx",
            "content": [{"json": {"weather": "Sunny", "temp": "25°C"}}],
            "status": "success"  # "success" | "error"
        }
    }]
}
```

---

## 6. Extended Thinking 差异

### 6.1 基本参数对比

| 方面 | Anthropic 直接 API | Bedrock |
|------|-------------------|---------|
| 启用方式 | `thinking.type = "enabled"` | **相同** |
| `budget_tokens` | 最小 1024，须 < max_tokens | **相同** |
| Adaptive thinking | `thinking.type = "adaptive"` | **支持** (Opus 4.6 推荐) |
| Display options | `"summarized"` / `"omitted"` | **部分差异** (见下文) |
| temperature 限制 | 启用时必须为 1 | **相同** |
| Interleaved thinking beta | `anthropic-beta: interleaved-thinking-2025-05-14` | `anthropic_beta: ["interleaved-thinking-2025-05-14"]` (body 内) |

### 6.2 关键差异: Summarized Thinking

| 方面 | Anthropic | Bedrock |
|------|-----------|---------|
| Claude 4 模型默认行为 | 返回 summarized thinking | **相同** |
| 获取 full thinking | 可直接请求 | **需要联系 AWS 账户团队开通** |
| Dev mode beta | `dev-full-thinking-2025-05-14` | **相同** (在 body 的 anthropic_beta 中) |

### 6.3 超时配置 (Bedrock 特有问题)

```python
# Bedrock 重要: AWS SDK 默认超时仅 1 分钟
# Extended thinking 场景必须手动增加超时!
import botocore.config

config = botocore.config.Config(read_timeout=3600)  # 1小时
client = boto3.client('bedrock-runtime', config=config, region_name='us-west-2')
```

> **关键注意**: Anthropic SDK 自动处理超时，但 boto3 默认 1 分钟会导致 thinking 请求超时失败。使用 `AnthropicBedrock` SDK 时也建议确认超时设置。

### 6.4 迁移代码示例

```python
# Anthropic 直接 API
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[{"role": "user", "content": "Solve this complex math problem..."}]
)

# Bedrock (Anthropic SDK) - 几乎相同
response = bedrock_client.messages.create(
    model="us.anthropic.claude-sonnet-4-6",  # 只需改模型 ID
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[{"role": "user", "content": "Solve this complex math problem..."}]
)

# Bedrock (boto3) - 需要手动构建 body
body = json.dumps({
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 16000,
    "thinking": {"type": "enabled", "budget_tokens": 10000},
    "messages": [{"role": "user", "content": "Solve this complex math problem..."}]
})
response = client.invoke_model(body=body, modelId="us.anthropic.claude-sonnet-4-6")
```

---

## 7. Prompt Caching 差异

### 7.1 支持情况对比

| 方面 | Anthropic | Bedrock |
|------|-----------|---------|
| 5 分钟 TTL | 支持 | **支持** |
| 1 小时 TTL | 支持 | **部分支持** (仅 Opus 4.5, Sonnet 4.5, Haiku 4.5 支持) |
| 自动缓存 (Automatic caching) | 支持 (顶层 `cache_control` 参数) | **不支持** (需手动在 content block 上添加) |
| 最大检查点数 | 4 | 4 |
| Lookback window | 20 blocks | 20 blocks |
| **Opus 4.6 / Sonnet 4.6 支持** | 支持 | **截至目前未在 Bedrock 缓存支持列表中** |

### 7.2 最小 Token 要求对比

| 模型 | Anthropic | Bedrock |
|------|-----------|---------|
| Opus 4.6 | 4096 | **截至目前未在支持列表中** |
| Sonnet 4.6 | 2048 | **截至目前未在支持列表中** |
| Opus 4.5 | 4096 | 4096 |
| Sonnet 4.5 | 1024 | 1024 |
| Haiku 4.5 | 4096 | 4096 |

> **注意**: Opus 4.6 和 Sonnet 4.6 是最新模型，Bedrock 的 Prompt Caching 支持列表可能尚未更新。实际使用时请测试验证。

### 7.3 InvokeModel 缓存语法 (基本相同)

```python
# Anthropic 直接 API - 使用 cache_control
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "You are an expert on a very long topic...",
            "cache_control": {"type": "ephemeral"}  # 默认 5m TTL
        }
    ],
    messages=[{"role": "user", "content": "Question?"}]
)

# Bedrock InvokeModel - 格式相同 (通过 Anthropic SDK)
response = bedrock_client.messages.create(
    model="us.anthropic.claude-sonnet-4-6",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "You are an expert on a very long topic...",
            "cache_control": {"type": "ephemeral"}  # 相同语法
        }
    ],
    messages=[{"role": "user", "content": "Question?"}]
)
```

### 7.4 Converse API 缓存语法 (差异很大)

```python
# Anthropic 格式: cache_control 附加在 content block 上
{
    "system": [{
        "type": "text",
        "text": "Long system prompt...",
        "cache_control": {"type": "ephemeral", "ttl": "5m"}
    }]
}

# Converse API 格式: 使用独立的 cachePoint content block
{
    "system": [
        {"text": "Long system prompt..."},
        {"cachePoint": {"type": "default", "ttl": "5m"}}  # 独立的缓存点块
    ]
}
```

### 7.5 自动缓存的 Workaround

Anthropic 直接 API 支持顶层 `cache_control` 参数，自动在最后一个可缓存块上设置缓存。Bedrock 不支持此功能。

**Workaround**: 手动在每次请求的最后一个可缓存块上添加 `cache_control`:

```python
def add_cache_to_last_block(messages):
    """手动模拟自动缓存: 在最后一个用户消息的最后一个内容块上添加缓存"""
    for msg in reversed(messages):
        if msg["role"] == "user" and isinstance(msg["content"], list):
            msg["content"][-1]["cache_control"] = {"type": "ephemeral"}
            break
    return messages
```

---

## 8. Streaming 差异

### 8.1 对比

| 方面 | Anthropic 直接 API | Bedrock |
|------|-------------------|---------|
| 启用方式 | body 中 `"stream": true` | 调用不同 API: `invoke_model_with_response_stream()` |
| 协议 | SSE (Server-Sent Events) | SSE (通过 EventStream) |
| 事件类型 | message_start, content_block_start, delta, stop 等 | **相同** (InvokeModel 时) |
| SDK 辅助 | `client.messages.stream()` | `bedrock_client.messages.stream()` |

### 8.2 迁移代码示例

```python
# Anthropic 直接 API
with client.messages.stream(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Tell me a story"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

# Bedrock (Anthropic SDK) - 用法完全相同
with bedrock_client.messages.stream(
    model="us.anthropic.claude-sonnet-4-6",  # 只改模型 ID
    max_tokens=1024,
    messages=[{"role": "user", "content": "Tell me a story"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

# Bedrock (boto3) - 需要手动解析 EventStream
response = client.invoke_model_with_response_stream(
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Tell me a story"}]
    }),
    modelId="us.anthropic.claude-sonnet-4-6"
)

for event in response['body']:
    chunk = json.loads(event['chunk']['bytes'])
    if chunk['type'] == 'content_block_delta':
        if chunk['delta']['type'] == 'text_delta':
            print(chunk['delta']['text'], end="", flush=True)
```

---

## 9. Vision (图像) 差异

### 9.1 对比

| 方面 | Anthropic 直接 API | Bedrock InvokeModel | Bedrock Converse |
|------|-------------------|---------------------|-----------------|
| base64 图像 | 支持 | **支持** | **支持** |
| URL 图像 | 支持 | **不支持** | **不支持** |
| File API 图像 | 支持 (Beta) | **不支持** | **不支持** |
| S3 图像 | 不支持 | 不支持 | **支持** |
| 最大图像大小 | 5 MB | 3.75 MB | 3.75 MB |
| 最大像素 | 8000x8000 | 8000x8000 | 8000x8000 |
| 格式 | JPEG, PNG, GIF, WebP | **相同** | **相同** |

### 9.2 URL 图像的 Workaround

Bedrock 不支持 URL 图像源。需要先下载再转 base64：

```python
import base64
import httpx

def url_to_base64_image(url: str) -> dict:
    """将 URL 图像转为 Bedrock 兼容的 base64 格式"""
    resp = httpx.get(url)
    content_type = resp.headers.get("content-type", "image/jpeg")
    data = base64.standard_b64encode(resp.content).decode("utf-8")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": content_type,
            "data": data
        }
    }

# 迁移前 (Anthropic)
content = [
    {"type": "image", "source": {"type": "url", "url": "https://example.com/photo.jpg"}},
    {"type": "text", "text": "Describe this image."}
]

# 迁移后 (Bedrock)
content = [
    url_to_base64_image("https://example.com/photo.jpg"),
    {"type": "text", "text": "Describe this image."}
]
```

---

## 10. PDF 支持差异

### 10.1 对比

| 方面 | Anthropic 直接 API | Bedrock InvokeModel | Bedrock Converse |
|------|-------------------|---------------------|-----------------|
| base64 PDF | 支持 | **支持** | **支持** |
| URL PDF | 支持 | **不支持** | **不支持** |
| File API PDF | 支持 (Beta) | **不支持** | **不支持** |
| S3 PDF | 不支持 | 不支持 | **支持** |
| 视觉分析 (图表等) | 完整支持 | **完整支持** | **需启用 citations** |
| 最大页数 | 600 | 600 | 需验证 |

### 10.2 Converse API PDF 注意事项

```python
# Converse API: 不启用 citations 时，PDF 仅做基本文本提取，无法理解图表/图像
# 必须启用 citations 才能获得完整的视觉 PDF 分析

# Converse API (完整 PDF 支持)
{
    "role": "user",
    "content": [
        {
            "document": {
                "format": "pdf",
                "name": "report",
                "source": {"bytes": pdf_bytes},
                "citations": {"enabled": True}   # 必须启用!
            }
        },
        {"text": "Summarize the charts in this PDF."}
    ]
}
```

---

## 11. Citations 差异

### 11.1 对比

| 方面 | Anthropic 直接 API | Bedrock |
|------|-------------------|---------|
| 启用方式 | `citations.enabled = true` 在 document block 上 | **相同** (InvokeModel) |
| plain_text 引用 | `char_location` 类型 | **相同** |
| PDF 引用 | `page_location` 类型 | **相同** |
| custom content 引用 | `content_block_location` 类型 | **相同** |
| 与 Structured Outputs 兼容性 | 不兼容 (返回 400) | **相同** |

使用 InvokeModel API 时，Citations 格式与 Anthropic 直接 API **完全相同**，无需修改。

---

## 12. Batch Processing 差异

### 12.1 对比

| 方面 | Anthropic 直接 API | Bedrock |
|------|-------------------|---------|
| API 模式 | REST API (`/v1/messages/batches`) | S3 JSONL + 异步作业 |
| 提交方式 | 直接 POST JSON array | 上传 JSONL 到 S3 → CreateModelInvocationJob |
| 结果获取 | GET API 获取 | 从 S3 下载结果文件 |
| 价格 | 50% 折扣 | 具体折扣查 AWS 定价页 |
| 最大请求数 | 100,000 / batch | 取决于服务配额 |
| 最大大小 | 256 MB | 取决于服务配额 |
| 自定义 ID | `custom_id` | `recordId` |
| 完成时间 | 通常 1 小时内 | 根据作业大小 |

### 12.2 迁移代码示例

**Anthropic 直接 API**:
```python
# 简洁的 REST API 方式
batch = client.messages.batches.create(
    requests=[
        {
            "custom_id": "req-1",
            "params": {
                "model": "claude-sonnet-4-6",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        },
        {
            "custom_id": "req-2",
            "params": {
                "model": "claude-sonnet-4-6",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "World"}]
            }
        }
    ]
)

# 获取结果
results = client.messages.batches.results(batch.id)
for result in results:
    print(result.custom_id, result.result.message.content)
```

**Bedrock 等效方案**:
```python
import boto3, json

# 步骤1: 创建 JSONL 输入文件并上传到 S3
records = [
    {
        "recordId": "req-1",
        "modelInput": {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}]
        }
    },
    {
        "recordId": "req-2",
        "modelInput": {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "World"}]
        }
    }
]

jsonl_content = "\n".join(json.dumps(r) for r in records)

s3 = boto3.client('s3')
s3.put_object(
    Bucket='my-bucket',
    Key='batch-input/requests.jsonl',
    Body=jsonl_content
)

# 步骤2: 创建批量推理作业
bedrock = boto3.client('bedrock', region_name='us-west-2')
job = bedrock.create_model_invocation_job(
    jobName='my-batch-job',
    modelId='us.anthropic.claude-sonnet-4-6',
    roleArn='arn:aws:iam::123456789012:role/BedrockBatchRole',
    inputDataConfig={
        's3InputDataConfig': {
            's3Uri': 's3://my-bucket/batch-input/'
        }
    },
    outputDataConfig={
        's3OutputDataConfig': {
            's3Uri': 's3://my-bucket/batch-output/'
        }
    }
)

# 步骤3: 查询作业状态
status = bedrock.get_model_invocation_job(jobIdentifier=job['jobArn'])
print(status['status'])  # InProgress, Completed, Failed, etc.

# 步骤4: 从 S3 下载并解析结果
```

---

## 13. Structured Outputs 差异

| 方面 | Anthropic 直接 API | Bedrock |
|------|-------------------|---------|
| JSON Schema 输出 | `output_config.format` = JSON Schema | **支持** |
| Tool strict mode | `strict: true` on tool definition | **支持** |
| Vertex AI | 不支持 | - |

```python
# 两个平台格式相同 (通过 InvokeModel)
response = client.messages.create(
    model="...",
    max_tokens=1024,
    output_config={
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"]
            }
        }
    },
    messages=[{"role": "user", "content": "Extract info about John who is 30"}]
)
```

---

## 14. Built-in Tools 差异

### 14.1 可用性矩阵

| Built-in Tool | Anthropic API | Bedrock | Workaround |
|--------------|---------------|---------|------------|
| **Bash** (`bash_20250124`) | 支持 | **支持** (client-side) | - |
| **Text Editor** (`text_editor_20250728`) | 支持 | **支持** (client-side) | - |
| **Computer Use** (`computer_20241022`) | 支持 (Beta) | **支持** (Beta, 工具 type 仍为 `computer_20241022`) | Bedrock beta header: `computer-use-2025-01-24` (Claude 3.7) 或 `computer-use-2024-10-22` |
| **Memory** (`memory`) | 支持 | **支持** | - |
| **Tool Search** (BM25/Regex) | 支持 | **支持** | - |
| **Web Search** (`web_search_20260209`) | 支持 (server-side) | **不支持** | 见下方 workaround |
| **Web Fetch** (`web_fetch_20260209`) | 支持 (server-side) | **不支持** | 见下方 workaround |
| **Code Execution** (`code_execution_20260120`) | 支持 (server-side) | **不支持** | 见下方 workaround |

### 14.2 Web Search Workaround

Bedrock 不支持 Anthropic 的 server-side Web Search tool。但 Vertex AI 支持 (Google Search grounding)。

**Workaround 方案**:
```python
import httpx

# 方案1: 自建 Web Search tool
def web_search_tool(query: str) -> str:
    """通过第三方搜索 API 实现 web search"""
    # 使用 Tavily, Serper, Brave Search, SerpAPI 等
    resp = httpx.get(
        "https://api.tavily.com/search",
        params={"query": query, "api_key": "tvly-xxx"}
    )
    results = resp.json()["results"]
    return "\n".join(f"[{r['title']}]({r['url']}): {r['content']}" for r in results[:5])

# 将其定义为 Claude 的 tool
tools = [{
    "name": "web_search",
    "description": "Search the web for current information. Use this when you need up-to-date data.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
    }
}]

# 在 tool_use 回调中调用 web_search_tool()
```

### 14.3 Code Execution Workaround

Bedrock 不支持 Anthropic 的 server-side Code Execution sandbox。

**Workaround 方案**:
```python
# 方案1: 使用 AWS Lambda 作为代码执行沙箱
# 方案2: 使用 Docker 容器执行代码
# 方案3: 自建 tool，在受控环境中执行 Python 代码

import subprocess
import tempfile

def code_execution_tool(code: str) -> str:
    """在受控环境中执行 Python 代码"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                ['python3', f.name],
                capture_output=True, text=True, timeout=30
            )
            return result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out"
```

### 14.4 Web Fetch Workaround

```python
import httpx

def web_fetch_tool(url: str) -> str:
    """获取网页内容"""
    resp = httpx.get(url, follow_redirects=True, timeout=30)
    # 可以用 beautifulsoup4 做 HTML → text 转换
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.text, 'html.parser')
    return soup.get_text(separator='\n', strip=True)[:10000]
```

---

## 15. Anthropic 独有功能 (Bedrock 不支持)

### 15.1 完全不支持的功能

| 功能 | 说明 | Workaround |
|------|------|------------|
| **Files API** | 上传文件复用，避免每次请求重传 | 自建文件存储 + base64 编码，或使用 S3 (Converse API) |
| ~~Token Counting API~~ | ~~发送前统计 token 数量~~ | **Bedrock 支持** (通过 `AnthropicBedrock` SDK 的 `count_tokens()` 方法) |
| **Data Residency** (`inference_geo`) | 请求级别的地理位置控制 | 使用 Bedrock 的 Regional 端点前缀 (`us.`/`eu.` 等) |
| **Automatic Prompt Caching** | 顶层 `cache_control` 参数 | 手动在最后一个可缓存块上添加 `cache_control` |
| **Prompt Caching (Opus 4.6/Sonnet 4.6)** | 截至目前未在 Bedrock 缓存支持列表 | 等待 AWS 更新，或使用其他已支持的模型 |
| **1 小时 Prompt Cache TTL** | 部分模型上不支持 | 仅对 Opus 4.5, Sonnet 4.5, Haiku 4.5 可用 |
| **Web Search Tool** (server-side) | 内置网页搜索 | 自建 tool + 第三方搜索 API |
| **Web Fetch Tool** (server-side) | 内置网页抓取 | 自建 tool + httpx |
| **Code Execution Tool** (server-side) | 内置沙箱代码执行 | 自建 tool + Lambda/Docker |
| **Programmatic Tool Calling** | 在代码执行容器内调用 tools | 自建逻辑 |
| **Agent Skills API** | 预构建的 Agent 技能 (PPT, Excel 等) | 自建 MCP server 或 tool |
| **MCP Connector** | 直接从 Messages API 连接 MCP server | 自建 MCP client + tool routing |
| **`container` 参数** | 跨请求复用容器 | 不适用 |
| **`service_tier` 参数** | 选择 `auto`/`standard_only` | Bedrock 有自己的 serviceTier 机制 |
| **Skills API** (Beta) | 创建管理自定义技能 | 自建 |
| **Models API** | `GET /v1/models` 列出可用模型 | `aws bedrock list-foundation-models` |
| **Compaction** | 服务端自动上下文压缩 | Bedrock 也支持 (Beta) |

### 15.2 Token Counting 说明

虽然 Anthropic 有专门的 `/v1/messages/count_tokens` 端点，Bedrock 没有完全对等的独立 API。但是：

- 使用 `AnthropicBedrock` SDK 时，可使用 `client.messages.count_tokens()` (SDK 在本地计算)
- Bedrock Converse API 在响应中返回 `usage.totalTokens`
- 可以使用 `anthropic` 库的 tokenizer 进行本地计算

```python
# Workaround: 使用 anthropic SDK 本地 token 计数
from anthropic import AnthropicBedrock

client = AnthropicBedrock(aws_region="us-west-2")
count = client.messages.count_tokens(
    model="us.anthropic.claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)
print(f"Input tokens: {count.input_tokens}")
```

---

## 16. Bedrock 独有功能

| 功能 | 说明 |
|------|------|
| **Converse API** | 跨模型统一接口 (Anthropic, Amazon, Meta, Cohere 等) |
| **Guardrails** | 内容过滤、PII 遮蔽、主题拒绝、幻觉检测 (注意: Claude 4+ 暂不支持 Guardrails) |
| **S3 文档/图片引用** | Converse API 支持直接引用 S3 中的文件 |
| **Service Tier** | reserved / priority / default / flex 层级 |
| **Global/Regional 端点** | 通过模型 ID 前缀控制路由 |
| **服务端 Tool Calling** | Lambda / AgentCore Gateway 集成 (Responses API) |
| **Video 支持** | Converse API 支持视频输入 (MP4, MOV 等) |
| **Prompt Management** | 模板化提示管理 + `promptVariables` |
| **Request Metadata** | 自定义元数据 + 调用日志过滤 |
| **Invocation Logging** | 完整的请求/响应日志 |
| **VPC 支持** | 批量推理作业可配置 VPC |
| **Bearer Token 认证** | 企业环境替代方案 (C#/Go/Java SDK) |

---

## 17. Claude Agent SDK 迁移

### 17.1 概述

Claude Agent SDK 原生支持 Bedrock 作为后端。**无需修改 SDK 代码**，只需设置环境变量。

### 17.2 配置步骤

```bash
# 必需: 启用 Bedrock
export CLAUDE_CODE_USE_BEDROCK=1
export AWS_REGION=us-east-1

# 强烈推荐: 固定模型版本
export ANTHROPIC_MODEL='us.anthropic.claude-sonnet-4-6'
export ANTHROPIC_DEFAULT_OPUS_MODEL='us.anthropic.claude-opus-4-6-v1'
export ANTHROPIC_DEFAULT_SONNET_MODEL='us.anthropic.claude-sonnet-4-6'
export ANTHROPIC_DEFAULT_HAIKU_MODEL='us.anthropic.claude-haiku-4-5-20251001-v1:0'

# 可选: 禁用 prompt caching (如果区域不支持)
export DISABLE_PROMPT_CACHING=1
```

### 17.3 Python 代码示例

```python
import asyncio
import os
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ResultMessage

# 配置 Bedrock (也可以在 shell 中设置)
os.environ["CLAUDE_CODE_USE_BEDROCK"] = "1"
os.environ["AWS_REGION"] = "us-east-1"
os.environ["ANTHROPIC_MODEL"] = "us.anthropic.claude-sonnet-4-6"

async def main():
    async for message in query(
        prompt="Review utils.py for bugs and fix them.",
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Edit", "Glob", "Grep", "Bash"],
            permission_mode="acceptEdits",
            max_turns=20,
            max_budget_usd=2.0,
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    print(block.text)
        elif isinstance(message, ResultMessage):
            print(f"Done: cost=${message.total_cost_usd}, turns={message.num_turns}")

asyncio.run(main())
```

### 17.4 Agent SDK 在 Bedrock 上的限制

| 限制 | 说明 |
|------|------|
| 使用 InvokeModel API | SDK 内部使用 InvokeModel，不是 Converse |
| Prompt Caching | 部分区域可能不可用 |
| `/login` `/logout` 不可用 | 认证通过 AWS 凭证链 |
| `AWS_REGION` 必须设为环境变量 | 不读取 `~/.aws/config` |
| 模型固定推荐 | 不固定版本可能因新模型发布而中断 |

### 17.5 Guardrails 集成

```python
# 通过自定义 headers 集成 Bedrock Guardrails
options = ClaudeAgentOptions(
    env={
        "ANTHROPIC_CUSTOM_HEADERS": "X-Amzn-Bedrock-GuardrailIdentifier: your-id\nX-Amzn-Bedrock-GuardrailVersion: 1"
    }
)
```

---

## 18. 完整迁移 Checklist

### Phase 1: 基础迁移 (最小改动)

- [ ] 安装 Bedrock SDK: `pip install "anthropic[bedrock]"`
- [ ] 将 `Anthropic()` 改为 `AnthropicBedrock(aws_region="...")`
- [ ] 将所有模型 ID 替换为 Bedrock 格式
- [ ] 确认 EC2 IAM Role 有 `bedrock:InvokeModel` 和 `bedrock:InvokeModelWithResponseStream` 权限
- [ ] 设置 boto3 超时: `read_timeout=3600` (如使用 extended thinking)

### Phase 2: 功能适配

- [ ] **URL 图像**: 改为先下载再 base64 编码
- [ ] **URL PDF**: 同上
- [ ] **Files API**: 改为每次请求附带 base64 内容
- [ ] **Prompt Caching**: 移除顶层 `cache_control`，改为手动标注
- [ ] **1h Cache TTL**: 确认目标模型是否支持，不支持则改用 5m

### Phase 3: 缺失功能替代

- [ ] **Web Search**: 接入第三方搜索 API (Tavily/Serper/Brave)
- [ ] **Web Fetch**: 使用 httpx + BeautifulSoup 自建
- [ ] **Code Execution**: 使用 Lambda 或 Docker 沙箱
- [ ] **Token Counting**: 使用 `AnthropicBedrock` SDK 的 `count_tokens()`

### Phase 4: Agent SDK 迁移

- [ ] 设置环境变量 `CLAUDE_CODE_USE_BEDROCK=1`
- [ ] 固定模型版本
- [ ] 测试所有 agent 功能

### Phase 5: 测试验证

- [ ] 基本对话功能
- [ ] Streaming
- [ ] Tool Use (多轮)
- [ ] Extended Thinking
- [ ] Prompt Caching (验证缓存命中)
- [ ] 批量处理流程
- [ ] Vision (base64 图像)
- [ ] PDF 处理
- [ ] Citations
- [ ] Agent SDK 功能

---

## 附录 A: IAM Policy 模板

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "BedrockClaudeAccess",
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
                "bedrock:ListInferenceProfiles",
                "bedrock:ListFoundationModels"
            ],
            "Resource": [
                "arn:aws:bedrock:*:*:inference-profile/*",
                "arn:aws:bedrock:*:*:application-inference-profile/*",
                "arn:aws:bedrock:*:*:foundation-model/*"
            ]
        },
        {
            "Sid": "BedrockBatchAccess",
            "Effect": "Allow",
            "Action": [
                "bedrock:CreateModelInvocationJob",
                "bedrock:GetModelInvocationJob",
                "bedrock:ListModelInvocationJobs",
                "bedrock:StopModelInvocationJob"
            ],
            "Resource": "*"
        }
    ]
}
```

## 附录 B: Beta Headers 在 Bedrock 上的使用

```python
# Anthropic 直接 API - HTTP Header
headers = {
    "anthropic-beta": "interleaved-thinking-2025-05-14,token-efficient-tools-2025-02-19"
}

# Bedrock InvokeModel - Body 字段 (数组格式)
body = {
    "anthropic_version": "bedrock-2023-05-31",
    "anthropic_beta": [
        "interleaved-thinking-2025-05-14",
        "token-efficient-tools-2025-02-19"
    ],
    # ... 其他参数
}

# Bedrock Anthropic SDK - 自动处理
response = bedrock_client.messages.create(
    model="...",
    betas=["interleaved-thinking-2025-05-14"],  # SDK 处理转换
    # ...
)
```

## 附录 C: 快速参考 - 从 Anthropic 到 Bedrock 的一行改动

```python
# 如果你当前的代码是这样:
from anthropic import Anthropic
client = Anthropic()
response = client.messages.create(model="claude-sonnet-4-6", ...)

# 最简迁移 (只改两行):
from anthropic import AnthropicBedrock                         # 改 import
client = AnthropicBedrock(aws_region="us-west-2")              # 改 client
response = client.messages.create(model="us.anthropic.claude-sonnet-4-6", ...)  # 改 model ID
# 其他所有参数 (messages, tools, thinking, etc.) 保持不变!
```

---

*报告完成。下一步：逐项核实检查。*

#!/usr/bin/env python3
"""
AWS Bedrock Claude API 迁移报告验证脚本
逐项验证 migration_report.md 中的所有功能点
"""

import json
import time
import base64
import traceback
import boto3
import botocore.config

# ============================================================
# 配置
# ============================================================
REGION = "us-west-2"
# 使用 us. 前缀的 regional model IDs
SONNET_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
HAIKU_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
OPUS_MODEL = "us.anthropic.claude-opus-4-5-20251101-v1:0"
# 最新模型
SONNET_46_MODEL = "us.anthropic.claude-sonnet-4-6"
OPUS_46_MODEL = "us.anthropic.claude-opus-4-6-v1"

# boto3 client with extended timeout
config = botocore.config.Config(read_timeout=300, connect_timeout=10)
boto_client = boto3.client('bedrock-runtime', config=config, region_name=REGION)

results = []

def test(name, func):
    """运行一个测试并记录结果"""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    try:
        result = func()
        print(f"  RESULT: PASS")
        if result:
            # 截断过长的输出
            result_str = str(result)
            if len(result_str) > 500:
                result_str = result_str[:500] + "..."
            print(f"  DETAIL: {result_str}")
        results.append({"test": name, "status": "PASS", "detail": str(result)[:200] if result else ""})
    except Exception as e:
        print(f"  RESULT: FAIL")
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results.append({"test": name, "status": "FAIL", "detail": str(e)[:200]})


# ============================================================
# 1. 基础 InvokeModel 调用
# ============================================================
def test_basic_invoke_model():
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Say hello in one word."}]
    })
    response = boto_client.invoke_model(body=body, modelId=SONNET_MODEL)
    result = json.loads(response['body'].read())
    assert result['type'] == 'message', f"Expected 'message', got {result['type']}"
    assert result['role'] == 'assistant'
    assert result['content'][0]['type'] == 'text'
    assert result['usage']['input_tokens'] > 0
    assert result['usage']['output_tokens'] > 0
    return f"Model: {result['model']}, Output: {result['content'][0]['text'][:100]}"


# ============================================================
# 2. Anthropic Bedrock SDK 调用
# ============================================================
def test_anthropic_bedrock_sdk():
    from anthropic import AnthropicBedrock
    client = AnthropicBedrock(aws_region=REGION)
    message = client.messages.create(
        model=SONNET_MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": "Say hi in one word."}]
    )
    assert message.type == 'message'
    assert message.role == 'assistant'
    assert len(message.content) > 0
    return f"Model: {message.model}, Output: {message.content[0].text[:100]}"


# ============================================================
# 3. 模型 ID 格式验证 (Global, Regional, Base)
# ============================================================
def test_model_id_global():
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Say OK"}]
    })
    response = boto_client.invoke_model(body=body, modelId="global.anthropic.claude-sonnet-4-5-20250929-v1:0")
    result = json.loads(response['body'].read())
    return f"Global endpoint OK. Model: {result['model']}"

def test_model_id_sonnet46():
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Say OK"}]
    })
    response = boto_client.invoke_model(body=body, modelId=SONNET_46_MODEL)
    result = json.loads(response['body'].read())
    return f"Sonnet 4.6 OK. Model: {result['model']}"

def test_model_id_opus46():
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Say OK"}]
    })
    response = boto_client.invoke_model(body=body, modelId=OPUS_46_MODEL)
    result = json.loads(response['body'].read())
    return f"Opus 4.6 OK. Model: {result['model']}"

def test_model_id_haiku45():
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Say OK"}]
    })
    response = boto_client.invoke_model(body=body, modelId=HAIKU_MODEL)
    result = json.loads(response['body'].read())
    return f"Haiku 4.5 OK. Model: {result['model']}"


# ============================================================
# 4. System Prompt
# ============================================================
def test_system_prompt():
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "system": "You are a pirate. Always respond in pirate speak.",
        "messages": [{"role": "user", "content": "Hello"}]
    })
    response = boto_client.invoke_model(body=body, modelId=HAIKU_MODEL)
    result = json.loads(response['body'].read())
    return f"Output: {result['content'][0]['text'][:100]}"


# ============================================================
# 5. Temperature, top_p, top_k, stop_sequences
# ============================================================
def test_sampling_params():
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "temperature": 0.5,
        "top_k": 100,
        "stop_sequences": ["STOP"],
        "messages": [{"role": "user", "content": "Count from 1 to 10, say STOP after 5"}]
    })
    response = boto_client.invoke_model(body=body, modelId=HAIKU_MODEL)
    result = json.loads(response['body'].read())
    return f"stop_reason: {result['stop_reason']}, Output: {result['content'][0]['text'][:100]}"


# ============================================================
# 6. Streaming (InvokeModelWithResponseStream)
# ============================================================
def test_streaming():
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Count from 1 to 5"}]
    })
    response = boto_client.invoke_model_with_response_stream(body=body, modelId=HAIKU_MODEL)

    event_types = set()
    text_chunks = []
    for event in response['body']:
        chunk = json.loads(event['chunk']['bytes'])
        event_types.add(chunk['type'])
        if chunk['type'] == 'content_block_delta' and chunk.get('delta', {}).get('type') == 'text_delta':
            text_chunks.append(chunk['delta']['text'])

    full_text = ''.join(text_chunks)
    return f"Event types: {sorted(event_types)}, Text: {full_text[:100]}"


# ============================================================
# 7. Streaming via Anthropic SDK
# ============================================================
def test_streaming_sdk():
    from anthropic import AnthropicBedrock
    client = AnthropicBedrock(aws_region=REGION)

    text_parts = []
    with client.messages.stream(
        model=HAIKU_MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": "Count 1 to 5"}]
    ) as stream:
        for text in stream.text_stream:
            text_parts.append(text)

    full = ''.join(text_parts)
    return f"Streamed text: {full[:100]}"


# ============================================================
# 8. Tool Use (InvokeModel)
# ============================================================
def test_tool_use():
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 300,
        "tools": [{
            "name": "get_weather",
            "description": "Get weather for a city",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }],
        "tool_choice": {"type": "auto"},
        "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]
    })
    response = boto_client.invoke_model(body=body, modelId=HAIKU_MODEL)
    result = json.loads(response['body'].read())
    assert result['stop_reason'] == 'tool_use', f"Expected tool_use, got {result['stop_reason']}"
    tool_block = [b for b in result['content'] if b['type'] == 'tool_use'][0]
    return f"Tool called: {tool_block['name']}, Input: {tool_block['input']}"


# ============================================================
# 9. Tool Use - tool_choice "any"
# ============================================================
def test_tool_choice_any():
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 300,
        "tools": [{
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        }],
        "tool_choice": {"type": "any"},
        "messages": [{"role": "user", "content": "Hello"}]
    })
    response = boto_client.invoke_model(body=body, modelId=HAIKU_MODEL)
    result = json.loads(response['body'].read())
    assert result['stop_reason'] == 'tool_use'
    return "tool_choice 'any' works"


# ============================================================
# 10. Tool Use - tool_choice "tool" (specific)
# ============================================================
def test_tool_choice_specific():
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 300,
        "tools": [{
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        }],
        "tool_choice": {"type": "tool", "name": "get_weather"},
        "messages": [{"role": "user", "content": "Hello"}]
    })
    response = boto_client.invoke_model(body=body, modelId=HAIKU_MODEL)
    result = json.loads(response['body'].read())
    assert result['stop_reason'] == 'tool_use'
    return "tool_choice 'tool' (specific) works"


# ============================================================
# 11. Tool Use - tool_choice "none" (报告声称不支持)
# ============================================================
def test_tool_choice_none():
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "tools": [{
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        }],
        "tool_choice": {"type": "none"},
        "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]
    })
    try:
        response = boto_client.invoke_model(body=body, modelId=HAIKU_MODEL)
        result = json.loads(response['body'].read())
        return f"UNEXPECTED: tool_choice 'none' WORKS! stop_reason={result['stop_reason']}"
    except Exception as e:
        return f"CONFIRMED: tool_choice 'none' fails as expected: {str(e)[:150]}"


# ============================================================
# 12. Extended Thinking
# ============================================================
def test_extended_thinking():
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8000,
        "thinking": {"type": "enabled", "budget_tokens": 2000},
        "messages": [{"role": "user", "content": "What is 15 * 37?"}]
    })
    response = boto_client.invoke_model(body=body, modelId=SONNET_MODEL)
    result = json.loads(response['body'].read())

    thinking_block = [b for b in result['content'] if b['type'] == 'thinking']
    text_block = [b for b in result['content'] if b['type'] == 'text']

    assert len(thinking_block) > 0, "No thinking block found"
    assert len(text_block) > 0, "No text block found"
    assert 'signature' in thinking_block[0], "No signature in thinking block"

    return f"Thinking ({len(thinking_block[0]['thinking'])} chars), Text: {text_block[0]['text'][:100]}"


# ============================================================
# 13. Extended Thinking - Adaptive (Opus 4.6)
# ============================================================
def test_adaptive_thinking():
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4000,
        "thinking": {"type": "adaptive"},
        "messages": [{"role": "user", "content": "What is 2+2?"}]
    })
    response = boto_client.invoke_model(body=body, modelId=OPUS_46_MODEL)
    result = json.loads(response['body'].read())
    return f"Adaptive thinking OK. Content types: {[b['type'] for b in result['content']]}"


# ============================================================
# 14. Beta Headers in body (anthropic_beta)
# ============================================================
def test_beta_headers_in_body():
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": ["interleaved-thinking-2025-05-14"],
        "max_tokens": 4000,
        "thinking": {"type": "enabled", "budget_tokens": 2000},
        "messages": [{"role": "user", "content": "Solve: 123 + 456"}]
    })
    response = boto_client.invoke_model(body=body, modelId=SONNET_MODEL)
    result = json.loads(response['body'].read())
    return f"Beta header accepted. Content types: {[b['type'] for b in result['content']]}"


# ============================================================
# 15. Prompt Caching (InvokeModel - cache_control)
# ============================================================
def test_prompt_caching():
    long_text = "This is a test of prompt caching on Bedrock. " * 300  # ~4500 tokens
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "system": [{
            "type": "text",
            "text": long_text,
            "cache_control": {"type": "ephemeral"}
        }],
        "messages": [{"role": "user", "content": "Summarize the system prompt in 5 words."}]
    })

    # First call - cache write
    response1 = boto_client.invoke_model(body=body, modelId=SONNET_MODEL)
    result1 = json.loads(response1['body'].read())
    usage1 = result1.get('usage', {})

    # Second call - should be cache read
    time.sleep(1)
    response2 = boto_client.invoke_model(body=body, modelId=SONNET_MODEL)
    result2 = json.loads(response2['body'].read())
    usage2 = result2.get('usage', {})

    return (
        f"Call 1 usage: {usage1}\n"
        f"  Call 2 usage: {usage2}\n"
        f"  cache_creation_input_tokens: {usage1.get('cache_creation_input_tokens', 'N/A')} -> {usage2.get('cache_creation_input_tokens', 'N/A')}\n"
        f"  cache_read_input_tokens: {usage1.get('cache_read_input_tokens', 'N/A')} -> {usage2.get('cache_read_input_tokens', 'N/A')}"
    )


# ============================================================
# 16. Prompt Caching on Sonnet 4.6 (报告说可能不在支持列表)
# ============================================================
def test_prompt_caching_sonnet46():
    long_text = "Prompt caching test for Sonnet 4.6. " * 300
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 50,
        "system": [{
            "type": "text",
            "text": long_text,
            "cache_control": {"type": "ephemeral"}
        }],
        "messages": [{"role": "user", "content": "Say OK"}]
    })
    response = boto_client.invoke_model(body=body, modelId=SONNET_46_MODEL)
    result = json.loads(response['body'].read())
    usage = result.get('usage', {})
    has_cache = usage.get('cache_creation_input_tokens', 0) > 0 or usage.get('cache_read_input_tokens', 0) > 0
    return f"Sonnet 4.6 cache: {'SUPPORTED' if has_cache else 'NOT SUPPORTED (no cache tokens in usage)'}. Usage: {usage}"


# ============================================================
# 17. Prompt Caching on Opus 4.6
# ============================================================
def test_prompt_caching_opus46():
    long_text = "Prompt caching test for Opus 4.6. " * 300
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 50,
        "system": [{
            "type": "text",
            "text": long_text,
            "cache_control": {"type": "ephemeral"}
        }],
        "messages": [{"role": "user", "content": "Say OK"}]
    })
    response = boto_client.invoke_model(body=body, modelId=OPUS_46_MODEL)
    result = json.loads(response['body'].read())
    usage = result.get('usage', {})
    has_cache = usage.get('cache_creation_input_tokens', 0) > 0 or usage.get('cache_read_input_tokens', 0) > 0
    return f"Opus 4.6 cache: {'SUPPORTED' if has_cache else 'NOT SUPPORTED (no cache tokens in usage)'}. Usage: {usage}"


# ============================================================
# 18. Vision - base64 image
# ============================================================
def test_vision_base64():
    # Create a minimal 1x1 red PNG
    import struct, zlib
    def create_png():
        sig = b'\x89PNG\r\n\x1a\n'
        ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
        ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff
        ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
        raw = b'\x00\xff\x00\x00'  # filter byte + RGB
        idat_data = zlib.compress(raw)
        idat_crc = zlib.crc32(b'IDAT' + idat_data) & 0xffffffff
        idat = struct.pack('>I', len(idat_data)) + b'IDAT' + idat_data + struct.pack('>I', idat_crc)
        iend_crc = zlib.crc32(b'IEND') & 0xffffffff
        iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
        return sig + ihdr + idat + iend

    png_data = base64.standard_b64encode(create_png()).decode('utf-8')

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png_data}},
                {"type": "text", "text": "What color is this pixel? Answer in one word."}
            ]
        }]
    })
    response = boto_client.invoke_model(body=body, modelId=HAIKU_MODEL)
    result = json.loads(response['body'].read())
    return f"Vision output: {result['content'][0]['text'][:100]}"


# ============================================================
# 19. PDF Support (base64)
# ============================================================
def test_pdf_support():
    # Create a minimal PDF
    pdf_content = b"""%PDF-1.0
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj
4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
5 0 obj<</Length 44>>stream
BT /F1 24 Tf 100 700 Td (Hello PDF Test) Tj ET
endstream
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000340 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
434
%%EOF"""

    pdf_b64 = base64.standard_b64encode(pdf_content).decode('utf-8')

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_b64}},
                {"type": "text", "text": "What text is in this PDF?"}
            ]
        }]
    })
    response = boto_client.invoke_model(body=body, modelId=HAIKU_MODEL)
    result = json.loads(response['body'].read())
    return f"PDF output: {result['content'][0]['text'][:100]}"


# ============================================================
# 20. Citations
# ============================================================
def test_citations():
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 300,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {"type": "text", "media_type": "text/plain", "data": "The capital of France is Paris. Paris has a population of 2.1 million people."},
                    "title": "France Facts",
                    "citations": {"enabled": True}
                },
                {"type": "text", "text": "What is the capital of France and its population?"}
            ]
        }]
    })
    response = boto_client.invoke_model(body=body, modelId=HAIKU_MODEL)
    result = json.loads(response['body'].read())

    # Check for citation blocks in the response
    has_citations = False
    for block in result['content']:
        if block['type'] == 'text' and 'citations' in block:
            has_citations = True

    return f"Citations found: {has_citations}, Content: {json.dumps(result['content'], indent=2)[:300]}"


# ============================================================
# 21. Structured Outputs (output_config with JSON schema)
# ============================================================
def test_structured_outputs():
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 300,
        "output_config": {
            "format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"}
                        },
                        "required": ["name", "age"]
                    }
                }
            }
        },
        "messages": [{"role": "user", "content": "Extract: John is 30 years old."}]
    })
    response = boto_client.invoke_model(body=body, modelId=SONNET_MODEL)
    result = json.loads(response['body'].read())
    text = result['content'][0]['text']
    parsed = json.loads(text)
    return f"Structured output: {parsed}"


# ============================================================
# 22. Effort parameter (with beta header)
# ============================================================
def test_effort_parameter():
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": ["effort-2025-11-24"],
        "max_tokens": 100,
        "output_config": {"effort": "low"},
        "messages": [{"role": "user", "content": "What is 2+2?"}]
    })
    response = boto_client.invoke_model(body=body, modelId=OPUS_MODEL)
    result = json.loads(response['body'].read())
    return f"Effort 'low' works. Output: {result['content'][0]['text'][:100]}"


# ============================================================
# 23. Converse API - basic call
# ============================================================
def test_converse_api():
    response = boto_client.converse(
        modelId=HAIKU_MODEL,
        messages=[{
            "role": "user",
            "content": [{"text": "Say hello in one word."}]
        }],
        inferenceConfig={
            "maxTokens": 100,
            "temperature": 0.5
        }
    )
    text = response['output']['message']['content'][0]['text']
    usage = response['usage']
    return f"Converse output: {text[:100]}, Usage: inputTokens={usage['inputTokens']}, outputTokens={usage['outputTokens']}"


# ============================================================
# 24. Converse API - Tool Use
# ============================================================
def test_converse_tool_use():
    response = boto_client.converse(
        modelId=HAIKU_MODEL,
        messages=[{
            "role": "user",
            "content": [{"text": "What's the weather in Tokyo?"}]
        }],
        toolConfig={
            "tools": [{
                "toolSpec": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"]
                        }
                    }
                }
            }]
        }
    )
    assert response['stopReason'] == 'tool_use'
    content = response['output']['message']['content']
    tool_use = [b for b in content if 'toolUse' in b][0]['toolUse']
    return f"Converse tool: {tool_use['name']}, Input: {tool_use['input']}"


# ============================================================
# 25. Converse API - Prompt Caching (cachePoint)
# ============================================================
def test_converse_cache():
    long_text = "Converse caching test content. " * 300

    response = boto_client.converse(
        modelId=SONNET_MODEL,
        system=[
            {"text": long_text},
            {"cachePoint": {"type": "default"}}
        ],
        messages=[{
            "role": "user",
            "content": [{"text": "Say OK"}]
        }],
        inferenceConfig={"maxTokens": 50}
    )
    usage = response['usage']
    cache_write = usage.get('cacheWriteInputTokens', usage.get('cacheWriteInputTokensCount', 'N/A'))
    cache_read = usage.get('cacheReadInputTokens', usage.get('cacheReadInputTokensCount', 'N/A'))
    return f"Converse cache - write: {cache_write}, read: {cache_read}"


# ============================================================
# 26. Converse API - additionalModelRequestFields (top_k)
# ============================================================
def test_converse_top_k():
    response = boto_client.converse(
        modelId=HAIKU_MODEL,
        messages=[{
            "role": "user",
            "content": [{"text": "Say hello"}]
        }],
        inferenceConfig={"maxTokens": 50},
        additionalModelRequestFields={"top_k": 50}
    )
    return f"top_k via additionalModelRequestFields works. Output: {response['output']['message']['content'][0]['text'][:50]}"


# ============================================================
# 27. Token Counting (via Anthropic SDK)
# ============================================================
def test_token_counting():
    from anthropic import AnthropicBedrock
    client = AnthropicBedrock(aws_region=REGION)
    count = client.messages.count_tokens(
        model=HAIKU_MODEL,
        messages=[{"role": "user", "content": "Hello, how are you doing today?"}]
    )
    return f"Token count: {count.input_tokens}"


# ============================================================
# 28. Multi-turn Tool Use (complete tool loop)
# ============================================================
def test_multi_turn_tool_use():
    tools = [{
        "name": "calculator",
        "description": "Evaluate a math expression",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]
        }
    }]

    messages = [{"role": "user", "content": "What is 15 * 37? Use the calculator tool."}]

    # First call - should trigger tool use
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 300,
        "tools": tools,
        "messages": messages
    })
    response = boto_client.invoke_model(body=body, modelId=HAIKU_MODEL)
    result = json.loads(response['body'].read())

    # Add assistant response to messages
    messages.append({"role": "assistant", "content": result['content']})

    # Find tool use and add result
    tool_use = [b for b in result['content'] if b['type'] == 'tool_use'][0]
    messages.append({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": tool_use['id'],
            "content": "555"
        }]
    })

    # Second call - should produce final answer
    body2 = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 300,
        "tools": tools,
        "messages": messages
    })
    response2 = boto_client.invoke_model(body=body2, modelId=HAIKU_MODEL)
    result2 = json.loads(response2['body'].read())

    return f"Multi-turn tool use OK. Final: {result2['content'][0]['text'][:100]}"


# ============================================================
# 29. disable_parallel_tool_use (报告说可能不支持)
# ============================================================
def test_disable_parallel_tool_use():
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 300,
        "tools": [
            {"name": "tool_a", "description": "Tool A", "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}}},
            {"name": "tool_b", "description": "Tool B", "input_schema": {"type": "object", "properties": {"y": {"type": "string"}}}}
        ],
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        "messages": [{"role": "user", "content": "Use tool_a with x='hello' and tool_b with y='world'"}]
    })
    try:
        response = boto_client.invoke_model(body=body, modelId=HAIKU_MODEL)
        result = json.loads(response['body'].read())
        tool_uses = [b for b in result['content'] if b['type'] == 'tool_use']
        return f"disable_parallel_tool_use WORKS. Tool uses count: {len(tool_uses)}"
    except Exception as e:
        return f"disable_parallel_tool_use FAILS: {str(e)[:150]}"


# ============================================================
# 30. URL image source (报告说不支持)
# ============================================================
def test_url_image_not_supported():
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "url", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"}},
                {"type": "text", "text": "Describe this image."}
            ]
        }]
    })
    try:
        response = boto_client.invoke_model(body=body, modelId=HAIKU_MODEL)
        result = json.loads(response['body'].read())
        return f"UNEXPECTED: URL image WORKS! Output: {result['content'][0]['text'][:100]}"
    except Exception as e:
        return f"CONFIRMED: URL image not supported: {str(e)[:150]}"


# ============================================================
# RUN ALL TESTS
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("AWS Bedrock Claude API 验证测试")
    print(f"Region: {REGION}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Basic connectivity
    test("1. InvokeModel 基础调用 (Sonnet 4.5)", test_basic_invoke_model)
    test("2. Anthropic Bedrock SDK 调用", test_anthropic_bedrock_sdk)

    # Model IDs
    test("3a. Global 模型 ID", test_model_id_global)
    test("3b. Sonnet 4.6 模型", test_model_id_sonnet46)
    test("3c. Opus 4.6 模型", test_model_id_opus46)
    test("3d. Haiku 4.5 模型", test_model_id_haiku45)

    # Parameters
    test("4. System Prompt", test_system_prompt)
    test("5. Sampling 参数 (temperature, top_k, stop_sequences)", test_sampling_params)

    # Streaming
    test("6. Streaming (boto3)", test_streaming)
    test("7. Streaming (Anthropic SDK)", test_streaming_sdk)

    # Tool Use
    test("8. Tool Use - auto", test_tool_use)
    test("9. Tool Use - tool_choice 'any'", test_tool_choice_any)
    test("10. Tool Use - tool_choice 'tool' (specific)", test_tool_choice_specific)
    test("11. Tool Use - tool_choice 'none' (预期失败)", test_tool_choice_none)
    test("28. Multi-turn Tool Use (完整工具循环)", test_multi_turn_tool_use)
    test("29. disable_parallel_tool_use (预期可能不支持)", test_disable_parallel_tool_use)

    # Extended Thinking
    test("12. Extended Thinking (Sonnet 4.5)", test_extended_thinking)
    test("13. Adaptive Thinking (Opus 4.6)", test_adaptive_thinking)
    test("14. Beta Headers in body", test_beta_headers_in_body)

    # Prompt Caching
    test("15. Prompt Caching (Sonnet 4.5 - InvokeModel)", test_prompt_caching)
    test("16. Prompt Caching (Sonnet 4.6 - 验证是否支持)", test_prompt_caching_sonnet46)
    test("17. Prompt Caching (Opus 4.6 - 验证是否支持)", test_prompt_caching_opus46)

    # Vision
    test("18. Vision - base64 PNG", test_vision_base64)
    test("30. Vision - URL image (预期不支持)", test_url_image_not_supported)

    # PDF
    test("19. PDF Support (base64)", test_pdf_support)

    # Citations
    test("20. Citations", test_citations)

    # Structured Outputs
    test("21. Structured Outputs (JSON Schema)", test_structured_outputs)

    # Effort
    test("22. Effort 参数 (Opus 4.5)", test_effort_parameter)

    # Converse API
    test("23. Converse API 基础调用", test_converse_api)
    test("24. Converse API Tool Use", test_converse_tool_use)
    test("25. Converse API Prompt Caching (cachePoint)", test_converse_cache)
    test("26. Converse API top_k (additionalModelRequestFields)", test_converse_top_k)

    # Token Counting
    test("27. Token Counting (Anthropic SDK)", test_token_counting)

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)

    passed = [r for r in results if r['status'] == 'PASS']
    failed = [r for r in results if r['status'] == 'FAIL']

    print(f"\n总计: {len(results)} 测试")
    print(f"通过: {len(passed)}")
    print(f"失败: {len(failed)}")

    if failed:
        print(f"\n失败的测试:")
        for r in failed:
            print(f"  FAIL: {r['test']}")
            print(f"        {r['detail']}")

    print("\n详细结果:")
    for r in results:
        status_marker = "PASS" if r['status'] == 'PASS' else "FAIL"
        print(f"  [{status_marker}] {r['test']}")

    # Save results to JSON
    with open('/home/ec2-user/bedrock-api/verification_results.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: /home/ec2-user/bedrock-api/verification_results.json")

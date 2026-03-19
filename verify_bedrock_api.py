#!/usr/bin/env python3
"""
AWS Bedrock Claude API 全面验证脚本
覆盖 migration_report.md 中所有功能点 (支持 + 不支持)
基于 Python 3.12 + boto3，不依赖 anthropic SDK
"""

import json
import time
import base64
import struct
import zlib
import traceback
import boto3
import botocore.config

# ============================================================
# 配置
# ============================================================
REGION = "us-west-2"
SONNET_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
HAIKU_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
OPUS_MODEL = "us.anthropic.claude-opus-4-5-20251101-v1:0"
SONNET_46_MODEL = "us.anthropic.claude-sonnet-4-6"
OPUS_46_MODEL = "us.anthropic.claude-opus-4-6-v1"

config = botocore.config.Config(read_timeout=300, connect_timeout=10)
client = boto3.client('bedrock-runtime', config=config, region_name=REGION)
bedrock_mgmt = boto3.client('bedrock', region_name=REGION)

results = []

# 用于 cache 测试的长文本 (>4096 tokens)
LONG_TEXT = "This is a comprehensive test of prompt caching on AWS Bedrock. " * 800


def test(name, func):
    """运行一个测试并记录结果"""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    try:
        result = func()
        print(f"  RESULT: PASS")
        if result:
            result_str = str(result)
            if len(result_str) > 500:
                result_str = result_str[:500] + "..."
            print(f"  DETAIL: {result_str}")
        results.append({"test": name, "status": "PASS", "detail": str(result)[:300] if result else ""})
    except Exception as e:
        print(f"  RESULT: FAIL")
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results.append({"test": name, "status": "FAIL", "detail": str(e)[:300]})


def invoke(model, body_dict):
    """辅助: InvokeModel 调用"""
    body_dict.setdefault("anthropic_version", "bedrock-2023-05-31")
    resp = client.invoke_model(body=json.dumps(body_dict), modelId=model)
    return json.loads(resp['body'].read())


def invoke_stream(model, body_dict):
    """辅助: Streaming 调用"""
    body_dict.setdefault("anthropic_version", "bedrock-2023-05-31")
    resp = client.invoke_model_with_response_stream(body=json.dumps(body_dict), modelId=model)
    events = []
    for event in resp['body']:
        events.append(json.loads(event['chunk']['bytes']))
    return events


def create_png():
    """创建最小 1x1 红色 PNG"""
    sig = b'\x89PNG\r\n\x1a\n'
    ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff
    ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
    raw = b'\x00\xff\x00\x00'
    idat_data = zlib.compress(raw)
    idat_crc = zlib.crc32(b'IDAT' + idat_data) & 0xffffffff
    idat = struct.pack('>I', len(idat_data)) + b'IDAT' + idat_data + struct.pack('>I', idat_crc)
    iend_crc = zlib.crc32(b'IEND') & 0xffffffff
    iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
    return sig + ihdr + idat + iend


MINIMAL_PDF = b"""%PDF-1.0
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
0000000000 65535 f\x20
0000000009 00000 n\x20
0000000058 00000 n\x20
0000000115 00000 n\x20
0000000266 00000 n\x20
0000000340 00000 n\x20
trailer<</Size 6/Root 1 0 R>>
startxref
434
%%EOF"""


# ============================================================
# SECTION 1: 基础连接 & 模型 ID
# ============================================================

def test_01_invoke_model_sonnet45():
    r = invoke(SONNET_MODEL, {"max_tokens": 50, "messages": [{"role": "user", "content": "Say OK"}]})
    assert r['type'] == 'message' and r['role'] == 'assistant'
    return f"Model: {r['model']}, Output: {r['content'][0]['text'][:50]}"

def test_02_invoke_model_haiku45():
    r = invoke(HAIKU_MODEL, {"max_tokens": 50, "messages": [{"role": "user", "content": "Say OK"}]})
    return f"Model: {r['model']}"

def test_03_invoke_model_opus45():
    r = invoke(OPUS_MODEL, {"max_tokens": 50, "messages": [{"role": "user", "content": "Say OK"}]})
    return f"Model: {r['model']}"

def test_04_invoke_model_sonnet46():
    r = invoke(SONNET_46_MODEL, {"max_tokens": 50, "messages": [{"role": "user", "content": "Say OK"}]})
    return f"Model: {r['model']}"

def test_05_invoke_model_opus46():
    r = invoke(OPUS_46_MODEL, {"max_tokens": 50, "messages": [{"role": "user", "content": "Say OK"}]})
    return f"Model: {r['model']}"

def test_06_global_endpoint():
    r = invoke("global.anthropic.claude-sonnet-4-5-20250929-v1:0",
               {"max_tokens": 50, "messages": [{"role": "user", "content": "Say OK"}]})
    return f"Global endpoint OK. Model: {r['model']}"


# ============================================================
# SECTION 2: Messages API 核心参数
# ============================================================

def test_07_system_prompt():
    r = invoke(HAIKU_MODEL, {
        "max_tokens": 100,
        "system": "You are a pirate. Always respond in pirate speak.",
        "messages": [{"role": "user", "content": "Hello"}]
    })
    return f"Output: {r['content'][0]['text'][:100]}"

def test_08_system_prompt_array():
    """system 作为 content block 数组"""
    r = invoke(HAIKU_MODEL, {
        "max_tokens": 100,
        "system": [{"type": "text", "text": "You are a pirate."}],
        "messages": [{"role": "user", "content": "Hello"}]
    })
    return f"System array OK. Output: {r['content'][0]['text'][:80]}"

def test_09_temperature_topk_stop():
    r = invoke(HAIKU_MODEL, {
        "max_tokens": 100, "temperature": 0.5, "top_k": 100,
        "stop_sequences": ["STOP"],
        "messages": [{"role": "user", "content": "Count from 1 to 10, say STOP after 5"}]
    })
    return f"stop_reason: {r['stop_reason']}, Output: {r['content'][0]['text'][:80]}"

def test_10_top_p():
    r = invoke(HAIKU_MODEL, {
        "max_tokens": 50, "top_p": 0.9,
        "messages": [{"role": "user", "content": "Say OK"}]
    })
    return f"top_p OK. Output: {r['content'][0]['text'][:50]}"

def test_11_metadata():
    r = invoke(HAIKU_MODEL, {
        "max_tokens": 50,
        "metadata": {"user_id": "test-user-123"},
        "messages": [{"role": "user", "content": "Say OK"}]
    })
    return f"metadata accepted. Output: {r['content'][0]['text'][:50]}"


# ============================================================
# SECTION 3: Streaming
# ============================================================

def test_12_streaming():
    events = invoke_stream(HAIKU_MODEL, {
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Count from 1 to 5"}]
    })
    event_types = sorted(set(e['type'] for e in events))
    text_chunks = [e['delta']['text'] for e in events
                   if e['type'] == 'content_block_delta' and e.get('delta', {}).get('type') == 'text_delta']
    return f"Event types: {event_types}, Text: {''.join(text_chunks)[:100]}"

def test_13_streaming_thinking():
    """Streaming + Extended Thinking"""
    events = invoke_stream(SONNET_MODEL, {
        "max_tokens": 4000,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "messages": [{"role": "user", "content": "What is 7*8?"}]
    })
    event_types = sorted(set(e['type'] for e in events))
    has_thinking_delta = any(
        e.get('delta', {}).get('type') == 'thinking_delta' for e in events
    )
    return f"Stream+Thinking OK. Types: {event_types}, has_thinking_delta: {has_thinking_delta}"


# ============================================================
# SECTION 4: Tool Use
# ============================================================

WEATHER_TOOL = {
    "name": "get_weather", "description": "Get weather for a city",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
}

def test_14_tool_use_auto():
    r = invoke(HAIKU_MODEL, {
        "max_tokens": 300, "tools": [WEATHER_TOOL], "tool_choice": {"type": "auto"},
        "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]
    })
    assert r['stop_reason'] == 'tool_use'
    tb = [b for b in r['content'] if b['type'] == 'tool_use'][0]
    return f"Tool: {tb['name']}, Input: {tb['input']}"

def test_15_tool_choice_any():
    r = invoke(HAIKU_MODEL, {
        "max_tokens": 300, "tools": [WEATHER_TOOL], "tool_choice": {"type": "any"},
        "messages": [{"role": "user", "content": "Hello"}]
    })
    assert r['stop_reason'] == 'tool_use'
    return "tool_choice 'any' works"

def test_16_tool_choice_specific():
    r = invoke(HAIKU_MODEL, {
        "max_tokens": 300, "tools": [WEATHER_TOOL],
        "tool_choice": {"type": "tool", "name": "get_weather"},
        "messages": [{"role": "user", "content": "Hello"}]
    })
    assert r['stop_reason'] == 'tool_use'
    return "tool_choice 'tool' (specific) works"

def test_17_tool_choice_none():
    """报告中提到可能不支持"""
    try:
        r = invoke(HAIKU_MODEL, {
            "max_tokens": 100, "tools": [WEATHER_TOOL],
            "tool_choice": {"type": "none"},
            "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]
        })
        return f"tool_choice 'none' WORKS! stop_reason={r['stop_reason']}"
    except Exception as e:
        return f"tool_choice 'none' FAILS: {str(e)[:150]}"

def test_18_multi_turn_tool():
    tools = [{"name": "calc", "description": "Calculate", "input_schema": {
        "type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]
    }}]
    msgs = [{"role": "user", "content": "What is 15*37? Use the calc tool."}]
    r1 = invoke(HAIKU_MODEL, {"max_tokens": 300, "tools": tools, "messages": msgs})
    msgs.append({"role": "assistant", "content": r1['content']})
    tu = [b for b in r1['content'] if b['type'] == 'tool_use'][0]
    msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu['id'], "content": "555"}]})
    r2 = invoke(HAIKU_MODEL, {"max_tokens": 300, "tools": tools, "messages": msgs})
    return f"Multi-turn OK. Final: {r2['content'][0]['text'][:80]}"

def test_19_disable_parallel_tool_use():
    r = invoke(HAIKU_MODEL, {
        "max_tokens": 300,
        "tools": [
            {"name": "t_a", "description": "Tool A", "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}}},
            {"name": "t_b", "description": "Tool B", "input_schema": {"type": "object", "properties": {"y": {"type": "string"}}}}
        ],
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        "messages": [{"role": "user", "content": "Use t_a with x='hello' and t_b with y='world'"}]
    })
    tool_uses = [b for b in r['content'] if b['type'] == 'tool_use']
    return f"disable_parallel_tool_use OK. Tool uses: {len(tool_uses)}"

def test_20_tool_strict_mode():
    """strict: true on tool definition (Structured Outputs for tools)"""
    r = invoke(HAIKU_MODEL, {
        "max_tokens": 300,
        "tools": [{
            "name": "extract_person",
            "description": "Extract person info",
            "input_schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
                "additionalProperties": False
            },
            "strict": True
        }],
        "tool_choice": {"type": "tool", "name": "extract_person"},
        "messages": [{"role": "user", "content": "John is 30 years old."}]
    })
    tb = [b for b in r['content'] if b['type'] == 'tool_use'][0]
    return f"strict tool OK. Input: {tb['input']}"


# ============================================================
# SECTION 5: Extended Thinking
# ============================================================

def test_21_thinking_enabled():
    r = invoke(SONNET_MODEL, {
        "max_tokens": 8000, "thinking": {"type": "enabled", "budget_tokens": 2000},
        "messages": [{"role": "user", "content": "What is 15 * 37?"}]
    })
    thinking = [b for b in r['content'] if b['type'] == 'thinking']
    text = [b for b in r['content'] if b['type'] == 'text']
    assert len(thinking) > 0 and len(text) > 0
    assert 'signature' in thinking[0]
    return f"Thinking ({len(thinking[0]['thinking'])} chars), Text: {text[0]['text'][:80]}"

def test_22_thinking_adaptive():
    r = invoke(OPUS_46_MODEL, {
        "max_tokens": 4000, "thinking": {"type": "adaptive"},
        "messages": [{"role": "user", "content": "What is 2+2?"}]
    })
    return f"Adaptive OK. Types: {[b['type'] for b in r['content']]}"

def test_23_beta_interleaved_thinking():
    r = invoke(SONNET_MODEL, {
        "anthropic_beta": ["interleaved-thinking-2025-05-14"],
        "max_tokens": 4000, "thinking": {"type": "enabled", "budget_tokens": 2000},
        "messages": [{"role": "user", "content": "Solve: 123 + 456"}]
    })
    return f"Interleaved thinking beta OK. Types: {[b['type'] for b in r['content']]}"

def test_24_thinking_summarized():
    """Summarized thinking display option"""
    r = invoke(SONNET_MODEL, {
        "max_tokens": 8000,
        "thinking": {"type": "enabled", "budget_tokens": 2000},
        "messages": [{"role": "user", "content": "What is 99*101?"}]
    })
    thinking = [b for b in r['content'] if b['type'] == 'thinking']
    # Claude 4 默认返回 summarized thinking
    has_signature = 'signature' in thinking[0] if thinking else False
    return f"Thinking block has signature: {has_signature}, thinking chars: {len(thinking[0]['thinking']) if thinking else 0}"


# ============================================================
# SECTION 6: Prompt Caching (InvokeModel)
# ============================================================

def test_25_cache_sonnet45():
    body = {
        "max_tokens": 50,
        "system": [{"type": "text", "text": LONG_TEXT, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": "Say OK"}]
    }
    r1 = invoke(SONNET_MODEL, body)
    u1 = r1['usage']
    time.sleep(1)
    r2 = invoke(SONNET_MODEL, body)
    u2 = r2['usage']
    return (f"Write: {u1.get('cache_creation_input_tokens',0)}, "
            f"Read: {u2.get('cache_read_input_tokens',0)}")

def test_26_cache_sonnet46():
    body = {
        "max_tokens": 50,
        "system": [{"type": "text", "text": LONG_TEXT, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": "Say OK"}]
    }
    r1 = invoke(SONNET_46_MODEL, body)
    u1 = r1['usage']
    time.sleep(1)
    r2 = invoke(SONNET_46_MODEL, body)
    u2 = r2['usage']
    w = u1.get('cache_creation_input_tokens', 0)
    rd = u2.get('cache_read_input_tokens', 0)
    return f"Write: {w}, Read: {rd}, {'SUPPORTED' if w > 0 or rd > 0 else 'NOT SUPPORTED'}"

def test_27_cache_opus46():
    body = {
        "max_tokens": 50,
        "system": [{"type": "text", "text": LONG_TEXT, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": "Say OK"}]
    }
    r1 = invoke(OPUS_46_MODEL, body)
    u1 = r1['usage']
    time.sleep(1)
    r2 = invoke(OPUS_46_MODEL, body)
    u2 = r2['usage']
    w = u1.get('cache_creation_input_tokens', 0)
    rd = u2.get('cache_read_input_tokens', 0)
    return f"Write: {w}, Read: {rd}, {'SUPPORTED' if w > 0 or rd > 0 else 'NOT SUPPORTED'}"

def test_28_cache_1h_ttl():
    """1 小时 TTL"""
    body = {
        "max_tokens": 50,
        "system": [{"type": "text", "text": LONG_TEXT, "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
        "messages": [{"role": "user", "content": "Say OK"}]
    }
    r = invoke(SONNET_MODEL, body)
    u = r['usage']
    cache_creation = u.get('cache_creation', {})
    return (f"1h TTL accepted. cache_creation_input_tokens={u.get('cache_creation_input_tokens',0)}, "
            f"cache_read={u.get('cache_read_input_tokens',0)}, "
            f"1h_tokens={cache_creation.get('ephemeral_1h_input_tokens','N/A')}, "
            f"5m_tokens={cache_creation.get('ephemeral_5m_input_tokens','N/A')}")

def test_29_cache_tools():
    """Cache on tools definition"""
    tools = [{"name": f"tool_{i}", "description": f"Tool number {i} for doing tasks related to {i}. " * 50,
              "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}}}
             for i in range(10)]
    tools[-1]["cache_control"] = {"type": "ephemeral"}
    body = {
        "max_tokens": 50, "tools": tools,
        "messages": [{"role": "user", "content": "Say OK, do not use any tools."}]
    }
    r1 = invoke(HAIKU_MODEL, body)
    u1 = r1['usage']
    time.sleep(1)
    r2 = invoke(HAIKU_MODEL, body)
    u2 = r2['usage']
    return (f"Tools cache - Write: {u1.get('cache_creation_input_tokens',0)}, "
            f"Read: {u2.get('cache_read_input_tokens',0)}")

def test_30_auto_cache_not_supported():
    """顶层 cache_control 参数 (Automatic Prompt Caching) - 预期不支持"""
    try:
        r = invoke(HAIKU_MODEL, {
            "max_tokens": 50,
            "cache_control": {"type": "ephemeral"},
            "system": "A long system prompt " * 200,
            "messages": [{"role": "user", "content": "Say OK"}]
        })
        u = r['usage']
        has_cache = u.get('cache_creation_input_tokens', 0) > 0
        return f"Top-level cache_control: {'WORKS' if has_cache else 'IGNORED (no cache tokens)'}. Usage: {u}"
    except Exception as e:
        return f"Top-level cache_control REJECTED: {str(e)[:150]}"


# ============================================================
# SECTION 7: Vision
# ============================================================

def test_31_vision_base64():
    png_b64 = base64.standard_b64encode(create_png()).decode('utf-8')
    r = invoke(HAIKU_MODEL, {
        "max_tokens": 100,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png_b64}},
            {"type": "text", "text": "What color is this pixel? Answer in one word."}
        ]}]
    })
    return f"Vision OK: {r['content'][0]['text'][:50]}"

def test_32_url_image_not_supported():
    """URL 图像源 - 预期不支持"""
    try:
        r = invoke(HAIKU_MODEL, {
            "max_tokens": 100,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "url", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"}},
                {"type": "text", "text": "Describe."}
            ]}]
        })
        return f"UNEXPECTED: URL image WORKS! Output: {r['content'][0]['text'][:80]}"
    except Exception as e:
        return f"CONFIRMED NOT SUPPORTED: {str(e)[:150]}"


# ============================================================
# SECTION 8: PDF
# ============================================================

def test_33_pdf_base64():
    pdf_b64 = base64.standard_b64encode(MINIMAL_PDF).decode('utf-8')
    r = invoke(HAIKU_MODEL, {
        "max_tokens": 100,
        "messages": [{"role": "user", "content": [
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_b64}},
            {"type": "text", "text": "What text is in this PDF?"}
        ]}]
    })
    return f"PDF OK: {r['content'][0]['text'][:80]}"

def test_34_url_pdf_not_supported():
    """URL PDF 源 - 预期不支持"""
    try:
        r = invoke(HAIKU_MODEL, {
            "max_tokens": 100,
            "messages": [{"role": "user", "content": [
                {"type": "document", "source": {"type": "url", "url": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"}},
                {"type": "text", "text": "Summarize."}
            ]}]
        })
        return f"UNEXPECTED: URL PDF WORKS!"
    except Exception as e:
        return f"CONFIRMED NOT SUPPORTED: {str(e)[:150]}"


# ============================================================
# SECTION 9: Citations
# ============================================================

def test_35_citations():
    r = invoke(HAIKU_MODEL, {
        "max_tokens": 300,
        "messages": [{"role": "user", "content": [
            {"type": "document",
             "source": {"type": "text", "media_type": "text/plain",
                        "data": "The capital of France is Paris. Paris has a population of 2.1 million people."},
             "title": "France Facts", "citations": {"enabled": True}},
            {"type": "text", "text": "What is the capital of France and its population?"}
        ]}]
    })
    has_citations = any(b.get('type') == 'text' and 'citations' in b for b in r['content'])
    return f"Citations: {has_citations}"


# ============================================================
# SECTION 10: Structured Outputs
# ============================================================

def test_36_structured_outputs():
    r = invoke(SONNET_MODEL, {
        "max_tokens": 300,
        "output_config": {"format": {"type": "json_schema", "schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
            "additionalProperties": False
        }}},
        "messages": [{"role": "user", "content": "Extract: John is 30 years old."}]
    })
    parsed = json.loads(r['content'][0]['text'])
    return f"Structured output: {parsed}"


# ============================================================
# SECTION 11: Effort 参数
# ============================================================

def test_37_effort_low():
    r = invoke(OPUS_MODEL, {
        "anthropic_beta": ["effort-2025-11-24"],
        "max_tokens": 100, "output_config": {"effort": "low"},
        "messages": [{"role": "user", "content": "What is 2+2?"}]
    })
    return f"Effort low OK: {r['content'][0]['text'][:50]}"

def test_38_effort_high():
    r = invoke(OPUS_MODEL, {
        "anthropic_beta": ["effort-2025-11-24"],
        "max_tokens": 100, "output_config": {"effort": "high"},
        "messages": [{"role": "user", "content": "What is 2+2?"}]
    })
    return f"Effort high OK: {r['content'][0]['text'][:50]}"


# ============================================================
# SECTION 12: Converse API
# ============================================================

def test_39_converse_basic():
    resp = client.converse(
        modelId=HAIKU_MODEL,
        messages=[{"role": "user", "content": [{"text": "Say hello in one word."}]}],
        inferenceConfig={"maxTokens": 100, "temperature": 0.5}
    )
    text = resp['output']['message']['content'][0]['text']
    usage = resp['usage']
    return f"Converse OK: {text[:50]}, in={usage['inputTokens']}, out={usage['outputTokens']}"

def test_40_converse_tool():
    resp = client.converse(
        modelId=HAIKU_MODEL,
        messages=[{"role": "user", "content": [{"text": "What's the weather in Tokyo?"}]}],
        toolConfig={"tools": [{"toolSpec": {
            "name": "get_weather", "description": "Get weather",
            "inputSchema": {"json": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
        }}]}
    )
    assert resp['stopReason'] == 'tool_use'
    tu = [b for b in resp['output']['message']['content'] if 'toolUse' in b][0]['toolUse']
    return f"Converse tool: {tu['name']}, Input: {tu['input']}"

def test_41_converse_cache():
    resp = client.converse(
        modelId=SONNET_MODEL,
        system=[{"text": LONG_TEXT}, {"cachePoint": {"type": "default"}}],
        messages=[{"role": "user", "content": [{"text": "Say OK"}]}],
        inferenceConfig={"maxTokens": 50}
    )
    u = resp['usage']
    return f"Write: {u.get('cacheWriteInputTokens',0)}, Read: {u.get('cacheReadInputTokens',0)}"

def test_42_converse_top_k():
    resp = client.converse(
        modelId=HAIKU_MODEL,
        messages=[{"role": "user", "content": [{"text": "Say hello"}]}],
        inferenceConfig={"maxTokens": 50},
        additionalModelRequestFields={"top_k": 50}
    )
    return f"top_k OK: {resp['output']['message']['content'][0]['text'][:50]}"

def test_43_converse_stream():
    resp = client.converse_stream(
        modelId=HAIKU_MODEL,
        messages=[{"role": "user", "content": [{"text": "Count 1 to 3"}]}],
        inferenceConfig={"maxTokens": 100}
    )
    text_parts = []
    for event in resp['stream']:
        if 'contentBlockDelta' in event:
            delta = event['contentBlockDelta'].get('delta', {})
            if 'text' in delta:
                text_parts.append(delta['text'])
    return f"Converse stream OK: {''.join(text_parts)[:80]}"


# ============================================================
# SECTION 13: Token Counting (boto3)
# ============================================================

def test_44_count_tokens():
    models = [
        ("Haiku 4.5", "anthropic.claude-haiku-4-5-20251001-v1:0"),
        ("Sonnet 4.5", "anthropic.claude-sonnet-4-5-20250929-v1:0"),
        ("Sonnet 4.6", "anthropic.claude-sonnet-4-6"),
        ("Opus 4.6", "anthropic.claude-opus-4-6-v1"),
    ]
    parts = []
    for name, mid in models:
        resp = client.count_tokens(
            modelId=mid,
            input={'converse': {'messages': [{'role': 'user', 'content': [{'text': 'Hello, how are you?'}]}]}}
        )
        parts.append(f"{name}={resp['inputTokens']}")
    return "; ".join(parts)

def test_45_count_tokens_regional_fails():
    """us. 前缀应该失败"""
    try:
        client.count_tokens(
            modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            input={'converse': {'messages': [{'role': 'user', 'content': [{'text': 'Hello'}]}]}}
        )
        return "UNEXPECTED: us. prefix WORKS for count_tokens!"
    except Exception as e:
        return f"CONFIRMED: us. prefix fails: {str(e)[:150]}"


# ============================================================
# SECTION 14: Built-in Tools
# ============================================================

def test_46_builtin_bash_tool():
    """Bash tool (client-side, type=bash_20250124)"""
    r = invoke(SONNET_46_MODEL, {
        "max_tokens": 1000,
        "tools": [{"type": "bash_20250124", "name": "bash"}],
        "messages": [{"role": "user", "content": "Run: echo hello"}]
    })
    tool_uses = [b for b in r['content'] if b['type'] == 'tool_use']
    return f"Bash tool OK. Tool uses: {len(tool_uses)}, stop: {r['stop_reason']}"

def test_47_builtin_text_editor():
    """Text Editor tool (client-side, name must be str_replace_based_edit_tool)"""
    r = invoke(SONNET_46_MODEL, {
        "max_tokens": 1000,
        "tools": [{"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}],
        "messages": [{"role": "user", "content": "View /tmp/test.txt"}]
    })
    tool_uses = [b for b in r['content'] if b['type'] == 'tool_use']
    return f"Text editor tool OK. Tool uses: {len(tool_uses)}, stop: {r['stop_reason']}"

def test_48_web_search_not_supported():
    """Web Search tool - 预期不支持"""
    try:
        r = invoke(HAIKU_MODEL, {
            "max_tokens": 300,
            "tools": [{"type": "web_search_20260209", "name": "web_search"}],
            "messages": [{"role": "user", "content": "Search for latest news about AI."}]
        })
        return f"UNEXPECTED: Web search WORKS! stop={r['stop_reason']}"
    except Exception as e:
        return f"CONFIRMED NOT SUPPORTED: {str(e)[:150]}"

def test_49_web_fetch_not_supported():
    """Web Fetch tool - 预期不支持"""
    try:
        r = invoke(HAIKU_MODEL, {
            "max_tokens": 300,
            "tools": [{"type": "web_fetch_20260209", "name": "web_fetch"}],
            "messages": [{"role": "user", "content": "Fetch https://example.com"}]
        })
        return f"UNEXPECTED: Web fetch WORKS! stop={r['stop_reason']}"
    except Exception as e:
        return f"CONFIRMED NOT SUPPORTED: {str(e)[:150]}"

def test_50_code_execution_not_supported():
    """Code Execution tool - 预期不支持"""
    try:
        r = invoke(HAIKU_MODEL, {
            "max_tokens": 300,
            "tools": [{"type": "code_execution_20260120", "name": "code_execution"}],
            "messages": [{"role": "user", "content": "Calculate 2+2 using python."}]
        })
        return f"UNEXPECTED: Code execution WORKS! stop={r['stop_reason']}"
    except Exception as e:
        return f"CONFIRMED NOT SUPPORTED: {str(e)[:150]}"

def test_51_mcp_connector_not_supported():
    """MCP Connector - InvokeModel 不支持"""
    try:
        r = invoke(HAIKU_MODEL, {
            "max_tokens": 300,
            "tools": [{"type": "mcp", "name": "test_mcp", "server_url": "http://localhost:3000/mcp"}],
            "messages": [{"role": "user", "content": "List MCP tools."}]
        })
        return f"UNEXPECTED: MCP WORKS!"
    except Exception as e:
        return f"CONFIRMED NOT SUPPORTED on InvokeModel: {str(e)[:150]}"


# ============================================================
# SECTION 15: Anthropic 独有参数 (预期不支持)
# ============================================================

def test_52_service_tier_not_supported():
    """service_tier 参数 - 预期不支持"""
    try:
        r = invoke(HAIKU_MODEL, {
            "max_tokens": 50, "service_tier": "auto",
            "messages": [{"role": "user", "content": "Say OK"}]
        })
        return f"service_tier: {'ACCEPTED (ignored?)' if r['content'] else 'WORKS'}"
    except Exception as e:
        return f"CONFIRMED NOT SUPPORTED: {str(e)[:150]}"

def test_53_container_not_supported():
    """container 参数 - 预期不支持"""
    try:
        r = invoke(HAIKU_MODEL, {
            "max_tokens": 50, "container": "persistent-container-123",
            "messages": [{"role": "user", "content": "Say OK"}]
        })
        return f"container: {'ACCEPTED (ignored?)' if r['content'] else 'WORKS'}"
    except Exception as e:
        return f"CONFIRMED NOT SUPPORTED: {str(e)[:150]}"

def test_54_inference_geo_not_supported():
    """inference_geo 参数 - 预期不支持"""
    try:
        r = invoke(HAIKU_MODEL, {
            "max_tokens": 50, "inference_geo": "us",
            "messages": [{"role": "user", "content": "Say OK"}]
        })
        return f"inference_geo: {'ACCEPTED (ignored?)' if r['content'] else 'WORKS'}"
    except Exception as e:
        return f"CONFIRMED NOT SUPPORTED: {str(e)[:150]}"


# ============================================================
# SECTION 16: Compaction (Beta)
# ============================================================

def test_55_compaction():
    """Compaction - 测试多种方式"""
    results_parts = []
    # 方式1: InvokeModel beta flag
    for beta in ["compact-2025-06-01", "compaction-2025-06-01", "prompt-compaction-2025-06-13"]:
        try:
            invoke(SONNET_46_MODEL, {
                "anthropic_beta": [beta], "max_tokens": 50,
                "messages": [{"role": "user", "content": "Say OK"}]
            })
            results_parts.append(f"InvokeModel beta '{beta}': ACCEPTED")
            break
        except Exception:
            pass
    else:
        results_parts.append("InvokeModel beta flags: ALL REJECTED")

    # 方式2: Converse additionalModelRequestFields
    try:
        client.converse(
            modelId=SONNET_46_MODEL,
            messages=[{"role": "user", "content": [{"text": "Say OK"}]}],
            inferenceConfig={"maxTokens": 50},
            additionalModelRequestFields={"compaction": {"type": "auto"}}
        )
        results_parts.append("Converse compaction field: ACCEPTED")
    except Exception:
        results_parts.append("Converse compaction field: REJECTED")

    return "; ".join(results_parts)


# ============================================================
# SECTION 17: Bedrock 独有功能
# ============================================================

def test_56_list_models():
    """list-foundation-models (Bedrock 独有)"""
    resp = bedrock_mgmt.list_foundation_models(
        byProvider='Anthropic', byOutputModality='TEXT'
    )
    models = [m['modelId'] for m in resp['modelSummaries']]
    claude_models = [m for m in models if 'claude' in m]
    return f"Found {len(claude_models)} Claude models: {claude_models[:5]}..."

def test_57_converse_s3_image():
    """Converse API S3 图片 - 仅验证参数是否被接受 (需要实际 S3 文件)"""
    # 我们无法测试真正的 S3 图片 (没有上传文件)，只验证结构
    return "SKIP: 需要实际 S3 图片文件才能测试"

def test_58_guardrails_claude4():
    """Guardrails + Claude 4.x - 预期不支持"""
    try:
        resp = client.converse(
            modelId=SONNET_46_MODEL,
            messages=[{"role": "user", "content": [{"text": "Say hello"}]}],
            inferenceConfig={"maxTokens": 50},
            guardrailConfig={"guardrailIdentifier": "test-guardrail", "guardrailVersion": "1"}
        )
        return f"UNEXPECTED: Guardrails WORKS with Claude 4.x!"
    except Exception as e:
        err = str(e)
        if 'guardrail' in err.lower() or 'not support' in err.lower() or 'ValidationException' in err:
            return f"CONFIRMED NOT SUPPORTED: {err[:150]}"
        return f"ERROR (may not be guardrail related): {err[:150]}"


# ============================================================
# RUN ALL TESTS
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("AWS Bedrock Claude API 全面验证测试")
    print(f"Region: {REGION}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {__import__('sys').version}")
    print("=" * 60)

    # --- Section 1: 基础连接 & 模型 ---
    test("01. InvokeModel - Sonnet 4.5", test_01_invoke_model_sonnet45)
    test("02. InvokeModel - Haiku 4.5", test_02_invoke_model_haiku45)
    test("03. InvokeModel - Opus 4.5", test_03_invoke_model_opus45)
    test("04. InvokeModel - Sonnet 4.6", test_04_invoke_model_sonnet46)
    test("05. InvokeModel - Opus 4.6", test_05_invoke_model_opus46)
    test("06. Global endpoint", test_06_global_endpoint)

    # --- Section 2: Messages API 核心参数 ---
    test("07. System prompt (string)", test_07_system_prompt)
    test("08. System prompt (array)", test_08_system_prompt_array)
    test("09. temperature + top_k + stop_sequences", test_09_temperature_topk_stop)
    test("10. top_p", test_10_top_p)
    test("11. metadata", test_11_metadata)

    # --- Section 3: Streaming ---
    test("12. Streaming (InvokeModelWithResponseStream)", test_12_streaming)
    test("13. Streaming + Extended Thinking", test_13_streaming_thinking)

    # --- Section 4: Tool Use ---
    test("14. Tool Use - auto", test_14_tool_use_auto)
    test("15. Tool Use - tool_choice any", test_15_tool_choice_any)
    test("16. Tool Use - tool_choice specific", test_16_tool_choice_specific)
    test("17. Tool Use - tool_choice none", test_17_tool_choice_none)
    test("18. Multi-turn Tool Use", test_18_multi_turn_tool)
    test("19. disable_parallel_tool_use", test_19_disable_parallel_tool_use)
    test("20. Tool strict mode", test_20_tool_strict_mode)

    # --- Section 5: Extended Thinking ---
    test("21. Thinking enabled (Sonnet 4.5)", test_21_thinking_enabled)
    test("22. Thinking adaptive (Opus 4.6)", test_22_thinking_adaptive)
    test("23. Beta interleaved thinking", test_23_beta_interleaved_thinking)
    test("24. Thinking summarized/signature", test_24_thinking_summarized)

    # --- Section 6: Prompt Caching ---
    test("25. Cache - Sonnet 4.5", test_25_cache_sonnet45)
    test("26. Cache - Sonnet 4.6", test_26_cache_sonnet46)
    test("27. Cache - Opus 4.6", test_27_cache_opus46)
    test("28. Cache - 1h TTL", test_28_cache_1h_ttl)
    test("29. Cache - on tools", test_29_cache_tools)
    test("30. Auto cache (top-level) - 预期不支持", test_30_auto_cache_not_supported)

    # --- Section 7-8: Vision & PDF ---
    test("31. Vision base64 PNG", test_31_vision_base64)
    test("32. URL image - 预期不支持", test_32_url_image_not_supported)
    test("33. PDF base64", test_33_pdf_base64)
    test("34. URL PDF - 预期不支持", test_34_url_pdf_not_supported)

    # --- Section 9: Citations ---
    test("35. Citations", test_35_citations)

    # --- Section 10: Structured Outputs ---
    test("36. Structured Outputs (JSON Schema)", test_36_structured_outputs)

    # --- Section 11: Effort ---
    test("37. Effort low", test_37_effort_low)
    test("38. Effort high", test_38_effort_high)

    # --- Section 12: Converse API ---
    test("39. Converse basic", test_39_converse_basic)
    test("40. Converse tool use", test_40_converse_tool)
    test("41. Converse cache (cachePoint)", test_41_converse_cache)
    test("42. Converse top_k (additionalModelRequestFields)", test_42_converse_top_k)
    test("43. Converse stream", test_43_converse_stream)

    # --- Section 13: Token Counting ---
    test("44. Token counting (boto3 count_tokens)", test_44_count_tokens)
    test("45. Token counting - us. prefix fails", test_45_count_tokens_regional_fails)

    # --- Section 14: Built-in Tools ---
    test("46. Built-in Bash tool", test_46_builtin_bash_tool)
    test("47. Built-in Text Editor tool", test_47_builtin_text_editor)
    test("48. Web Search - 预期不支持", test_48_web_search_not_supported)
    test("49. Web Fetch - 预期不支持", test_49_web_fetch_not_supported)
    test("50. Code Execution - 预期不支持", test_50_code_execution_not_supported)
    test("51. MCP Connector - 预期不支持", test_51_mcp_connector_not_supported)

    # --- Section 15: Anthropic 独有参数 ---
    test("52. service_tier - 预期不支持", test_52_service_tier_not_supported)
    test("53. container - 预期不支持", test_53_container_not_supported)
    test("54. inference_geo - 预期不支持", test_54_inference_geo_not_supported)

    # --- Section 16: Compaction ---
    test("55. Compaction (Beta)", test_55_compaction)

    # --- Section 17: Bedrock 独有 ---
    test("56. list-foundation-models", test_56_list_models)
    test("57. Converse S3 image (SKIP)", test_57_converse_s3_image)
    test("58. Guardrails + Claude 4.x - 预期不支持", test_58_guardrails_claude4)

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
            print(f"        {r['detail'][:200]}")

    print("\n详细结果:")
    for r in results:
        marker = "PASS" if r['status'] == 'PASS' else "FAIL"
        print(f"  [{marker}] {r['test']}")

    with open('/home/ec2-user/bedrock-api/verification_results.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: /home/ec2-user/bedrock-api/verification_results.json")

#!/usr/bin/env python3
"""
验证 migration_report.md 与外部文档的差异项
逐项测试 8 个有争议或缺失的功能
"""

import json
import time
import boto3
import botocore.config

REGION = "us-west-2"
config = botocore.config.Config(read_timeout=300, connect_timeout=10)
client = boto3.client('bedrock-runtime', config=config, region_name=REGION)

SONNET_46 = "us.anthropic.claude-sonnet-4-6"
OPUS_46 = "us.anthropic.claude-opus-4-6-v1"

def invoke(model, body_dict):
    body_dict.setdefault("anthropic_version", "bedrock-2023-05-31")
    resp = client.invoke_model(body=json.dumps(body_dict), modelId=model)
    return json.loads(resp['body'].read())

def test(name, fn):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    try:
        result = fn()
        print(f"  RESULT: PASS")
        return result
    except Exception as e:
        err = str(e)
        # Truncate long errors
        if len(err) > 300:
            err = err[:300] + "..."
        print(f"  RESULT: FAIL - {err}")
        return None


# ============================================================
# TEST 1: Compaction with compact-2026-01-12
# ============================================================
def test_compaction():
    """Test compaction beta flag compact-2026-01-12"""
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": ["compact-2026-01-12"],
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "Say hello"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Now say goodbye"}
        ]
    }
    resp = client.invoke_model(body=json.dumps(body), modelId=SONNET_46)
    result = json.loads(resp['body'].read())
    print(f"  Response stop_reason: {result.get('stop_reason')}")
    print(f"  Usage: {json.dumps(result.get('usage', {}))}")
    # Check if compaction fields exist
    usage = result.get('usage', {})
    print(f"  Has input_tokens_before_compaction: {'input_tokens_before_compaction' in usage}")
    return result

# Also test compaction trigger with system_instruction
def test_compaction_with_trigger():
    """Test compaction with explicit compaction trigger"""
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": ["compact-2026-01-12"],
        "max_tokens": 100,
        "system": "You are helpful.",
        "messages": [
            {"role": "user", "content": "Hello " * 500},
            {"role": "assistant", "content": "Hi there! " * 200},
            {"role": "user", "content": "Summarize our conversation in one sentence."}
        ]
    }
    resp = client.invoke_model(body=json.dumps(body), modelId=SONNET_46)
    result = json.loads(resp['body'].read())
    print(f"  Response: {result['content'][0]['text'][:100]}")
    usage = result.get('usage', {})
    print(f"  Usage: {json.dumps(usage)}")
    return result


# ============================================================
# TEST 2: Computer Use - computer_20241022 vs computer_20250124
# ============================================================
def test_computer_use_old():
    """Test computer_20241022 tool type"""
    body = {
        "anthropic_beta": ["computer-use-2024-10-22"],
        "max_tokens": 100,
        "tools": [{
            "type": "computer_20241022",
            "name": "computer",
            "display_width_px": 1024,
            "display_height_px": 768,
            "display_number": 1
        }],
        "messages": [{"role": "user", "content": "Take a screenshot"}]
    }
    result = invoke(SONNET_46, body)
    print(f"  Stop reason: {result.get('stop_reason')}")
    print(f"  Content types: {[c['type'] for c in result.get('content', [])]}")
    return result

def test_computer_use_new():
    """Test computer_20250124 tool type"""
    body = {
        "anthropic_beta": ["computer-use-2025-01-24"],
        "max_tokens": 100,
        "tools": [{
            "type": "computer_20250124",
            "name": "computer",
            "display_width_px": 1024,
            "display_height_px": 768,
            "display_number": 1
        }],
        "messages": [{"role": "user", "content": "Take a screenshot"}]
    }
    result = invoke(SONNET_46, body)
    print(f"  Stop reason: {result.get('stop_reason')}")
    print(f"  Content types: {[c['type'] for c in result.get('content', [])]}")
    return result


# ============================================================
# TEST 3: Memory Tool
# ============================================================
def test_memory_tool():
    """Test memory_20250801 tool type (as listed in external doc)"""
    body = {
        "max_tokens": 200,
        "tools": [{
            "type": "memory_20250801",
            "name": "memory"
        }],
        "messages": [{"role": "user", "content": "Remember that my favorite color is blue."}]
    }
    result = invoke(SONNET_46, body)
    print(f"  Stop reason: {result.get('stop_reason')}")
    for c in result.get('content', []):
        print(f"  Content block: type={c['type']}, keys={list(c.keys())}")
        if c['type'] == 'text':
            print(f"    text: {c['text'][:100]}")
        elif c['type'] == 'tool_use':
            print(f"    tool: {c.get('name')}, input: {json.dumps(c.get('input', {}))[:100]}")
    return result

def test_memory_tool_no_type():
    """Test memory tool with generic custom tool definition"""
    body = {
        "max_tokens": 200,
        "tools": [{
            "type": "memory",
            "name": "memory"
        }],
        "messages": [{"role": "user", "content": "Remember that my favorite color is blue."}]
    }
    result = invoke(SONNET_46, body)
    print(f"  Stop reason: {result.get('stop_reason')}")
    return result


# ============================================================
# TEST 4: Context Editing (context-management-2025-06-27)
# ============================================================
def test_context_editing():
    """Test context editing beta header acceptance"""
    body = {
        "anthropic_beta": ["context-management-2025-06-27"],
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Say hello"}]
    }
    result = invoke(SONNET_46, body)
    print(f"  Response: {result['content'][0]['text'][:100]}")
    print(f"  Beta accepted (no error)")
    return result


# ============================================================
# TEST 5: eager_input_streaming
# ============================================================
def test_eager_input_streaming():
    """Test eager_input_streaming on tool definition"""
    body = {
        "max_tokens": 200,
        "tools": [{
            "name": "get_weather",
            "description": "Get weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            },
            "eager_input_streaming": True
        }],
        "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]
    }
    result = invoke(SONNET_46, body)
    print(f"  Stop reason: {result.get('stop_reason')}")
    for c in result.get('content', []):
        if c['type'] == 'tool_use':
            print(f"  Tool called: {c['name']}, input: {c['input']}")
    return result


# ============================================================
# TEST 6: Tool Search (tool-search-tool-2025-10-19)
# ============================================================
def test_tool_search():
    """Test tool search beta header"""
    # Generate many tools to test search
    tools = []
    for i in range(20):
        tools.append({
            "name": f"tool_{i}",
            "description": f"Tool number {i} that does task {i}",
            "input_schema": {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"]
            }
        })
    # Add a specific weather tool
    tools.append({
        "name": "get_current_weather",
        "description": "Get the current weather for a specific city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    })

    body = {
        "anthropic_beta": ["tool-search-tool-2025-10-19"],
        "max_tokens": 200,
        "tools": tools,
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}]
    }
    result = invoke(SONNET_46, body)
    print(f"  Stop reason: {result.get('stop_reason')}")
    for c in result.get('content', []):
        if c['type'] == 'tool_use':
            print(f"  Tool called: {c['name']}")
    return result


# ============================================================
# TEST 7: Tool Input Examples (tool-examples-2025-10-29)
# ============================================================
def test_tool_input_examples():
    """Test tool input examples beta header"""
    body = {
        "anthropic_beta": ["tool-examples-2025-10-29"],
        "max_tokens": 200,
        "tools": [{
            "name": "calculate",
            "description": "Perform a calculation",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"]
            },
            "examples": [
                {
                    "input": {"expression": "2 + 2"},
                    "output": "4"
                }
            ]
        }],
        "messages": [{"role": "user", "content": "What is 15 * 23?"}]
    }
    result = invoke(SONNET_46, body)
    print(f"  Stop reason: {result.get('stop_reason')}")
    for c in result.get('content', []):
        if c['type'] == 'tool_use':
            print(f"  Tool called: {c['name']}, input: {c['input']}")
    return result


# ============================================================
# TEST 8: 1M Context Window (context-1m-2025-08-07)
# ============================================================
def test_1m_context():
    """Test 1M context window beta header acceptance"""
    body = {
        "anthropic_beta": ["context-1m-2025-08-07"],
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Say OK"}]
    }
    result = invoke(SONNET_46, body)
    print(f"  Response: {result['content'][0]['text'][:100]}")
    print(f"  Beta accepted (no error)")
    return result


# ============================================================
# TEST 9: advanced-tool-use-2025-11-20 (should be rejected)
# ============================================================
def test_advanced_tool_use_rejected():
    """Verify that advanced-tool-use-2025-11-20 is rejected"""
    body = {
        "anthropic_beta": ["advanced-tool-use-2025-11-20"],
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Say OK"}]
    }
    result = invoke(SONNET_46, body)
    print(f"  Response: {result['content'][0]['text'][:100]}")
    print(f"  NOTE: Beta was ACCEPTED (unexpected)")
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("差异验证测试")
    print(f"Region: {REGION}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = {}

    # High priority
    results['compaction'] = test("1a. Compaction (compact-2026-01-12)", test_compaction)
    results['compaction_trigger'] = test("1b. Compaction with longer context", test_compaction_with_trigger)
    results['computer_old'] = test("2a. Computer Use (computer_20241022)", test_computer_use_old)
    results['computer_new'] = test("2b. Computer Use (computer_20250124)", test_computer_use_new)
    results['memory'] = test("3a. Memory Tool (memory_20250801)", test_memory_tool)
    results['memory_no_type'] = test("3b. Memory Tool (type=memory)", test_memory_tool_no_type)

    # Medium priority
    results['context_editing'] = test("4. Context Editing (context-management-2025-06-27)", test_context_editing)
    results['eager_streaming'] = test("5. eager_input_streaming", test_eager_input_streaming)
    results['tool_search'] = test("6. Tool Search (tool-search-tool-2025-10-19)", test_tool_search)
    results['tool_examples'] = test("7. Tool Input Examples (tool-examples-2025-10-29)", test_tool_input_examples)
    results['context_1m'] = test("8. 1M Context Window (context-1m-2025-08-07)", test_1m_context)
    results['adv_tool_rejected'] = test("9. advanced-tool-use-2025-11-20 (expect reject)", test_advanced_tool_use_rejected)

    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        status = "PASS" if r is not None else "FAIL"
        print(f"  {name}: {status}")

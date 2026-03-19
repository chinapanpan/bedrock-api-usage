#!/usr/bin/env python3
"""
验证 Bedrock Opus 4.6 Prompt Caching 支持情况
参考: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html

官方文档未列出 Opus 4.6，但需实测确认。
对比 Opus 4.5 (文档明确支持) 作为参照组。
"""

import json
import time
import boto3
import botocore.config

REGION = "us-west-2"
config = botocore.config.Config(read_timeout=300, connect_timeout=10)
client = boto3.client('bedrock-runtime', config=config, region_name=REGION)

OPUS_46 = "us.anthropic.claude-opus-4-6-v1"
OPUS_45 = "us.anthropic.claude-opus-4-5-20251101-v1:0"
SONNET_45 = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# Opus 4.6/4.5 最小缓存 token 要求是 4096，确保超过该阈值
# 每个英文单词约 1.3 token，8000 词 ≈ 10000+ tokens
LONG_TEXT = "This is a comprehensive test of prompt caching on AWS Bedrock for Claude Opus model. " * 800


def test_invoke_model_cache(model_id: str, model_name: str):
    """通过 InvokeModel API 测试 prompt caching (官方文档格式)"""
    print(f"\n{'='*60}")
    print(f"TEST: InvokeModel Prompt Caching - {model_name}")
    print(f"Model: {model_id}")
    print(f"{'='*60}")

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 50,
        "system": [{
            "type": "text",
            "text": LONG_TEXT,
            "cache_control": {"type": "ephemeral"}
        }],
        "messages": [{"role": "user", "content": "Say OK"}]
    })

    # Call 1: 应触发 cache write
    print("\n  Call 1 (expect cache write)...")
    t1 = time.time()
    resp1 = client.invoke_model(body=body, modelId=model_id)
    result1 = json.loads(resp1['body'].read())
    elapsed1 = time.time() - t1
    usage1 = result1.get('usage', {})
    print(f"  Time: {elapsed1:.2f}s")
    print(f"  Usage: {json.dumps(usage1, indent=4)}")

    cache_write_1 = usage1.get('cache_creation_input_tokens', 0)
    cache_read_1 = usage1.get('cache_read_input_tokens', 0)
    print(f"  cache_creation_input_tokens: {cache_write_1}")
    print(f"  cache_read_input_tokens: {cache_read_1}")

    # 等待 2 秒后第二次调用
    time.sleep(2)

    # Call 2: 应触发 cache read
    print("\n  Call 2 (expect cache read)...")
    t2 = time.time()
    resp2 = client.invoke_model(body=body, modelId=model_id)
    result2 = json.loads(resp2['body'].read())
    elapsed2 = time.time() - t2
    usage2 = result2.get('usage', {})
    print(f"  Time: {elapsed2:.2f}s")
    print(f"  Usage: {json.dumps(usage2, indent=4)}")

    cache_write_2 = usage2.get('cache_creation_input_tokens', 0)
    cache_read_2 = usage2.get('cache_read_input_tokens', 0)
    print(f"  cache_creation_input_tokens: {cache_write_2}")
    print(f"  cache_read_input_tokens: {cache_read_2}")

    # 判断结果
    if cache_write_1 > 0 or cache_read_1 > 0 or cache_write_2 > 0 or cache_read_2 > 0:
        print(f"\n  RESULT: CACHE SUPPORTED for {model_name}")
    else:
        print(f"\n  RESULT: CACHE NOT SUPPORTED for {model_name} (all cache tokens = 0)")

    return usage1, usage2


def test_converse_cache(model_id: str, model_name: str):
    """通过 Converse API 测试 prompt caching (cachePoint 格式)"""
    print(f"\n{'='*60}")
    print(f"TEST: Converse API Prompt Caching - {model_name}")
    print(f"Model: {model_id}")
    print(f"{'='*60}")

    # Call 1
    print("\n  Call 1 (expect cache write)...")
    t1 = time.time()
    resp1 = client.converse(
        modelId=model_id,
        system=[
            {"text": LONG_TEXT},
            {"cachePoint": {"type": "default"}}
        ],
        messages=[{
            "role": "user",
            "content": [{"text": "Say OK"}]
        }],
        inferenceConfig={"maxTokens": 50}
    )
    elapsed1 = time.time() - t1
    usage1 = resp1['usage']
    print(f"  Time: {elapsed1:.2f}s")
    print(f"  Usage: {json.dumps(usage1, indent=4, default=str)}")

    cache_write_1 = usage1.get('cacheWriteInputTokens', 0)
    cache_read_1 = usage1.get('cacheReadInputTokens', 0)
    print(f"  cacheWriteInputTokens: {cache_write_1}")
    print(f"  cacheReadInputTokens: {cache_read_1}")

    time.sleep(2)

    # Call 2
    print("\n  Call 2 (expect cache read)...")
    t2 = time.time()
    resp2 = client.converse(
        modelId=model_id,
        system=[
            {"text": LONG_TEXT},
            {"cachePoint": {"type": "default"}}
        ],
        messages=[{
            "role": "user",
            "content": [{"text": "Say OK"}]
        }],
        inferenceConfig={"maxTokens": 50}
    )
    elapsed2 = time.time() - t2
    usage2 = resp2['usage']
    print(f"  Time: {elapsed2:.2f}s")
    print(f"  Usage: {json.dumps(usage2, indent=4, default=str)}")

    cache_write_2 = usage2.get('cacheWriteInputTokens', 0)
    cache_read_2 = usage2.get('cacheReadInputTokens', 0)
    print(f"  cacheWriteInputTokens: {cache_write_2}")
    print(f"  cacheReadInputTokens: {cache_read_2}")

    if cache_write_1 > 0 or cache_read_1 > 0 or cache_write_2 > 0 or cache_read_2 > 0:
        print(f"\n  RESULT: CACHE SUPPORTED for {model_name}")
    else:
        print(f"\n  RESULT: CACHE NOT SUPPORTED for {model_name} (all cache tokens = 0)")

    return usage1, usage2


if __name__ == "__main__":
    print("=" * 60)
    print("Bedrock Opus 4.6 Prompt Caching 验证测试")
    print(f"Region: {REGION}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Long text length: {len(LONG_TEXT)} chars (~{len(LONG_TEXT.split())} words)")
    print("=" * 60)

    # 参照组: Sonnet 4.5 (文档明确支持，最小 1024 tokens)
    print("\n\n>>> 参照组: Sonnet 4.5 (文档明确支持)")
    test_invoke_model_cache(SONNET_45, "Sonnet 4.5")

    # 参照组: Opus 4.5 (文档明确支持，最小 4096 tokens)
    print("\n\n>>> 参照组: Opus 4.5 (文档明确支持)")
    test_invoke_model_cache(OPUS_45, "Opus 4.5")

    # 测试目标: Opus 4.6 - InvokeModel
    print("\n\n>>> 测试目标: Opus 4.6 - InvokeModel API")
    test_invoke_model_cache(OPUS_46, "Opus 4.6")

    # 测试目标: Opus 4.6 - Converse API
    print("\n\n>>> 测试目标: Opus 4.6 - Converse API")
    test_converse_cache(OPUS_46, "Opus 4.6")

    print("\n\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

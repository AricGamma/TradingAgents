#!/usr/bin/env python3
"""
测试百炼 API 响应格式
验证 result.content 是否正确返回
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# 设置百炼 API Key
os.environ["OPENAI_API_KEY"] = "sk-88d6237c8d284482a866ff0c8911f528"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

from langchain_openai import ChatOpenAI

# 创建与 TradingAgents 相同的配置
llm = ChatOpenAI(
    model="qwen3.5-flash",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.7,
)

print("=" * 60)
print("测试百炼 API 响应格式")
print("=" * 60)

# 测试简单消息
print("\n1. 测试简单消息响应...")
try:
    response = llm.invoke("你好，请用中文回答：1+1 等于多少？")
    print(f"   响应类型：{type(response)}")
    print(f"   response.content 类型：{type(response.content)}")
    print(f"   response.content 内容：{repr(response.content[:100] if response.content else 'EMPTY')}")
    print(f"   response.additional_kwargs: {response.additional_kwargs}")
except Exception as e:
    print(f"   错误：{e}")

# 测试带工具的消息
print("\n2. 测试带工具的消息响应...")

def dummy_tool(query: str) -> str:
    """一个简单的工具，返回查询结果"""
    return f"查询结果：{query}"

try:
    llm_with_tools = llm.bind_tools([dummy_tool])
    response = llm_with_tools.invoke("请调用 dummy_tool 工具查询 'META 股票'")
    print(f"   响应类型：{type(response)}")
    print(f"   response.content 类型：{type(response.content)}")
    print(f"   response.content 内容：{repr(response.content[:100] if response.content else 'EMPTY')}")
    print(f"   response.tool_calls: {response.tool_calls if hasattr(response, 'tool_calls') else 'N/A'}")
except Exception as e:
    print(f"   错误：{e}")

# 测试多轮对话（模拟 TradingAgents 的工作流程）
print("\n3. 测试多轮对话响应...")
try:
    messages = [
        ("system", "你是一个金融分析师助手。"),
        ("human", "请分析 META 公司的财务状况。"),
    ]
    response = llm.invoke(messages)
    print(f"   响应类型：{type(response)}")
    print(f"   response.content 类型：{type(response.content)}")
    print(f"   response.content 长度：{len(response.content) if response.content else 0}")
    print(f"   response.content 前 200 字符：{repr(response.content[:200] if response.content else 'EMPTY')}")
except Exception as e:
    print(f"   错误：{e}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)

#!/usr/bin/env python3
"""
运行 BABA 股票分析报告
使用阿里云百炼作为 LLM 提供商
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 设置百炼 API Key
os.environ["OPENAI_API_KEY"] = "sk-88d6237c8d284482a866ff0c8911f528"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# 复制默认配置并修改
config = DEFAULT_CONFIG.copy()

# 使用 OpenAI 兼容模式 (百炼)
config["llm_provider"] = "openai"
config["backend_url"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 使用百炼的 Qwen 模型
config["deep_think_llm"] = "qwen3.5-plus"
config["quick_think_llm"] = "qwen3.5-flash"

# 设置最大辩论轮次，确保完整输出
config["max_debate_rounds"] = 1
config["max_risk_discuss_rounds"] = 1

# 数据源使用 yfinance
config["data_vendors"] = {
    "core_stock_apis": "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data": "yfinance",
    "news_data": "yfinance",
}

# 设置结果目录
config["results_dir"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

# 启用调试模式
debug = True

print("=" * 60)
print("TradingAgents - BABA 股票分析")
print("=" * 60)
print(f"分析时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"LLM 提供商：{config['llm_provider']} (百炼兼容模式)")
print(f"API Endpoint: {config['backend_url']}")
print(f"深度思考模型：{config['deep_think_llm']}")
print(f"快速思考模型：{config['quick_think_llm']}")
print(f"结果目录：{config['results_dir']}")
print("=" * 60)

# 初始化 TradingAgentsGraph
print("\n初始化 TradingAgents...")
ta = TradingAgentsGraph(debug=debug, config=config)

# 运行分析 - 使用最近交易日
analysis_date = "2026-04-09"
ticker = "BABA"

print(f"\n开始分析 {ticker} (分析日期：{analysis_date})")
print("-" * 60)

try:
    # 执行分析
    state, decision = ta.propagate(ticker, analysis_date)
    
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)
    print(f"\n最终决策：{decision}")
    
    # 打印所有子报告
    print("\n" + "=" * 60)
    print("所有子报告输出状态")
    print("=" * 60)
    
    if state:
        # 检查各个报告部分
        report_sections = [
            "market_report",
            "sentiment_report", 
            "news_report",
            "fundamentals_report",
            "investment_plan",
            "trader_investment_plan",
            "portfolio_report",
            "final_trade_decision"
        ]
        
        for section in report_sections:
            if section in state:
                content = state[section]
                if content:
                    print(f"✓ {section}: {len(str(content))} 字符")
                else:
                    print(f"✗ {section}: 空内容")
            else:
                print(f"✗ {section}: 未在 state 中找到")
    
    print("\n" + "=" * 60)
    print("报告已保存到 reports/ 目录")
    print("=" * 60)
    
except Exception as e:
    print(f"\n错误：{e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

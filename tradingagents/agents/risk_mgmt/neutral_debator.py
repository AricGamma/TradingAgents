import time
import json


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_aggressive_response = risk_debate_state.get(
            "current_aggressive_response", ""
        )
        current_conservative_response = risk_debate_state.get(
            "current_conservative_response", ""
        )

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""作为一名中立的风险分析师，你的角色是为交易员的决策或计划提供平衡的观点，权衡潜在的利益和风险。你优先考虑全面的方法，评估上行和下限，同时考虑更广泛的市场趋势、潜在的经济变化和多元化策略。以下是交易员的决策：

{trader_decision}

你的任务是挑戢激进派和保守派分析师，指出每种观点可能过于乐观或过于谨慎的地方。利用以下数据来源的见解来支持一个适度、可持续的策略来调整交易员的决策：

市场研究报告：{market_research_report}
社交媒体情绪报告：{sentiment_report}
最新国际事务报告：{news_report}
公司基本面报告：{fundamentals_report}
以下是当前的对话历史：{history} 以下是激进派分析师的最新回应：{current_aggressive_response} 以下是保守派分析师的最新回应：{current_conservative_response}。如果其他观点还没有回应，请基于现有数据提出你自己的论点。

通过批判性地分析双方，解决激进和保守论点中的弱点，倡导更平衡的方法。挑战他们的每一个观点，说明为什么适度的风险策略可以提供两全其美的效果，在保护免受极端波动的同时提供增长潜力。专注于辩论而不是简单地展示数据，旨在表明平衡的观点可以带来最可靠的结果。像说话一样输出，不要使用特殊的格式。

**重要提示：请使用中文回复。**"""

        response = llm.invoke(prompt)

        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": neutral_history + "\n" + argument,
            "latest_speaker": "Neutral",
            "current_aggressive_response": risk_debate_state.get(
                "current_aggressive_response", ""
            ),
            "current_conservative_response": risk_debate_state.get(
                "current_conservative_response", ""
            ),
            "current_neutral_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node

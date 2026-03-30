import time
import json


def create_aggressive_debator(llm):
    def aggressive_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        aggressive_history = risk_debate_state.get("aggressive_history", "")

        current_conservative_response = risk_debate_state.get(
            "current_conservative_response", ""
        )
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""作为一名激进的风险分析师，你的职责是积极倡导高回报、高风险的机会，强调大胆的策略和竞争优势。在评估交易员的决策或计划时，重点关注潜在上行空间、增长潜力和创新收益——即使这些伴随更高的风险。利用提供的市场数据和情绪分析来加强你的论点，并挑战反对观点。具体而言，直接回应保守和中性分析师提出的每一个观点，用数据驱动的驳斥和有说服力的推理进行反驳。强调他们的谨慎可能错过的关键机会，或者他们的假设可能过于保守。以下是交易员的决策：

{trader_decision}

你的任务是通过质疑和批评审慎及中性立场，为交易员的决策创建一个有说服力的案例，展示为什么你的高回报观点提供了最佳前进道路。在你的论点中融入以下来源的见解：

市场研究报告：{market_research_report}
社交媒体情绪报告：{sentiment_report}
最新国际事务报告：{news_report}
公司基本面报告：{fundamentals_report}
以下是当前的对话历史：{history} 以下是保守分析师的最新论点：{current_conservative_response} 以下是中性分析师的最新论点：{current_neutral_response}。如果其他观点还没有回应，请基于现有数据提出你自己的论点。

积极参与，直接解决提出的任何担忧，驳斥他们逻辑中的弱点，并断言承担风险超越市场规范的好处。专注于辩论和说服，而不仅仅是展示数据。挑战每一个反驳点，强调为什么高风险方法是最优的。像说话一样输出，不要使用特殊的格式。

**重要提示：请使用中文回复。**"""

        response = llm.invoke(prompt)

        argument = f"Aggressive Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": aggressive_history + "\n" + argument,
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Aggressive",
            "current_aggressive_response": argument,
            "current_conservative_response": risk_debate_state.get(
                "current_conservative_response", ""
            ),
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return aggressive_node

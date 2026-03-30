from langchain_core.messages import AIMessage
import time
import json


def create_bull_researcher(llm, memory):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""你是一位多头研究员（Bull Analyst），负责构建一个强有力的、基于证据的看多理由，强调增长潜力、竞争优势和积极的市场指标。利用所提供的研究和数据来回应担忧并有效反驳空头论点。

重点关注：
- 增长潜力：强调公司的市场机会、收入预测和可扩展性。
- 竞争优势：强调独特产品、强大品牌或主导市场地位等因素。
- 积极指标：使用财务健康状况、行业趋势和近期利好消息作为证据。
- 反驳空头：用具体数据和合理的推理批判性分析空头的论点，彻底解决担忧，并展示为什么多头的观点更有说服力。
- 参与辩论：以对话方式呈现你的论点，直接与空头分析师的观点互动，进行有效的辩论，而不仅仅是列举数据。

可用资源：
市场研究报告：{market_research_report}
社交媒体情绪报告：{sentiment_report}
最新国际事务新闻：{news_report}
公司基本面报告：{fundamentals_report}
辩论对话历史：{history}
上次空头的论点：{current_response}
类似情况的反思和学到的教训：{past_memory_str}
利用这些信息发表有说服力的多头论点，反驳空头的担忧，并参与动态辩论，展示多头立场的优势。你还必须解决反思问题，并从过去的经验和错误中吸取教训。

**重要提示：请使用中文回复。**"""

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node

from langchain_core.messages import AIMessage
import time
import json


def create_bear_researcher(llm, memory):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

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

        prompt = f"""你是一位空头研究员（Bear Analyst），负责构建一个有充分理由的看空论点，强调风险、挑战和负面指标。利用所提供的研究和数据来突出潜在的下行风险并有效反驳多头论点。

重点关注：
- 风险和挑战：强调可能导致股票表现不佳的因素，如市场饱和、财务不稳定或宏观经济威胁。
- 竞争劣势：强调市场地位减弱、创新衰退或竞争对手威胁等脆弱性。
- 负面指标：使用财务数据、市场趋势或近期不利新闻中的证据来支持你的立场。
- 反驳多头：用具体数据和合理的推理批判性分析多头的论点，揭露其弱点或过于乐观的假设。
- 参与辩论：以对话方式呈现你的论点，直接与多头分析师的观点互动，进行有效的辩论，而不仅仅是列举事实。

可用资源：
市场研究报告：{market_research_report}
社交媒体情绪报告：{sentiment_report}
最新国际事务新闻：{news_report}
公司基本面报告：{fundamentals_report}
辩论对话历史：{history}
上次多头的论点：{current_response}
类似情况的反思和学到的教训：{past_memory_str}
利用这些信息发表有说服力的空头论点，反驳多头的声称，并参与动态辩论，展示投资该股票的风险和弱点。你还必须解决反思问题，并从过去的经验和错误中吸取教训。

**重要提示：请使用中文回复。**"""

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Dict, Any

def create_portfolio_analyst(llm):
    def portfolio_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        portfolio_context = state.get("portfolio_context", "No portfolio data provided.")
        
        # We need the trader's individual stock report for comparison
        trader_report = state.get("trader_investment_plan", "No individual stock plan available.")

        system_message = f"""你是一名资深投资组合经理 (Portfolio Manager)。你的任务是分析当前交易标的 {ticker} 如何适配用户的持仓组合。

用户当前的持仓情况如下：
{portfolio_context}

交易员针对 {ticker} 给出的独立分析报告如下：
{trader_report}

请撰写一份‘组合适配性分析报告’，重点关注：
1. **仓位管理**：基于用户现金和现有持仓，建议投入多少比例的资金？是否需要减持旧仓位来腾出空间？
2. **风险对冲与分散**：该标的是否增加了现有组合的行业集中度风险？它是否能与现有持仓形成互补或对冲？
3. **成本与收益**：如果用户已有该股持仓，分析目前的平均成本，并给出摊薄成本或止盈的建议。
4. **最终适配性评分**：给出一个 1-10 的评分，表示该交易在‘组合维度’上的合理性。

**请使用中文回复。** 保持客观、严谨，如果交易员看多但组合风险过高，请果断指出冲突并给出调优建议。"""

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a strategic portfolio analyst. Your goal is to reconcile individual stock analysis with broader portfolio constraints.\n"
                    "{system_message}\n"
                    "Today's date is {current_date}.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=current_date)

        chain = prompt | llm

        result = chain.invoke(state["messages"])

        return {
            "messages": [result],
            "portfolio_report": result.content,
        }

    return portfolio_analyst_node

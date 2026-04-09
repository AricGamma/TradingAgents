from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolMessage
import time
import json
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_insider_transactions,
)
from tradingagents.dataflows.config import get_config


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        system_message = (
            "你是一名研究员，负责分析公司过去一周的基本面信息。请撰写一份全面的报告，涵盖公司的基本面信息，如财务文件、公司简介、基本公司财务数据和公司财务历史，以全面了解公司的基本面信息，为交易者提供参考。确保包含尽可能多的细节。提供具体、可操作的洞察，辅以支持证据，帮助交易者做出知情决策。**请使用中文回复。**"
            + " 在报告末尾附上一个 Markdown 表格，整理报告中的关键点，使其有条理且易于阅读。"
            + " 使用可用工具：`get_fundamentals` 用于全面公司分析，`get_balance_sheet`、`get_cashflow` 和 `get_income_statement` 用于具体财务报表。",
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        # ============================================
        # 多轮工具调用循环 - 修复百炼兼容性问题
        # ============================================
        # 问题：当 LLM 调用工具时，response.content 为空字符串
        # 解决：实现多轮调用，执行工具后将结果反馈给 LLM，直到获得最终报告
        # ============================================
        messages = list(state["messages"])
        max_iterations = 10  # 进一步增加迭代次数，确保复杂查询能完成
        iteration = 0
        report = ""
        last_result = None
        
        while iteration < max_iterations:
            iteration += 1
            result = chain.invoke(messages)
            last_result = result
            
            # 检查是否有工具调用
            tool_calls = getattr(result, 'tool_calls', [])
            
            if tool_calls:
                # LLM 请求调用工具，执行工具并将结果添加回消息
                messages.append(result)  # 添加 AI 消息（包含 tool_calls）
                
                # 执行每个工具调用
                for tool_call in tool_calls:
                    tool_name = tool_call.get('name')
                    tool_args = tool_call.get('args', {})
                    tool_id = tool_call.get('id')
                    
                    # 查找并执行工具
                    tool_result = None
                    for tool in tools:
                        if tool.name == tool_name:
                            try:
                                tool_result = tool.invoke(tool_args)
                            except Exception as e:
                                tool_result = f"Error executing tool {tool_name}: {str(e)}"
                            break
                    
                    if tool_result is not None:
                        # 添加工具结果到消息
                        messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_id))
                
                # 继续下一轮迭代，让 LLM 基于工具结果生成报告
                continue
            else:
                # 没有工具调用，LLM 返回了最终报告
                report = result.content if result.content else ""
                messages.append(result)
                break
        
        # 如果达到最大迭代次数仍未获得报告，尝试使用最后一次响应的内容
        if not report and last_result:
            report = last_result.content if last_result.content else ""
            # 确保最后一条消息是 AIMessage（不是 ToolMessage），以便后续条件判断
            if not isinstance(last_result, ToolMessage):
                messages.append(last_result)

        # 重要：确保返回的 messages 最后一条是 AIMessage
        # 这样 conditional_logic.py 中的 last_message.tool_calls 检查才能正常工作
        # 如果最后一条是 ToolMessage，需要添加一个空的 AIMessage 作为标记
        if messages and isinstance(messages[-1], ToolMessage):
            from langchain_core.messages import AIMessage
            messages.append(AIMessage(content=report if report else "Analysis completed."))

        return {
            "messages": messages,
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node

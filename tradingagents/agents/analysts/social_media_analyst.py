from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolMessage
import time
import json
from tradingagents.agents.utils.agent_utils import build_instrument_context, get_news
from tradingagents.dataflows.config import get_config


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_news,
        ]

        system_message = (
            "你是一名社交媒体和公司特定新闻的研究员/分析师，负责分析社交媒体帖子、近期公司新闻和公众情绪。请撰写一份全面的长篇报告，详细分析、洞察和 implications，帮助交易者和投资者了解该公司的当前状态。分析社交媒体和人们对该公司的看法，分析人们每天对公司的情绪数据，并查看近期公司新闻。使用 get_news(query, start_date, end_date) 工具搜索公司特定新闻和社交媒体讨论。尽量查看所有来源，从社交媒体到情绪到新闻。提供具体、可操作的洞察，辅以支持证据，帮助交易者做出知情决策。**请使用中文回复。**"
            + """ 在报告末尾附上一个 Markdown 表格，整理报告中的关键点，使其有条理且易于阅读。"""
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
            
            tool_calls = getattr(result, 'tool_calls', [])
            
            if tool_calls:
                messages.append(result)
                
                for tool_call in tool_calls:
                    tool_name = tool_call.get('name')
                    tool_args = tool_call.get('args', {})
                    tool_id = tool_call.get('id')
                    
                    tool_result = None
                    for tool in tools:
                        if tool.name == tool_name:
                            try:
                                tool_result = tool.invoke(tool_args)
                            except Exception as e:
                                tool_result = f"Error executing tool {tool_name}: {str(e)}"
                            break
                    
                    if tool_result is not None:
                        messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_id))
                
                continue
            else:
                report = result.content if result.content else ""
                messages.append(result)
                break
        
        if not report and last_result:
            report = last_result.content if last_result.content else ""
            if not isinstance(last_result, ToolMessage):
                messages.append(last_result)

        if messages and isinstance(messages[-1], ToolMessage):
            from langchain_core.messages import AIMessage
            messages.append(AIMessage(content=report if report else "Analysis completed."))

        return {
            "messages": messages,
            "sentiment_report": report,
        }

    return social_media_analyst_node

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_indicators,
    get_stock_data,
)
from tradingagents.dataflows.config import get_config


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_stock_data,
            get_indicators,
        ]

        system_message = (
            """你是一名交易助手，负责分析金融市场。你的职责是为给定的市场状况或交易策略选择**最相关的技术指标**。目标是最多选**8 个指标**，提供互补的洞察而不重复。类别和指标如下：

移动平均线：
- close_50_sma：50 日简单移动平均线。用途：识别中短期趋势方向，作为动态支撑/阻力。提示：它滞后于价格；与更快的指标结合使用以获得及时信号。
- close_200_sma：200 日简单移动平均线。长期趋势基准。用途：确认整体市场趋势，识别金叉/死叉。提示：反应较慢；最适合战略趋势确认。
- close_10_ema：10 日指数移动平均线。响应迅速的短期平均。用途：捕捉快速动量变化和潜在入场点。提示：在震荡市场中容易产生噪声；与其他较长平均线结合过滤假信号。

MACD 相关：
- macd：MACD：通过 EMA 差值计算动量。用途：寻找交叉和背离作为趋势变化信号。提示：在低波动或横盘市场中用其他指标确认。
- macds：MACD 信号线：MACD 线的 EMA 平滑。用途：用 MACD 线的交叉触发交易。提示：应作为更广泛策略的一部分以避免假阳性。
- macdh：MACD 柱状图：显示 MACD 线与信号线之间的差距。用途：直观显示动量强度，尽早发现背离。提示：可能波动较大；在快速变化的市场中用额外过滤器补充。

动量指标：
- rsi：RSI：测量动量以标记超买/超卖条件。用途：应用 70/30 阈值并观察背离以信号反转。提示：在强趋势中 RSI 可能保持极端；始终与趋势分析交叉检查。

波动率指标：
- boll：布林带中轨：20 日简单移动平均线，作为布林带的基础。用途：作为价格动态基准。提示：与上下轨结合，有效识别突破或反转。
- boll_ub：布林带上轨：通常比中轨高 2 个标准差。用途：信号潜在超买条件和突破区域。提示：用其他工具确认；在强趋势中价格可能沿轨运行。
- boll_lb：布林带下轨：通常比中轨低 2 个标准差。用途：指示潜在超卖条件。提示：使用额外分析以避免假反转信号。
- atr：ATR：平均真实波幅，测量波动率。用途：设置止损水平，根据当前市场波动调整仓位规模。提示：是反应性指标，因此应作为更广泛风险管理策略的一部分使用。

基于成交量的指标：
- vwma：成交量加权移动平均线：按成交量加权的移动平均。用途：通过将价格行动与成交量数据整合来确认趋势。提示：注意成交量峰值可能导致偏差；与其他成交量分析结合使用。

- 选择提供多样化和互补信息的指标。避免重复（例如，不要同时选择 rsi 和 stochrsi）。还要简要解释为什么它们适合给定的市场环境。调用工具时，请使用上述指标的准确名称，因为它们是定义的参数，否则调用将失败。请确保先调用 get_stock_data 来生成指标所需的 CSV 数据。然后使用 get_indicators 与具体的指标名称。撰写一份详细且细致的趋势报告。提供具体、可操作的洞察，辅以支持证据，帮助交易者做出知情决策。"""
            + """ 在报告末尾附上一个 Markdown 表格，整理报告中的关键点，使其有条理且易于阅读。**请使用中文回复。**"""
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
                    " You have access to the following tools: {tool_names}.\n{system_message}"
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

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node

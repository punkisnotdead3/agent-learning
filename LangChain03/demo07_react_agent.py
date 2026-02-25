import os
import sys
import json
import datetime

sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# ReAct 是什么？
#
# ReAct = Reasoning（推理）+ Acting（行动）
# 是一种让 LLM 像人一样"边想边做"的 Agent 框架。
#
# 普通 LLM 的工作方式：
#   问题 → 模型直接给答案（一步到位）
#   问题：如果需要查天气、算数学、搜数据库，模型本身做不到
#
# ReAct 的工作方式：
#   问题 → 模型先思考（Thought）
#         → 决定用什么工具（Action）
#         → 执行工具获得结果（Observation）
#         → 再思考下一步（Thought）
#         → 循环，直到能给出最终答案（Final Answer）
#
# 完整循环：
#   Question（问题）
#     ↓
#   Thought（我该怎么做？）
#     ↓
#   Action（用哪个工具）+ Action Input（传什么参数）
#     ↓
#   Observation（工具返回的结果）
#     ↓
#   Thought（根据结果继续思考）
#     ↓
#   ...（可以循环多次）
#     ↓
#   Final Answer（最终回答给用户）
#
# 类比：就像一个人解决问题的过程：
#   "今天出门要带伞吗？"
#   → 想：我需要查一下今天天气
#   → 行动：打开天气 App
#   → 观察：显示有雨
#   → 想：有雨，需要带伞
#   → 回答："需要带伞"
# ============================================================

from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    temperature=0,      # Agent 场景推荐设为 0，让模型更稳定、确定性更强
)


# ============================================================
# Part 1：定义工具（Tools）
#
# 工具就是 Agent 可以调用的函数。
# 用 @tool 装饰器把普通函数变成工具：
#   - 函数名        → 工具名称
#   - 函数的 docstring → 工具说明（模型靠这个决定什么时候用它，非常重要！）
#   - 函数参数      → 工具的输入
#   - 函数返回值    → Observation（工具执行结果）
# ============================================================

@tool
def get_weather(city: str) -> str:
    """
    查询指定城市的今日天气信息。
    当用户询问某个城市的天气、温度、是否需要带伞时使用此工具。
    输入：城市名称，例如：北京、上海、广州
    """
    # 模拟天气数据（真实场景中调用天气 API）
    weather_data = {
        "北京": "晴，气温 -2°C 到 8°C，西北风 3 级，不需要带伞",
        "上海": "小雨，气温 5°C 到 12°C，东南风 2 级，需要带伞",
        "广州": "多云，气温 15°C 到 22°C，南风 1 级，不需要带伞",
        "成都": "阴，气温 6°C 到 14°C，无持续风向，可能有小雨建议带伞",
    }
    return weather_data.get(city, f"暂无 {city} 的天气数据，请尝试其他城市")


@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式，返回计算结果。
    当用户需要做数学计算时使用此工具。
    输入：合法的数学表达式字符串，例如：'3 * (4 + 5)' 或 '100 / 4 + 28'
    """
    try:
        # 只允许数字和基本运算符，防止代码注入
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "表达式包含非法字符，只支持基本数学运算"
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算失败：{e}"


@tool
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """
    获取当前日期和时间信息。
    当用户询问现在几点、今天是几号、今天星期几时使用此工具。
    输入：时区名称，默认为 Asia/Shanghai（北京时间）
    """
    now = datetime.datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday = weekdays[now.weekday()]
    return f"当前时间：{now.strftime('%Y年%m月%d日 %H:%M:%S')}，{weekday}"


@tool
def search_employee(name: str) -> str:
    """
    在公司员工数据库中查询员工信息。
    当用户询问某位员工的职位、部门、联系方式时使用此工具。
    输入：员工姓名
    """
    # 模拟员工数据库
    employees = {
        "张三": {"职位": "后端工程师", "部门": "技术部", "邮箱": "zhangsan@company.com"},
        "李四": {"职位": "产品经理", "部门": "产品部", "邮箱": "lisi@company.com"},
        "王五": {"职位": "AI工程师", "部门": "技术部", "邮箱": "wangwu@company.com"},
    }
    if name in employees:
        info = employees[name]
        return f"{name} 的信息：职位={info['职位']}，部门={info['部门']}，邮箱={info['邮箱']}"
    return f"未找到员工：{name}"


# 把所有工具放到列表里
tools = [get_weather, calculate, get_current_time, search_employee]

# 查看工具的名称和描述（模型会读这些来决定用哪个工具）
print("=" * 55)
print("已注册的工具列表：")
print("=" * 55)
for t in tools:
    print(f"  工具名：{t.name}")
    print(f"  描述：{t.description[:50]}...")
    print()


# ============================================================
# Part 2：定义 ReAct 提示词模板
#
# ReAct Agent 需要一个特定格式的提示词，
# 告诉模型如何进行 Thought → Action → Observation 循环。
#
# 这个模板有几个固定的占位符：
#   {tools}          —— 工具清单（自动填入）
#   {tool_names}     —— 工具名称列表（自动填入）
#   {input}          —— 用户的问题
#   {agent_scratchpad} —— 模型的推理过程记录（自动维护）
# ============================================================

react_prompt = PromptTemplate.from_template("""
你是一个智能助手，尽力回答用户的问题。你可以使用以下工具：

{tools}

回答时必须严格按照以下格式（每个标签单独一行）：

Question: 用户的问题
Thought: 分析问题，决定下一步怎么做
Action: 选择一个工具名称，必须是以下之一：[{tool_names}]
Action Input: 传给工具的参数
Observation: 工具返回的结果
...（Thought/Action/Action Input/Observation 可以循环多次）
Thought: 我现在知道最终答案了
Final Answer: 给用户的最终回答

开始！

Question: {input}
Thought: {agent_scratchpad}
""")


# ============================================================
# Part 3：创建 Agent 和 AgentExecutor
#
# create_react_agent：把模型 + 工具 + 提示词组合成 Agent
# AgentExecutor：负责运行 Agent 的循环（执行工具、传回结果、继续推理）
# ============================================================

# 创建 ReAct Agent
agent = create_react_agent(
    llm=model,
    tools=tools,
    prompt=react_prompt,
)

# AgentExecutor 是 Agent 的"执行引擎"
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,           # 打印完整的推理过程（Thought/Action/Observation）
    max_iterations=5,       # 最多循环 5 次，防止无限循环
    handle_parsing_errors=True,  # 模型输出格式不对时自动处理，不直接报错
)


# ============================================================
# Part 4：运行 Agent，观察 ReAct 推理过程
# ============================================================

def run_agent(question: str):
    print("\n" + "=" * 55)
    print(f"用户问题：{question}")
    print("=" * 55)
    result = executor.invoke({"input": question})
    print(f"\n最终答案：{result['output']}")
    print()


# 示例 1：单工具调用
run_agent("上海今天需要带伞吗？")

# 示例 2：数学计算
run_agent("我买了3件衣服，分别是128元、256元和99元，一共花了多少钱？")

# 示例 3：需要多步推理（先查人，再查天气）
run_agent("王五在哪个部门工作？他所在城市（上海）今天天气怎么样？")

# 示例 4：无需工具，直接回答
run_agent("Python 和 JavaScript 各自适合做什么？")

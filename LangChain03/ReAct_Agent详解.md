# ReAct Agent 详解

## ReAct 是什么？

ReAct = **Re**asoning（推理）+ **Act**ing（行动）

是一种让 LLM 像人一样"边想边做"的 Agent 框架。

---

## 普通 LLM vs ReAct Agent

```
普通 LLM：
  问题 → 模型直接给答案（一步到位）
  缺点：如果需要查天气、算数学、搜数据库，模型本身做不到

ReAct Agent：
  问题 → 思考 → 用工具 → 看结果 → 继续思考 → 最终回答
  优点：可以调用外部工具，解决模型"不知道"的问题
```

---

## ReAct 的完整循环

```
Question（用户问题）
    ↓
Thought（我该怎么做？需要用什么工具？）
    ↓
Action（选择工具名称）
Action Input（传给工具的参数）
    ↓
Observation（工具执行后返回的结果）
    ↓
Thought（根据结果继续思考，够了吗？还需要再查吗？）
    ↓
...（可以循环多次）
    ↓
Final Answer（最终回答给用户）
```

---

## 生活类比

```
问题："今天出门要带伞吗？"

人的思考过程：
  Thought：我需要知道今天天气
  Action：打开天气 App（使用工具）
  Action Input：当前城市
  Observation：显示今天有雨
  Thought：有雨，需要带伞，我知道答案了
  Final Answer："需要带伞"

ReAct Agent 的过程完全一样，只是由 LLM 来做"思考"，
由代码来执行"工具"。
```

---

## 核心组件

### 1. `@tool` 装饰器 —— 定义工具

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """
    查询指定城市的今日天气信息。
    当用户询问某个城市的天气、温度、是否需要带伞时使用此工具。
    输入：城市名称，例如：北京、上海
    """
    # 真实场景中调用天气 API
    return "上海：小雨，需要带伞"
```

**docstring 是最重要的部分** —— 模型完全靠 docstring 来决定什么时候该用这个工具，写得越清楚，Agent 选对工具的概率越高。

### 2. ReAct 提示词模板 —— 固定格式

```
{tools}              ← 工具清单（自动填入）
{tool_names}         ← 工具名称列表（自动填入）
{input}              ← 用户的问题
{agent_scratchpad}   ← 模型推理过程的记录（自动维护）
```

提示词规定了 Thought / Action / Action Input / Observation / Final Answer 的固定格式，模型必须按这个格式输出。

### 3. `create_react_agent` —— 创建 Agent

```python
from langchain.agents import create_react_agent

agent = create_react_agent(
    llm=model,       # 使用的模型
    tools=tools,     # 工具列表
    prompt=prompt,   # ReAct 提示词模板
)
```

### 4. `AgentExecutor` —— 执行引擎

```python
from langchain.agents import AgentExecutor

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,              # 打印完整推理过程
    max_iterations=5,          # 最多循环 5 次，防止死循环
    handle_parsing_errors=True # 输出格式出错时自动处理
)

result = executor.invoke({"input": "上海今天需要带伞吗？"})
```

---

## 实际运行效果（verbose=True 时）

```
用户问题：上海今天需要带伞吗？

> Entering new AgentExecutor chain...

Thought: 用户想知道上海天气，我需要查询天气工具
Action: get_weather
Action Input: 上海
Observation: 小雨，气温 5°C 到 12°C，需要带伞

Thought: 我现在知道答案了
Final Answer: 上海今天有小雨，建议带伞出门。

> Finished chain.
```

---

## 多步推理示例

```
用户问题：王五在哪个部门？他所在城市今天天气怎样？

Thought: 先查王五的信息
Action: search_employee
Action Input: 王五
Observation: 王五，AI工程师，技术部，上海

Thought: 知道他在上海了，再查上海天气
Action: get_weather
Action Input: 上海
Observation: 小雨，需要带伞

Thought: 两个问题都有答案了
Final Answer: 王五在技术部，上海今天小雨建议带伞。
```

这就是 ReAct 的核心价值：**自主规划多个步骤，依次调用工具，最终整合成完整回答**。

---

## 关键参数说明

| 参数 | 说明 |
|------|------|
| `temperature=0` | Agent 场景推荐设为 0，让模型更稳定确定 |
| `verbose=True` | 打印完整推理过程，调试时必开 |
| `max_iterations` | 最大循环次数，防止死循环 |
| `handle_parsing_errors` | 模型输出格式不对时自动修复，不直接报错 |

---

## 一句话总结

> ReAct = 让模型"先想清楚再行动"，通过 Thought → Action → Observation 循环，
> 自主决策调用哪些工具、调用几次，最终整合所有信息给出回答。
> 模型负责"思考"，代码负责"执行工具"，两者分工协作。

# AutoGPT Agent 原理详解

## 一、AutoGPT 是什么？

### 1.1 普通 Agent vs AutoGPT

```
普通 Agent（demo07 的 ReAct）：

  用户："上海今天要带伞吗？"
  Agent：调用 get_weather → 回答 → 结束

  特点：一问一答，每次都需要人来推进


AutoGPT：

  用户："帮我研究 Python，写一份选型报告"
  AutoGPT：
    自己想→"我要拆成几个子任务"
    自己做→依次执行每个任务（调工具、存记忆）
    自己总结→生成最终报告

  特点：给一个「目标」，AI 全程自主，不需要人逐步干预
```

### 1.2 一个生活类比

```
普通 AI 助手 = 餐厅服务员
  你说"我要一杯水"，服务员去拿水回来
  你说"我要点菜"，服务员递菜单
  你说一步，他做一步

AutoGPT = 全能助理（秘书）
  你说"帮我安排明天的出差"
  助理自己拆：订机票 → 订酒店 → 打印行程 → 备份文件
  全部搞定后给你一份「出差安排清单」
  过程中不需要你再开口
```

---

## 二、本 Demo 的整体架构

```
用户输入「目标」
       ↓
┌──────────────────────────────────┐
│          规划器（Planner）        │
│  LLM 把目标拆成 4-5 个子任务     │
│  输出：["任务1", "任务2", ...]   │
└──────────────────┬───────────────┘
                   ↓
┌──────────────────────────────────────────────────────┐
│                   执行循环（for task in tasks）        │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │          ReAct Agent（执行器）                  │  │
│  │                                                │  │
│  │  Thought: 我需要搜索 Python 的适用场景...      │  │
│  │  Action: search_knowledge("Python适合")        │  │
│  │  Observation: Python适合AI/数据分析/后端...    │  │
│  │  Thought: 找到了，要把这个保存起来...          │  │
│  │  Action: save_to_memory("Python适合AI...")     │  │
│  │  Final Answer: 已完成场景研究                  │  │
│  └───────────────────┬────────────────────────────┘  │
│                      │                               │
│         ┌────────────▼──────────────┐               │
│         │       记忆系统            │               │
│         │  短期记忆：任务执行记录   │               │
│         │  长期记忆：重要发现笔记   │               │
│         └───────────────────────────┘               │
└──────────────────────────────────────────────────────┘
                   ↓
┌──────────────────────────────────┐
│          汇总器（Summarizer）     │
│  读取长期记忆 + 所有任务结果     │
│  LLM 整合 → 最终报告             │
└──────────────────────────────────┘
```

---

## 三、四大核心知识点

本 Demo 综合运用了前面学过的所有知识点，下面逐一讲解。

---

## 知识点一：任务规划（Planning）

### 这是什么？

规划就是「把大目标拆成小任务」。人类解决复杂问题时，首先会思考「有哪些步骤」，AutoGPT 也一样。

### 实现原理

用一个普通的 LLM Chain 来完成规划，输出 JSON 格式的任务列表：

```python
# 规划器 = 一个普通的 Chain
planner_chain = planner_prompt | model | StrOutputParser()

# 调用方式
raw_output = planner_chain.invoke({"goal": "研究 Python 的适用场景..."})

# 输出大概是这样（JSON 数组）：
# ["搜索Python适合哪些场景并保存", "搜索Python的优点并保存", "搜索Python的缺点", "写报告"]
```

### 提示词的关键技巧

```python
planner_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个任务规划专家。把用户目标拆解成 4-5 个具体的子任务。\n"
        "每个子任务要说明：用什么工具、查什么内容、输出什么结果\n\n"
        "输出格式：只输出 JSON 数组，不要有其他文字"  # ← 关键：要求 JSON 格式，方便解析
    ),
    ("human", "请把以下目标拆解成子任务：\n\n{goal}"),
])
```

告诉 LLM「只输出 JSON 数组」，这样我们用 `json.loads()` 解析就很简单。

### 解析鲁棒性处理

LLM 有时会在 JSON 前后加说明文字，所以需要先用正则提取 JSON：

```python
# 从 LLM 输出中提取 JSON 数组（防止 LLM 输出多余内容）
match = re.search(r'\[.*?\]', raw_output, re.DOTALL)
if match:
    tasks = json.loads(match.group())
```

**re.DOTALL 的作用**：让 `.` 能匹配换行符，因为 JSON 数组可能跨多行。

---

## 知识点二：工具调用（Tools）

### 这是什么？

和 demo07 完全一样的技术。工具就是 Agent 可以调用的「函数」，用 `@tool` 装饰器定义。

### 本 Demo 的 5 个工具

| 工具名 | 作用 | 对应人类行为 |
|--------|------|-------------|
| `search_knowledge` | 搜索技术知识 | 用搜索引擎查资料 |
| `save_to_memory` | 保存重要发现 | 在笔记本上记下关键点 |
| `retrieve_from_memory` | 检索历史记忆 | 翻看之前的笔记 |
| `calculate` | 数学计算 | 用计算器算数 |
| `write_report` | 写入最终报告 | 把成果写成文档 |

### 工具定义示例

```python
@tool
def search_knowledge(topic: str) -> str:
    """
    搜索指定主题的技术知识和行业信息。
    当需要了解某个技术、语言、框架的特点、适用场景时使用此工具。
    输入：要搜索的主题，例如：'Python适合做什么'、'Python的优点'
    """
    # 函数体：真实项目中调用搜索 API / RAG 系统
    # 本 Demo：用字典模拟知识库
    knowledge_base = {
        "Python适合": "Python最适合AI、数据分析、Web后端...",
        "Python优点": "语法简洁、生态丰富...",
    }
    for key, value in knowledge_base.items():
        if key in topic:
            return value
    return "未找到相关信息"
```

**docstring 是核心**：LLM 靠 docstring 决定「什么时候调用这个工具、传什么参数」。docstring 写得越清楚，Agent 调用越准确。

---

## 知识点三：短期记忆（Short-term Memory）

### 这是什么？

短期记忆存储的是**本次 AutoGPT 运行中的任务执行记录**。

作用：执行任务 N 时，Agent 能看到任务 1~N-1 做了什么，不会重复劳动。

### 和 demo05 的关系

**完全相同的技术**，只是使用场景不同：

```
demo05 用途：记住用户的多轮聊天内容
本 Demo 用途：记住各子任务的执行结果
```

### 代码实现

```python
# 创建短期记忆（和 demo05 完全一样）
short_term_memory = InMemoryChatMessageHistory()

# 执行完任务后，把结果写入短期记忆
short_term_memory.add_ai_message(
    f"任务{task_num}「{task[:30]}」完成，结论：{task_result[:100]}"
)
```

### 如何注入到 Agent

```python
# 执行下一个任务前，先读取短期记忆，作为背景信息
history_msgs = short_term_memory.messages
context = "之前已完成的任务记录：\n" + "\n".join(
    [f"  • {msg.content}" for msg in history_msgs[-6:]]
)

# 把 context 注入到 ReAct 提示词
executor.invoke({
    "input": task,
    "context": context,   # ← 这里注入短期记忆的内容
})
```

**执行流程图**：

```
任务1执行 → 结果写入短期记忆
任务2执行 → 读取短期记忆（知道任务1做了什么）→ 结果写入短期记忆
任务3执行 → 读取短期记忆（知道任务1、2做了什么）→ ...
```

---

## 知识点四：长期记忆（Long-term Memory / 向量存储）

### 这是什么？

长期记忆用于保存 **Agent 主动认为重要的发现**，可以跨任务检索。

### 短期 vs 长期记忆的区别

```
短期记忆（每轮自动记录）：
  "任务1完成：查到了Python适合AI和数据分析"
  "任务2完成：查到了Python的3个优点"
  自动记录，不可选择，存储格式固定

长期记忆（Agent 主动存储）：
  "Python最大优势是AI生态无可替代，适合AI团队"
  "Python不适合游戏开发和移动端"
  由 Agent 判断重要性后主动调用 save_to_memory 保存
  存储更精炼，检索更灵活
```

### 真实的向量数据库原理

**为什么要用向量？**

```
普通关键字搜索的局限：
  存了："Python最适合做AI开发"
  查询："Python的强项是什么"  ← 没有"AI"这个词
  结果：查不到！（关键字不匹配）

向量搜索的优势：
  存了："Python最适合做AI开发"  → 转成向量 [0.2, 0.8, 0.1, ...]
  查询："Python的强项是什么"    → 转成向量 [0.3, 0.7, 0.2, ...]
  结果：向量距离很近 → 搜索到了！（语义相似）
```

**向量数据库（FAISS）的完整代码**：

```python
# 第一步：安装依赖
# pip install faiss-cpu langchain-community

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings        # 把文字转成向量的模型

# 第二步：初始化向量数据库
embeddings = OpenAIEmbeddings()                      # 文字 → 向量 的转换器
vectorstore = FAISS.from_texts(["初始占位文本"], embeddings)

# 第三步：保存记忆（把文字转成向量存入数据库）
vectorstore.add_texts(["Python最适合做AI和数据分析"])
vectorstore.add_texts(["Python的缺点是速度慢、不适合移动端"])

# 第四步：检索记忆（语义相似度搜索）
results = vectorstore.similarity_search("Python的优势是什么", k=3)
for doc in results:
    print(doc.page_content)   # 找到与查询最相关的 3 条记忆
```

**向量搜索的工作原理**：

```
文字 "Python最适合AI" → Embedding模型 → [0.12, 0.87, 0.34, ...] (1536维向量)
文字 "Python的强项"   → Embedding模型 → [0.15, 0.82, 0.31, ...] (1536维向量)

计算两个向量的"距离"（余弦相似度）：
  距离很近（0.95）→ 语义相似 → 搜索命中！
  距离很远（0.12）→ 语义不同 → 不返回
```

### 本 Demo 的简化实现

```python
# 简化版：用 Python 列表代替向量数据库
long_term_memory: List[str] = []

# save_to_memory 工具（Agent 调用它来保存发现）
@tool
def save_to_memory(note: str) -> str:
    long_term_memory.append(note)          # 真实版：vectorstore.add_texts([note])
    return f"已保存（共{len(long_term_memory)}条）"

# retrieve_from_memory 工具（Agent 调用它来检索）
@tool
def retrieve_from_memory(query: str) -> str:
    all_notes = "\n".join(long_term_memory)
    return all_notes                       # 真实版：vectorstore.similarity_search(query)
```

---

## 四、ReAct Agent 执行器详解

### 执行器的作用

执行器 = 执行某个子任务的「工作单元」，内部是 demo07 学过的 ReAct 机制。

```
接收任务："搜索 Python 适合哪些场景，并保存到记忆"
   ↓
Thought: 我需要先搜索，然后保存
Action: search_knowledge("Python适合")
Observation: Python最适合AI、数据分析、Web后端...
Thought: 找到了，现在保存到记忆
Action: save_to_memory("Python最适合AI开发和数据分析")
Observation: 已保存（共1条）
Thought: 任务完成了
Final Answer: 已查询并保存 Python 的适用场景
```

### 提示词中的 {context} 变量（关键！）

本 Demo 在标准 ReAct 提示词上**加了一个 {context} 变量**，用来注入短期记忆：

```python
REACT_PROMPT = PromptTemplate.from_template("""
你可以使用以下工具：
{tools}

【执行背景 - 之前已完成的任务】
{context}       ← 这里注入短期记忆的内容

...（其余 ReAct 格式）

Question: {input}
Thought:{agent_scratchpad}
""")
```

传入方式：

```python
executor.invoke({
    "input": task,       # ← ReAct 必须有的变量
    "context": context,  # ← 我们新增的背景信息变量
})
```

### 每个任务都创建新的 Agent，为什么？

```python
def execute_task(task, task_num, total_tasks):
    agent = create_react_agent(llm=model, tools=tools, prompt=REACT_PROMPT)
    executor = AgentExecutor(agent=agent, ...)
    result = executor.invoke({"input": task, "context": context})
```

每次都新建 agent/executor，这样**每个任务的 agent_scratchpad 是干净的**，不会被上一个任务的推理过程干扰。上一个任务的结论通过 `context`（短期记忆）传入，这是更干净的设计。

---

## 五、汇总器（Summarizer）

### 为什么不用 Agent？

汇总阶段不需要调用工具，只需要「把已有信息整理成报告」，用普通 Chain 就够了：

```python
summarizer_chain = summarizer_prompt | model | StrOutputParser()

report = summarizer_chain.invoke({})
```

### 汇总的信息来源

```python
# 来源1：长期记忆（Agent 在执行过程中主动保存的重要发现）
memory_section = "【关键发现】\n" + "\n".join(long_term_memory)

# 来源2：每个任务的 Final Answer（任务执行结果）
results_section = "【各任务结果】\n" + "\n".join(task_results)

# 把两者一起发给 LLM，生成报告
summarizer_chain.invoke({
    "memory": memory_section,
    "results": results_section
})
```

---

## 六、完整执行流程演示

运行 `auto_gpt_agent.py` 后，你会看到以下输出：

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🤖  AutoGPT Agent 启动！
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

用户目标：请帮我研究 Python 编程语言...

============================================================
【Step 1 - 规划器】正在把目标拆解成子任务...
============================================================

已规划 4 个子任务：
  1. 使用 search_knowledge 查询 Python 适合哪些场景，用 save_to_memory 保存
  2. 使用 search_knowledge 查询 Python 的优点，用 save_to_memory 保存
  3. 使用 search_knowledge 查询 Python 的缺点，用 save_to_memory 保存
  4. 使用 retrieve_from_memory 整合所有发现，用 write_report 输出报告


────────────────────────────────────────────────────────────
【Step 2 - 执行器】任务 1/4
任务内容：使用 search_knowledge 查询 Python 适合哪些场景...
────────────────────────────────────────────────────────────

> Entering new AgentExecutor chain...

Thought: 我需要搜索 Python 的适用场景
Action: search_knowledge
Action Input: Python适合
Observation: Python最适合以下场景：
1. 数据分析与科学计算...
2. 机器学习/AI开发...

Thought: 找到了，现在把这个保存到记忆
Action: save_to_memory
Action Input: Python最适合AI、数据分析、Web后端、自动化脚本

    [💾 长期记忆 +1] Python最适合AI、数据分析、Web后端...

Observation: 已保存到长期记忆（当前共1条记录）

Thought: 任务完成了
Final Answer: 已查询并保存Python的适用场景

> Finished chain.

  ✅ 任务 1 完成

...（任务2、3类似）...

============================================================
【Step 3 - 汇总器】整合所有结果，生成最终报告...
============================================================

  📄 最终报告已生成：AutoGpt06/final_report.md


─── 记忆使用统计 ───
  短期记忆：4 条（任务执行记录）
  长期记忆：4 条（Agent 主动保存的重要发现）

  长期记忆内容（Agent 认为重要的发现）：
    [1] Python最适合AI、数据分析、Web后端、自动化脚本
    [2] Python主要优点：语法简洁、生态丰富、AI首选语言
    [3] Python主要缺点：速度慢、GIL限制多线程、不适合移动端
    [4] Python适合AI团队、初创公司、数据团队
```

---

## 七、各组件技术对照表

| 组件 | 技术 | 已学的 Demo | 在 AutoGPT 中的作用 |
|------|------|-------------|---------------------|
| 规划器 | `ChatPromptTemplate` + `StrOutputParser` | demo02、demo03 | 把目标拆成子任务列表 |
| 执行器 | `create_react_agent` + `AgentExecutor` | demo07 | 自主执行单个子任务 |
| 短期记忆 | `InMemoryChatMessageHistory` | demo05 | 记录任务执行历史 |
| 长期记忆 | 列表（生产用 FAISS） | memory05 原理 | 存储重要发现，跨任务检索 |
| 工具 | `@tool` 装饰器 | demo07 | 搜索、保存、计算、写文件 |
| 汇总器 | 普通 `prompt \| model \| parser` | demo04 | 整合所有结果写报告 |

---

## 八、需要掌握的知识点清单

### 必须掌握

- [ ] `@tool` 装饰器：如何定义工具，docstring 为什么重要
- [ ] `create_react_agent` + `AgentExecutor`：ReAct 的 Thought→Action→Observation 循环
- [ ] `InMemoryChatMessageHistory`：短期记忆的读写方式
- [ ] `ChatPromptTemplate`：多消息格式的提示词
- [ ] `StrOutputParser` + JSON 解析：处理 LLM 的结构化输出

### 理解即可（知道概念，不必手写）

- [ ] 向量数据库（FAISS/Chroma）的工作原理：文字→向量→相似度搜索
- [ ] Embedding（嵌入）：把文字转成数字向量的过程
- [ ] 任务规划（Planning）：用 LLM 把大目标拆成小任务的思路

### 进阶（了解有这个方向）

- [ ] 摘要记忆（Summary Memory）：用 LLM 压缩旧历史，节省 token
- [ ] 目标评估（Goal Evaluation）：让 Agent 自己判断是否完成目标
- [ ] 人工介入（Human in the loop）：关键步骤让人确认后再继续

---

## 九、生产环境的改进点

本 Demo 是为了教学而简化的，真实项目中需要改进：

| 简化点 | 生产环境改进 |
|--------|-------------|
| 知识库用字典 | 接入真实搜索 API（Bing/Google）或 RAG 系统 |
| 长期记忆用列表 | 换成 FAISS / Chroma / Pinecone 向量数据库 |
| 短期记忆只用内存 | 存入 Redis 或数据库，实现持久化 |
| 目标完成靠任务列表 | 加「目标评估器」，Agent 自己判断目标是否完成 |
| 单线程执行 | 可以并发执行多个独立任务，加快速度 |
| 无错误恢复 | 某个任务失败时，Agent 能自动重试或换策略 |

---

## 十、一句话总结

```
AutoGPT = 规划器 + (ReAct执行器 × N) + 短期记忆 + 长期记忆 + 汇总器

用户只需给「目标」，
Agent 自己想怎么做（规划），
自己去做（执行），
自己记住做了什么（记忆），
自己写总结（汇总）。
```

import os
import sys
import json
import re
from typing import List

sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# auto_gpt_agent.py  —  模仿 AutoGPT 的自主 Agent Demo
#
# 【AutoGPT 是什么？】
# AutoGPT 是 2023 年爆红的开源 AI Agent 项目。
# 它的革命性思想：
#   普通 AI 助手 = 一问一答，每次都需要人来推进
#   AutoGPT     = 给一个"目标"，AI 自主规划、执行、评估，直到完成
#
# 打个比方：
#   普通助手：老板每次告诉员工"去做A"，做完了再说"去做B"
#   AutoGPT ：老板只说"帮我完成这个项目"，员工自己拆分任务、执行、汇报
#
# 【本 Demo 演示的四大核心技术】
#
#   ① 任务规划（Planning）
#      LLM 把高层目标自动拆解成 3-5 个可执行的子任务
#
#   ② 工具调用（Tools）
#      Agent 自主决定调用哪些工具、传什么参数
#      （和 demo07_react_agent.py 相同的 @tool + ReAct 机制）
#
#   ③ 短期记忆（Short-term Memory）
#      存储本次运行中所有任务的执行记录
#      让 Agent 知道"之前已经做了什么"，避免重复执行
#      实现：InMemoryChatMessageHistory（和 demo05_memory.py 相同）
#
#   ④ 长期记忆（Long-term Memory / 向量存储）
#      把执行过程中的重要发现持久保存
#      需要时用"语义检索"找回相关记忆
#      教学中用 Python 列表实现；生产环境用 FAISS 向量数据库
#
# 【整体流程图】
#
#   用户目标："研究 Python 的适用场景并写选型建议"
#       ↓
#   ┌─────────────────┐
#   │  【规划器】      │  LLM 把目标拆成子任务列表
#   │  Planner        │  → [任务1, 任务2, 任务3, ...]
#   └────────┬────────┘
#            ↓
#   ┌─────────────────────────────────────────┐
#   │  【执行循环】 for task in tasks:         │
#   │                                         │
#   │  ┌─────────────────────────────────┐   │
#   │  │ ReAct Agent（demo07 的技术）     │   │
#   │  │   Thought: 我需要搜索 Python... │   │
#   │  │   Action: search_knowledge      │   │
#   │  │   Observation: Python 适合...  │   │
#   │  │   Thought: 要保存这个发现...    │   │
#   │  │   Action: save_to_memory        │   │
#   │  │   Final Answer: 任务完成        │   │
#   │  └──────────┬──────────────────────┘   │
#   │             │                           │
#   │  ┌──────────▼──────────────────────┐   │
#   │  │ 短期记忆：记录任务执行结果       │   │
#   │  │ 长期记忆：保存重要发现           │   │
#   │  └─────────────────────────────────┘   │
#   └─────────────────────────────────────────┘
#            ↓
#   ┌─────────────────┐
#   │  【汇总器】      │  读取所有任务结果 + 长期记忆
#   │  Summarizer     │  → 生成最终报告
#   └─────────────────┘
# ============================================================

from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.agents import create_react_agent, AgentExecutor


# ============================================================
# 初始化 LLM
# temperature=0：让规划和推理更稳定（减少随机性）
# ============================================================
model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    temperature=0,
)


# ============================================================
# 短期记忆（Short-term Memory）
#
# 定义：本次 AutoGPT 运行期间，记录每个任务的执行结果。
#
# 作用：
#   Agent 在执行任务 N 时，能看到任务 1~N-1 已经得到了什么结论，
#   不会重复研究同一个问题，执行效率更高。
#
# 类比：
#   就像写笔记的草稿纸，在做一个大项目时，
#   你会随时记下"第一步已完成：找到了 Python 的适用场景"
#
# 实现：和 demo05_memory.py 一样，用 InMemoryChatMessageHistory
# 生产环境：可换成 Redis + 数据库持久化
# ============================================================
short_term_memory = InMemoryChatMessageHistory()


# ============================================================
# 长期记忆（Long-term Memory）
#
# 定义：把执行过程中发现的重要信息永久保存，随时可以检索。
#
# 真实的向量数据库存储方式（FAISS）：
#
#   【保存】把文字转成向量（Embedding），存入数据库
#
#     from langchain_community.vectorstores import FAISS
#     from langchain_openai import OpenAIEmbeddings
#
#     embeddings = OpenAIEmbeddings()     # 文字 → 向量 的转换器
#     vectorstore = FAISS.from_texts(["初始文本"], embeddings)
#
#     # 保存新发现：
#     vectorstore.add_texts(["Python 最适合做 AI 和数据分析"])
#
#   【检索】把查询也转成向量，找相似度最高的记忆
#
#     results = vectorstore.similarity_search("Python 的优势", k=3)
#     for doc in results:
#         print(doc.page_content)   # 语义最相关的 3 条记忆
#
#   向量的魔力：
#     "Python 的优势" 和 "Python 的优点" 的向量很接近
#     即使措辞不同，也能找到相关内容（关键字搜索做不到这点）
#
# 本 Demo 简化实现：
#   用 Python 列表存储，检索时返回全部记忆给 LLM 判断相关性。
#   这在记忆数量少时效果基本一致，且不需要额外安装 FAISS。
#
# 向量数据库选型参考：
#   FAISS     → 本地，Facebook 开源，适合中小型项目
#   Chroma    → 本地，轻量级，开发调试首选
#   Pinecone  → 云端，生产级别，高并发场景
#   Milvus    → 自托管，企业级，大规模向量检索
# ============================================================
long_term_memory: List[str] = []     # 模拟向量数据库（生产环境替换成 FAISS 等）


# ============================================================
# Part 1：定义工具（Tools）
#
# AutoGPT 的工具就像人的"双手"，能执行各种外部操作。
# 工具越多，Agent 能解决的问题越复杂。
#
# 真实 AutoGPT 的工具：
#   - 浏览器：搜索网页、读取文章
#   - 代码执行器：运行 Python 代码
#   - 文件系统：读写文件
#   - 邮件：发送邮件
#   - 数据库：查询 SQL
#   - 长期记忆：保存和检索信息
#
# 本 Demo 的工具：
#   - search_knowledge：搜索知识库（模拟网络搜索）
#   - save_to_memory  ：保存发现到长期记忆
#   - retrieve_from_memory：从长期记忆检索
#   - calculate       ：数学计算
#   - write_report    ：写文件（最终报告）
#
# 工具的 docstring 非常关键！
# LLM 完全靠 docstring 来决定：什么时候调用哪个工具、传什么参数。
# docstring 写得越清楚，Agent 调用工具就越准确。
# ============================================================

@tool
def search_knowledge(topic: str) -> str:
    """
    搜索指定主题的技术知识和行业信息。
    当需要了解某个技术、语言、框架的特点、适用场景、优缺点时使用此工具。
    输入：要搜索的主题，例如：'Python适合做什么'、'Python的优点'、'Python的缺点'
    """
    # 模拟知识库（真实环境中调用搜索 API、RAG 系统或网络爬虫）
    knowledge_base = {
        "Python适合": (
            "Python 最适合以下场景：\n"
            "1. 数据分析与科学计算（pandas, numpy, matplotlib）\n"
            "2. 机器学习 / AI 开发（PyTorch, TensorFlow, scikit-learn）\n"
            "3. Web 后端开发（Django, FastAPI, Flask）\n"
            "4. 自动化脚本（文件处理、定时任务、爬虫）\n"
            "5. 快速原型开发（语法简洁，开发效率极高）\n"
            "6. DevOps 工具链（Ansible, 运维脚本）"
        ),
        "Python优点": (
            "Python 的主要优点：\n"
            "1. 语法极简，接近自然语言，新人上手快\n"
            "2. 生态丰富，PyPI 有超过 50 万个开源包\n"
            "3. AI/数据科学的第一语言，社区生态无可替代\n"
            "4. 跨平台，一份代码运行在 Windows/Mac/Linux\n"
            "5. 胶水语言，能方便调用 C/C++ 扩展（如 numpy 底层）"
        ),
        "Python缺点": (
            "Python 的主要缺点：\n"
            "1. 运行速度比 C/Java 慢（解释型语言，无预编译优化）\n"
            "2. GIL（全局解释器锁）限制真正的多线程并发性能\n"
            "3. 不适合原生移动端 App 开发（iOS/Android）\n"
            "4. 内存消耗比编译型语言高\n"
            "5. 不适合游戏引擎或操作系统级别的高性能计算"
        ),
        "Python适合团队": (
            "Python 特别适合以下团队：\n"
            "1. AI / 机器学习团队：生态无可替代，PyTorch/TF 均以 Python 为主\n"
            "2. 数据工程 / 数据分析团队：pandas+numpy 是行业标准\n"
            "3. 初创公司：开发效率高，能快速验证想法、快速上线\n"
            "4. 后端 API 服务团队：FastAPI 性能已媲美 Node.js/Go\n"
            "5. 学术研究团队：Jupyter Notebook 是科研标配"
        ),
        "Python不适合": (
            "Python 不适合以下场景：\n"
            "1. 高性能游戏开发（用 C++/C#/Rust 更合适）\n"
            "2. 原生移动 App（用 Swift/Kotlin/Flutter 更合适）\n"
            "3. 前端开发（JavaScript/TypeScript 才是王道）\n"
            "4. 对实时性要求极高的系统（如高频量化交易）\n"
            "5. 嵌入式系统开发（用 C/C++ 更合适）"
        ),
    }

    # 找最匹配的知识点
    for key, value in knowledge_base.items():
        if any(word in topic for word in key.split()):
            return value

    return (
        f"关于'{topic}'暂无精确结果，"
        "建议尝试更具体的关键词，例如：'Python适合'、'Python优点'、'Python缺点'"
    )


@tool
def save_to_memory(note: str) -> str:
    """
    把重要发现、关键结论保存到长期记忆，供后续任务使用。
    当搜索到重要信息，或者完成了某项研究发现关键结论时，调用此工具保存。
    输入：要保存的笔记，简洁清晰，例如：'Python最适合AI和数据分析，生态最强'
    """
    long_term_memory.append(note)
    print(f"\n    [💾 长期记忆 +1] {note[:60]}...")
    return f"已保存到长期记忆（当前共 {len(long_term_memory)} 条记录）"


@tool
def retrieve_from_memory(query: str) -> str:
    """
    从长期记忆中检索与查询相关的历史记录和重要笔记。
    当需要查看之前保存过的研究结论时，调用此工具。
    输入：检索关键词，例如：'Python的研究结论'、'所有已保存的发现'
    """
    if not long_term_memory:
        return "长期记忆目前为空，还没有保存任何内容。"

    # 本 Demo：返回全部记忆
    # 真实版本（FAISS）：vectorstore.similarity_search(query, k=5) 返回最相关的 5 条
    all_notes = "\n".join(
        [f"  [{i+1}] {note}" for i, note in enumerate(long_term_memory)]
    )
    return (
        f"长期记忆中共有 {len(long_term_memory)} 条记录：\n"
        f"{all_notes}\n\n"
        "（真实环境：此处会用向量相似度搜索，只返回与查询最相关的几条）"
    )


@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式，返回计算结果。
    当需要做数学计算、统计分析时使用此工具。
    输入：合法的数学表达式，例如：'100 * 0.7 + 50' 或 '(30 + 70) / 2'
    """
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "表达式包含非法字符，只支持基本数学运算"
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算失败：{e}"


@tool
def write_report(content: str) -> str:
    """
    将内容写入 Markdown 报告文件，完成最终输出。
    只在所有研究完成，需要输出最终结论报告时调用此工具。
    输入：报告的完整正文内容（Markdown 格式）
    """
    os.makedirs("AutoGpt06", exist_ok=True)
    path = "AutoGpt06/final_report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# AutoGPT 技术研究报告\n\n")
        f.write(content)
    print(f"\n    [📄 报告] 已写入 {path}")
    return f"报告已保存到 {path}"


# 把所有工具放到列表
tools = [search_knowledge, save_to_memory, retrieve_from_memory, calculate, write_report]


# ============================================================
# Part 2：任务规划器（Planner）
#
# 规划器的角色：项目经理
#   接收高层目标 → 拆解成 3-5 个具体的子任务 → 返回任务列表
#
# 为什么要规划？
#   如果直接把复杂目标丢给 Agent，它可能不知道从哪里下手，
#   或者会漫无目的地探索，效率极低。
#   规划先把"大目标"变成"小任务"，每个任务边界清晰、可验证完成。
#
# 实现：用一个专门的 LLM Chain 来生成任务列表，
#       输出格式要求是 JSON 数组，方便后续解析。
# ============================================================

def plan_tasks(goal: str) -> List[str]:
    """
    用 LLM 把高层目标拆解成可执行的子任务列表
    返回：任务描述字符串列表，例如 ["搜索Python适合场景", "整理优缺点", ...]
    """
    print("\n" + "=" * 60)
    print("【Step 1 - 规划器】正在把目标拆解成子任务...")
    print("=" * 60)

    planner_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是一个任务规划专家。你的工作是把用户的目标拆解成 4-5 个具体的子任务。\n"
            "每个子任务的要求：\n"
            "1. 具体可执行（要清楚说明要用什么工具、查什么内容）\n"
            "2. 有明确输出（知道做完是什么样）\n"
            "3. 与其他任务顺序合理\n\n"
            "可用的工具有：search_knowledge（查知识）、save_to_memory（存记忆）、"
            "retrieve_from_memory（取记忆）、calculate（计算）、write_report（写报告）\n\n"
            "输出格式：只输出 JSON 数组，不要有任何其他文字，例如：\n"
            '["使用 search_knowledge 查询 Python 适合的场景，并用 save_to_memory 保存结论", '
            '"使用 search_knowledge 查询 Python 的优点", '
            '"使用 retrieve_from_memory 整合所有记忆，用 write_report 输出报告"]'
        ),
        (
            "human",
            "请把以下目标拆解成子任务：\n\n{goal}"
        ),
    ])

    # StrOutputParser 先拿到文本，再手动解析 JSON
    # 比 JsonOutputParser 更健壮（不怕 LLM 输出前后有多余文字）
    planner_chain = planner_prompt | model | StrOutputParser()
    raw_output = planner_chain.invoke({"goal": goal})

    # 从输出中提取 JSON 数组（LLM 有时会在 JSON 前后加说明文字）
    match = re.search(r'\[.*?\]', raw_output, re.DOTALL)
    if match:
        try:
            tasks = json.loads(match.group())
        except json.JSONDecodeError:
            # JSON 解析失败时，按行分割作为备选
            tasks = [line.strip().strip('"').strip(',') for line in raw_output.split('\n') if line.strip()]
    else:
        tasks = [raw_output]  # 最坏情况：把整个输出当做一个任务

    print(f"目标：{goal}")
    print(f"\n已规划 {len(tasks)} 个子任务：")
    for i, task in enumerate(tasks, 1):
        print(f"  {i}. {task}")

    return tasks


# ============================================================
# Part 3：任务执行器（Executor）
#
# 执行器的角色：负责执行某个子任务的专业员工
#   接收一个子任务 → 用 ReAct Agent 自主决策如何完成 → 返回结果
#
# 内部机制（和 demo07_react_agent.py 完全一样）：
#   ReAct = Reasoning（推理）+ Acting（行动）
#   循环：Thought → Action → Observation → Thought → ...
#         （想要做什么）→（调用工具）→（看结果）→（继续思考）
#   直到有了足够信息，输出 Final Answer
#
# 记忆的使用：
#   ① 执行前：从短期记忆读取"之前做了什么"，作为上下文背景注入 prompt
#   ② 执行中：Agent 可以调用 save_to_memory 保存发现（长期记忆）
#             Agent 可以调用 retrieve_from_memory 查看历史发现（长期记忆）
#   ③ 执行后：把任务结果写入短期记忆，供后续任务参考
# ============================================================

# ReAct 提示词模板（与 demo07 相同的结构，增加了 {context} 背景信息注入）
# 注意：{tools} {tool_names} {input} {agent_scratchpad} 是 ReAct Agent 必须有的占位符
REACT_PROMPT = PromptTemplate.from_template("""你是 AutoGPT 的执行引擎，负责完成指定的子任务。

你可以使用的工具：
{tools}

【执行背景 - 之前已完成的任务】
{context}

执行规则：
1. 必须严格按照 Thought/Action/Action Input/Observation 格式输出
2. 每次调用工具后，根据结果决定下一步行动
3. 搜索到重要信息后，必须用 save_to_memory 工具保存，避免遗忘
4. 需要查看之前研究的内容时，用 retrieve_from_memory 检索
5. 完成任务后输出 Final Answer

可用工具名称列表：{tool_names}

开始执行！

Question: {input}
Thought:{agent_scratchpad}""")


def execute_task(task: str, task_num: int, total_tasks: int) -> str:
    """
    用 ReAct Agent 执行单个子任务

    参数：
      task        - 任务描述
      task_num    - 当前第几个任务（从1开始）
      total_tasks - 总任务数

    返回：任务的执行结果（Final Answer 的内容）
    """
    print(f"\n{'─' * 60}")
    print(f"【Step 2 - 执行器】任务 {task_num}/{total_tasks}")
    print(f"任务内容：{task}")
    print('─' * 60)

    # ──────────────────────────────────────────────────────────
    # 构建「执行背景」：让 Agent 知道之前已经做了什么
    #
    # 这就是短期记忆的核心用途：
    # Agent 在执行任务 N 时，能看到任务 1~N-1 的结论，
    # 不会重复查询，执行更高效。
    # ──────────────────────────────────────────────────────────
    history_msgs = short_term_memory.messages
    if history_msgs:
        # 只取最近 6 条记录，避免 context 过长
        recent = history_msgs[-6:]
        context = "之前已完成的任务记录：\n" + "\n".join(
            [f"  • {msg.content}" for msg in recent]
        )
    else:
        context = "这是第一个任务，还没有执行历史记录。"

    # 创建 ReAct Agent（和 demo07 完全一样的写法）
    agent = create_react_agent(
        llm=model,
        tools=tools,
        prompt=REACT_PROMPT,
    )

    # AgentExecutor 负责运行 Thought→Action→Observation 循环
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,               # 打印完整推理过程（学习时强烈建议开启）
        max_iterations=8,           # 最多 8 步，防止死循环
        handle_parsing_errors=True, # 格式出错时自动修复，不直接报错
    )

    # 执行任务（传入 input + context 两个变量，prompt 模板中都有对应占位符）
    result = executor.invoke({
        "input": task,
        "context": context,
    })

    task_result = result.get("output", "任务执行完成（无输出）")

    # ──────────────────────────────────────────────────────────
    # 把任务结果写入短期记忆
    # 下一个任务执行时，会从这里读取"已经做了什么"
    # ──────────────────────────────────────────────────────────
    short_term_memory.add_ai_message(
        f"任务{task_num}「{task[:30]}...」完成，结论：{task_result[:100]}"
    )

    print(f"\n  ✅ 任务 {task_num} 完成")
    return task_result


# ============================================================
# Part 4：汇总器（Summarizer）
#
# 汇总器的角色：撰写项目总结报告的高级顾问
#   读取所有任务执行结果 + 长期记忆中保存的重要发现
#   → 整合成一份结构清晰的最终报告
#
# 这一步不用 ReAct Agent，因为不需要调用工具，
# 只需要 LLM 把已有信息"整理成报告"即可，
# 用普通的 Chain（prompt | model | StrOutputParser）就够了。
# ============================================================

def summarize_and_report(goal: str, task_results: List[str]) -> str:
    """
    整合所有研究结果，生成最终报告

    参数：
      goal         - 原始目标
      task_results - 每个子任务的执行结果列表
    """
    print(f"\n{'=' * 60}")
    print("【Step 3 - 汇总器】正在整合所有结果，生成最终报告...")
    print("=" * 60)

    # 整合长期记忆（Agent 执行过程中 save_to_memory 保存的所有发现）
    if long_term_memory:
        memory_section = (
            "【Agent 在执行过程中保存的关键发现（长期记忆）】\n" +
            "\n".join([f"  • {note}" for note in long_term_memory])
        )
    else:
        memory_section = "（Agent 执行过程中未保存任何记忆）"

    # 整合所有任务结果（短期记忆的内容）
    results_section = "【各子任务执行结果】\n" + "\n".join(
        [f"  任务{i+1}：{result[:300]}" for i, result in enumerate(task_results)]
    )

    # 用 LLM 把所有信息整合成报告
    summarizer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是一个专业的技术报告撰写专家。\n"
            "请根据 AI Agent 执行过程中收集到的信息，\n"
            "撰写一份结构清晰、内容实用的技术报告。\n"
            "要求：有概述、主要发现（分点列举）、结论和建议，使用 Markdown 格式。\n"
            "语言简洁，控制在 600 字以内。"
        ),
        (
            "human",
            f"研究目标：{goal}\n\n"
            f"{memory_section}\n\n"
            f"{results_section}\n\n"
            "请根据以上信息撰写完整的技术报告。"
        ),
    ])

    summarizer_chain = summarizer_prompt | model | StrOutputParser()
    report_content = summarizer_chain.invoke({})

    # 写入文件
    os.makedirs("AutoGpt06", exist_ok=True)
    with open("AutoGpt06/final_report.md", "w", encoding="utf-8") as f:
        f.write(f"# AutoGPT 研究报告\n\n")
        f.write(f"**研究目标**：{goal}\n\n")
        f.write("---\n\n")
        f.write(report_content)

    print("  📄 最终报告已生成：AutoGpt06/final_report.md")
    return report_content


# ============================================================
# Part 5：AutoGPT 主循环
#
# 把三个组件串联起来：
#   规划（Planner）→ 执行（Executor × N）→ 汇总（Summarizer）
#
# 这就是 AutoGPT 的完整工作流程！
# ============================================================

def run_autogpt(goal: str):
    """
    AutoGPT 主入口：给定目标，自主完成整个流程

    完整流程：
      Step 1 - 规划：把目标拆解成子任务列表
      Step 2 - 执行：逐一执行每个子任务（ReAct Agent + Tools + Memory）
      Step 3 - 汇总：整合所有结果，生成最终报告
    """
    print("\n")
    print("━" * 60)
    print("  🤖  AutoGPT Agent 启动！")
    print("━" * 60)
    print(f"\n用户目标：{goal}\n")
    print("技术组成：")
    print("  ① 任务规划（Planner）  → LLM 拆分子任务")
    print("  ② ReAct Agent          → 自主思考+工具调用（demo07技术）")
    print("  ③ 短期记忆              → InMemoryChatMessageHistory（demo05技术）")
    print("  ④ 长期记忆              → 列表模拟（生产用 FAISS 向量数据库）")
    print("  ⑤ 汇总报告              → LLM 整合生成")
    print()

    # ── Step 1：规划 ──────────────────────────────────────────
    tasks = plan_tasks(goal)

    # ── Step 2：逐一执行 ──────────────────────────────────────
    print(f"\n\n开始执行 {len(tasks)} 个子任务...")
    task_results = []
    for i, task in enumerate(tasks):
        result = execute_task(task, i + 1, len(tasks))
        task_results.append(result)

    # ── Step 3：汇总 ──────────────────────────────────────────
    final_report = summarize_and_report(goal, task_results)

    # 打印完成信息
    print("\n")
    print("━" * 60)
    print("  ✅  AutoGPT 完成！")
    print("━" * 60)

    print("\n─── 最终报告（节选）───")
    preview = final_report[:600]
    print(preview + ("..." if len(final_report) > 600 else ""))
    print("\n完整报告已保存：AutoGpt06/final_report.md")

    # 打印记忆统计，加深对记忆机制的理解
    print("\n─── 记忆使用统计 ───")
    print(f"  短期记忆：{len(short_term_memory.messages)} 条（任务执行记录）")
    print(f"  长期记忆：{len(long_term_memory)} 条（Agent 主动保存的重要发现）")
    if long_term_memory:
        print("\n  长期记忆内容（Agent 认为重要的发现）：")
        for i, note in enumerate(long_term_memory, 1):
            print(f"    [{i}] {note[:80]}")


# ============================================================
# 运行 Demo
#
# 目标：研究 Python 的适用场景
# AutoGPT 会：
#   1. 自动把目标拆成 4-5 个子任务
#   2. 逐个执行，调用工具搜索知识、保存发现
#   3. 整合所有发现，写出一份技术选型建议报告
# ============================================================

if __name__ == "__main__":
    # 用一个具体的研究目标来驱动 AutoGPT
    goal = (
        "请帮我研究 Python 编程语言：包括它最适合哪些应用场景、主要优点和缺点、"
        "适合哪类团队使用，最终整合成一份技术选型建议报告"
    )

    run_autogpt(goal)

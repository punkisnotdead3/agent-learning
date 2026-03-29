import os
import sys
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# 设置输出编码，防止 Windows 下乱码
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

# ============================================================
# LangGraph Multi-Agent 多智能体协作系统
#
# 什么是 Multi-Agent（多智能体）？
#
# 在复杂的 AI 应用中，单一 Agent 往往难以应对多样化的任务需求。
# Multi-Agent 架构将不同的职责分配给专门的 Agent，它们相互协作完成更复杂的任务。
#
# 典型场景：
#   - 写作团队：研究员 → 写手 → 编辑 → 审稿人
#   - 客服系统：意图识别 → 问题分类 → 专业解答 → 满意度回访
#   - 代码助手：需求分析 → 架构设计 → 代码编写 → 代码审查
#
# 本 Demo 实现的多智能体流程：
#
#                    START
#                      ↓
#              ┌───────────────┐
#              │  router_agent │ ← 路由智能体：分析用户意图，决定任务分配给哪个 Agent
#              └───────────────┘
#                      ↓
#           ┌─────────┼─────────┐
#           ↓         ↓         ↓
#      ┌────────┐ ┌────────┐ ┌────────┐
#      │ writer │ │coder   │ │ analyst│ ← 三个专业智能体，分别负责不同领域
#      │_agent  │ │_agent  │ │_agent  │
#      └────────┘ └────────┘ └────────┘
#           ↓         ↓         ↓
#           └─────────┴─────────┘
#                      ↓
#                     END
#
# 核心设计模式：
#
#   1. Router Pattern（路由模式）
#      - 由 router_agent 根据用户输入决定下一步执行哪个专业 Agent
#      - 使用条件边（conditional edges）实现动态路由
#
#   2. Specialized Agents（专业化 Agent）
#      - 每个 Agent 有自己的系统提示词和专业领域
#      - 专注于解决特定类型的问题
#
#   3. Shared State（共享状态）
#      - 所有 Agent 共享同一个 State，可以访问完整对话历史
#      - 便于协作和上下文理解
#
#   4. Supervisor Pattern（监督者模式，扩展方向）
#      - 每个专业 Agent 执行完后可以返回 router
#      - router 决定是否继续、切换 Agent 或结束
# ============================================================

# ============================================================
# Step 1: 定义共享状态（State）
#
# 在 Multi-Agent 系统中，State 是所有 Agent 共享的"全局内存"。
# 每个 Agent 可以读取之前的对话历史，保证上下文连贯性。
#
# 本 demo 的 State 包含：
#   - messages: 对话历史（所有 Agent 的交互记录）
#   - next_node: 路由决策结果（由 router_agent 设置）
#   - current_agent: 当前正在执行的 Agent 名称（用于调试和追踪）
# ============================================================


class State(TypedDict):
    # messages: 使用 add_messages reducer 确保消息是追加而非覆盖
    messages: Annotated[list[BaseMessage], add_messages]

    # next_node: 路由决策结果，用于条件边决定下一个节点
    # Literal 类型限制只能是这几个值之一，提高类型安全
    next_node: Literal["writer", "coder", "analyst", END]

    # current_agent: 记录当前是哪个 Agent 在执行，便于调试和日志追踪
    current_agent: str


# ============================================================
# Step 2: 初始化模型
#
# 在 Multi-Agent 系统中，可以使用：
#   - 同一个模型实例（本 demo 方式）：简单、统一
#   - 不同模型：不同 Agent 用不同模型（如 coder 用代码专用模型）
#   - 不同配置：同一个模型但不同参数（temperature 等）
#
# 这里我们使用同一个模型，但通过不同的系统提示词来实现专业化。
# ============================================================

model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    temperature=0.7,
)

# ============================================================
# Step 3: 定义各个 Agent 的系统提示词
#
# 系统提示词是 Agent 的"人设"，决定了它的行为方式和专业领域。
# 在 Multi-Agent 系统中，清晰的职责边界很重要。
# ============================================================

# 路由智能体：负责分析用户意图，决定调用哪个专业 Agent
ROUTER_SYSTEM_PROMPT = SystemMessage(
    content="""你是一个智能路由助手。你的任务是分析用户的需求，并将其分配给最合适的专业 Agent。

可选的专业 Agent：
1. writer - 写作助手：擅长创意写作、文章撰写、文案优化、故事创作、诗歌生成等
2. coder - 编程助手：擅长代码编写、算法设计、代码调试、技术解释、编程教学等
3. analyst - 分析助手：擅长数据分析、逻辑推理、数学计算、观点分析、商业分析等

重要规则：
- 只输出 Agent 名称，不要输出其他内容
- 如果用户需求不明确，选择 writer（写作是通用能力）
- 如果涉及多个领域，选择最主要的一个

输出格式（严格遵循）：writer / coder / analyst"""
)

# 写作助手：负责创意写作和文案相关任务
WRITER_SYSTEM_PROMPT = SystemMessage(
    content="""你是一位才华横溢的写作助手，擅长各种创作任务。

你的专长：
- 创意写作：故事、小说、诗歌、剧本
- 商业文案：广告文案、产品描述、营销邮件
- 学术写作：论文润色、摘要撰写、报告整理
- 内容优化：改写、扩写、缩写、润色

风格特点：
- 语言优美流畅，富有感染力
- 善于把握不同文体的特点
- 能够根据需求调整语气和风格

记住：你是写作专家，请用专业作家的水准回应用户。"""
)

# 编程助手：负责代码相关任务
CODER_SYSTEM_PROMPT = SystemMessage(
    content="""你是一位经验丰富的编程助手，精通多种编程语言和技术栈。

你的专长：
- 代码编写：Python、JavaScript、Java、C++ 等各种语言
- 算法设计：数据结构、算法优化、复杂度分析
- 代码调试：错误排查、Bug 修复、性能优化
- 技术教学：概念解释、最佳实践、代码审查

风格特点：
- 代码规范，注释清晰
- 解释通俗易懂，由浅入深
- 注重实用性和可维护性

记住：你是编程专家，请用资深工程师的水准回应用户。"""
)

# 分析助手：负责数据分析和逻辑推理
ANALYST_SYSTEM_PROMPT = SystemMessage(
    content="""你是一位严谨的分析助手，擅长逻辑推理和数据洞察。

你的专长：
- 数据分析：趋势分析、统计分析、数据可视化建议
- 逻辑推理：问题拆解、因果分析、方案评估
- 数学计算：公式推导、数值计算、数学建模
- 商业分析：SWOT 分析、竞品分析、市场洞察

风格特点：
- 逻辑严密，条理清晰
- 数据驱动，有据可依
- 善于发现隐藏的模式和洞察

记住：你是分析专家，请用数据分析师的水准回应用户。"""
)

# ============================================================
# Step 4: 定义节点函数（Nodes）
#
# 每个 Agent 都是一个节点函数，接收 State，返回更新。
# 在 Multi-Agent 系统中，节点函数需要：
#   1. 设置 current_agent 标识当前执行者
#   2. 根据专业领域处理消息
#   3. 返回更新后的 State
# ============================================================


def router_agent(state: State) -> dict:
    """
    路由智能体：分析用户意图，决定调用哪个专业 Agent。

    工作流程：
    1. 将用户最后一条消息发给模型
    2. 模型根据 ROUTER_SYSTEM_PROMPT 的指示输出 Agent 名称
    3. 解析模型输出，设置 next_node 用于条件边路由

    Args:
        state: 当前状态，包含 messages 等

    Returns:
        状态更新：next_node（路由决策）和 current_agent
    """
    # 获取用户的最后一条消息
    last_message = state["messages"][-1]

    # 构建给路由模型的消息
    # 注意：我们只发系统提示 + 用户最后一条消息，不发完整历史
    # 因为路由只需要知道"当前要做什么"，不需要完整对话上下文
    router_messages = [ROUTER_SYSTEM_PROMPT, last_message]

    # 调用模型获取路由决策
    response = model.invoke(router_messages)

    # 解析模型输出，提取 Agent 名称
    # 模型应该只返回 "writer"/"coder"/"analyst" 之一
    decision = response.content.strip().lower()

    # 安全性检查：确保输出有效
    valid_agents = ["writer", "coder", "analyst"]
    if decision not in valid_agents:
        # 如果模型输出不合法，默认使用 writer
        decision = "writer"

    print(f"  [路由决策] 用户意图分析结果 → 分配给 '{decision}' Agent")

    # 返回状态更新
    return {
        "next_node": decision,      # 用于条件边决定下一跳
        "current_agent": "router",  # 记录当前执行的是 router
    }


def writer_agent(state: State) -> dict:
    """
    写作助手：处理创意写作和文案相关任务。

    工作流程：
    1. 将系统提示 + 完整对话历史发给模型
    2. 模型以写作专家的身份回复
    3. 返回 AI 回复消息

    Args:
        state: 当前状态，包含完整 messages

    Returns:
        状态更新：新增的 AI 消息
    """
    # 构建完整的消息列表：系统提示 + 所有历史消息
    # 这里我们使用完整历史，让写作助手了解上下文
    all_messages = [WRITER_SYSTEM_PROMPT] + state["messages"]

    # 调用模型生成回复
    response = model.invoke(all_messages)

    print(f"  [写作助手] 正在创作回复...")

    return {
        "messages": [response],           # 新增 AI 回复
        "current_agent": "writer",        # 记录当前执行者
    }


def coder_agent(state: State) -> dict:
    """
    编程助手：处理代码编写、调试、解释等任务。

    工作流程与 writer_agent 类似，但使用 CODER_SYSTEM_PROMPT。
    """
    all_messages = [CODER_SYSTEM_PROMPT] + state["messages"]
    response = model.invoke(all_messages)

    print(f"  [编程助手] 正在编写代码...")

    return {
        "messages": [response],
        "current_agent": "coder",
    }


def analyst_agent(state: State) -> dict:
    """
    分析助手：处理数据分析、逻辑推理等任务。

    工作流程与 writer_agent 类似，但使用 ANALYST_SYSTEM_PROMPT。
    """
    all_messages = [ANALYST_SYSTEM_PROMPT] + state["messages"]
    response = model.invoke(all_messages)

    print(f"  [分析助手] 正在进行分析...")

    return {
        "messages": [response],
        "current_agent": "analyst",
    }


# ============================================================
# Step 5: 定义条件边逻辑
#
# 条件边（Conditional Edge）是 Multi-Agent 系统的核心机制。
# 它根据当前 State 动态决定下一个执行节点。
#
# 在本 demo 中：
#   - router_agent 设置 state["next_node"]
#   - route_to_agent 读取这个值，返回目标节点名称
#   - LangGraph 根据返回值跳转到对应节点
# ============================================================


def route_to_agent(state: State) -> Literal["writer", "coder", "analyst"]:
    """
    条件边函数：根据 router_agent 的决策返回目标节点。

    这是连接 router 和专业 Agent 的"桥梁"。
    返回值必须是图中存在的节点名称。

    Args:
        state: 当前状态，包含 next_node（路由决策）

    Returns:
        目标节点名称："writer"/"coder"/"analyst"
    """
    return state["next_node"]


# ============================================================
# Step 6: 构建图（Graph Construction）
#
# 这是 Multi-Agent 系统最复杂的部分，需要正确设置节点和边。
#
# 图结构：
#   START → router → [conditional] → writer → END
#                              ↓
#                              → coder → END
#                              ↓
#                              → analyst → END
#
# 关键设计决策：
#   1. 使用 MemorySaver 实现多轮对话记忆
#   2. 每个专业 Agent 直接连接到 END（单轮交互）
#   3. 未来可扩展为循环结构，实现多轮协作
# ============================================================

# 创建图构建器
workflow = StateGraph(State)

# 添加所有节点
workflow.add_node("router", router_agent)
workflow.add_node("writer", writer_agent)
workflow.add_node("coder", coder_agent)
workflow.add_node("analyst", analyst_agent)

# 设置入口点：所有对话都从 router 开始
workflow.add_edge(START, "router")

# 添加条件边：从 router 根据 route_to_agent 的返回值分支
# 这是一个"一对多"的分支，根据 State 动态选择
workflow.add_conditional_edges(
    "router",           # 源节点
    route_to_agent,     # 条件函数
    ["writer", "coder", "analyst"]  # 可能的目标节点列表
)

# 添加结束边：每个专业 Agent 执行完后直接结束
workflow.add_edge("writer", END)
workflow.add_edge("coder", END)
workflow.add_edge("analyst", END)

# 初始化记忆保存器（可选，用于多轮对话）
memory = MemorySaver()

# 编译图
# 传入 checkpointer 实现跨轮次的状态持久化
app = workflow.compile(checkpointer=memory)

# ============================================================
# Step 7: 运行演示
# ============================================================


def run_demo():
    """
    运行 Multi-Agent 演示，展示不同场景下的智能路由和专业处理。
    """
    print("=" * 70)
    print("LangGraph Multi-Agent 多智能体协作系统 Demo")
    print("=" * 70)
    print()
    print("本 Demo 演示了一个路由+三专业的 Multi-Agent 架构：")
    print("  - router_agent:   分析用户意图，决定分配给哪个 Agent")
    print("  - writer_agent:   写作助手，擅长创意写作和文案")
    print("  - coder_agent:    编程助手，擅长代码和技术问题")
    print("  - analyst_agent:  分析助手，擅长逻辑推理和数据分析")
    print()
    print("=" * 70)

    # 使用 thread_id 实现会话隔离
    config = {"configurable": {"thread_id": "multi_agent_demo"}}

    # 测试场景列表
    test_cases = [
        {
            "name": "写作任务",
            "question": "请帮我写一段关于春天的短诗，要优美动人。",
            "expected": "writer"
        },
        {
            "name": "编程任务",
            "question": "用 Python 写一个冒泡排序算法，并加上注释。",
            "expected": "coder"
        },
        {
            "name": "分析任务",
            "question": "帮我分析一下为什么现在的年轻人更喜欢养猫而不是养狗？",
            "expected": "analyst"
        },
        {
            "name": "多轮对话（测试记忆）",
            "question": "刚才那首诗能帮我改成现代诗风格吗？",
            "expected": "writer（应记住上文）"
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n【测试 {i}】{test['name']}（预期：{test['expected']}）")
        print("-" * 70)
        print(f"用户：{test['question']}")
        print()

        # 运行图
        events = app.invoke(
            {"messages": [HumanMessage(content=test["question"])]},
            config=config
        )

        # 获取最后一条 AI 回复
        last_ai_message = None
        for msg in reversed(events["messages"]):
            if isinstance(msg, AIMessage):
                last_ai_message = msg
                break

        if last_ai_message:
            print(f"助手：{last_ai_message.content}")
        print()

    # 可视化图结构
    print("=" * 70)
    print("图结构可视化")
    print("=" * 70)
    print()
    try:
        print(app.get_graph().draw_ascii())
    except Exception:
        print("""
    +--------+     +--------+     +--------+
    | START  |---->| router |---->| writer |---> END
    +--------+     +--------+     +--------+
                          |
                          +-------->| coder  |---> END
                          |         +--------+
                          |
                          +-------->| analyst|---> END
                                    +--------+
        """)

    print()
    print("=" * 70)
    print("总结：Multi-Agent 核心概念")
    print("=" * 70)
    print("""
1. 路由模式（Router Pattern）
   - 由专门的 Agent 分析用户意图，决定任务分配
   - 使用条件边（conditional_edges）实现动态路由

2. 专业化 Agent（Specialized Agents）
   - 每个 Agent 有自己的系统提示词和专业领域
   - 职责清晰，边界明确

3. 共享状态（Shared State）
   - 所有 Agent 访问同一个 State
   - 通过 messages 共享对话历史

4. 扩展方向
   ✦ 添加 Supervisor：专业 Agent 执行后返回 Router 决策
   ✦ 添加循环：多 Agent 协作完成复杂任务
   ✦ 添加工具：每个 Agent 可以调用专业工具
   ✦ 添加 Handoff：Agent 之间可以直接移交任务
""")


if __name__ == "__main__":
    run_demo()

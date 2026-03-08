import os
import sys

sys.stdout.reconfigure(encoding="utf-8") # type: ignore

# ============================================================
# LangGraph 是什么？
#
# LangGraph 是 LangChain 团队推出的框架，专门用来构建
# 有状态、可循环的 AI 应用（Agent、多轮对话、工作流等）。
#
# 核心概念：
#
#   State（状态）
#     ─ 一个贯穿整个图的"共享内存"，每个节点都能读写它。
#     ─ 本 demo 中 State 就是对话历史（消息列表）。
#
#   Node（节点）
#     ─ 图中的每个处理单元，本质上就是一个 Python 函数。
#     ─ 接收当前 State，返回对 State 的更新。
#
#   Edge（边）
#     ─ 连接节点的箭头，决定执行顺序。
#     ─ 普通边：A → B，固定跳转。
#     ─ 条件边：根据当前 State 动态决定下一跳。
#
#   Graph（图）
#     ─ 把节点和边组合起来，形成完整的执行流程。
#     ─ 必须指定 START（入口）和 END（出口）。
#
# 本 demo 实现的流程：
#
#   用户输入
#       ↓
#   [START] → [chatbot 节点] → [END]
#
#   chatbot 节点：把用户消息加入历史，调用模型，返回回复
#
# 与普通 LangChain 的区别：
#   LangChain：手动维护 history 列表，手动调用 model.invoke()
#   LangGraph：把流程定义成图，State 自动流转，更容易扩展
# ============================================================

from typing import Annotated
from typing_extensions import TypedDict

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ============================================================
# Step 1：定义 State（状态）
#
# State 是图中节点之间共享的数据结构，用 TypedDict 定义。
#
# messages 字段：
#   - 类型是 list[BaseMessage]（消息列表）
#   - Annotated[..., add_messages] 是关键！
#     它告诉 LangGraph：当节点返回新消息时，
#     不要覆盖旧列表，而是"追加"到原列表末尾。
#     （这就是 add_messages 减法器/reducer 的作用）
# ============================================================
class State(TypedDict):
    # Annotated[实际类型, 附加元数据] 是 Python 的类型提示语法：
    #   - 第一个参数 list[BaseMessage]：字段的实际类型，即消息列表，
    #     每项都是 BaseMessage（HumanMessage/AIMessage/SystemMessage 的父类）
    #   - 第二个参数 add_messages：附加给 LangGraph 读取的元数据（reducer），
    #     不改变类型本身，但告诉框架"更新此字段时追加而非覆盖"
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# Step 2：初始化模型
# ============================================================
model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    temperature=0.7,
)


# ============================================================
# Step 3：定义节点函数（Node）
#
# 每个节点函数的签名固定为：
#   def node_name(state: State) -> dict
#
# 接收当前 State，返回对 State 的"增量更新"（不是全量替换）。
#
# 这里的 chatbot 节点：
#   1. 从 state["messages"] 里取出全部对话历史
#   2. 把历史交给模型，让模型生成回复
#   3. 返回 {"messages": [ai_reply]}
#      ↑ LangGraph 会自动把这条消息追加到历史里（因为 add_messages）
# ============================================================
SYSTEM_PROMPT = SystemMessage(
    content="你是一个友好、专业的 AI 助手。回答要简洁清晰，如果不确定就如实说。"
)

def chatbot(state: State) -> dict:
    """
    聊天节点：调用模型，生成回复。

    state["messages"] 是到目前为止的完整对话历史。
    把系统提示 + 历史一起发给模型，得到回复后返回。
    """
    # 构建完整的消息列表：系统提示 + 对话历史
    all_messages = [SYSTEM_PROMPT] + state["messages"]

    # 调用模型
    response = model.invoke(all_messages)

    # 返回新增的消息（LangGraph 会自动追加到 state["messages"]）
    return {"messages": [response]}


# ============================================================
# Step 4：构建图（Graph）
#
# StateGraph(State)：创建一个以 State 为状态的图
# add_node：注册节点（节点名, 节点函数）
# add_edge：添加边（从哪里 → 到哪里）
# compile()：编译图，得到可执行的 app
# ============================================================
# 创建图构建器
builder = StateGraph(State)

# 注册节点
builder.add_node("chatbot", chatbot)

# 添加边：START → chatbot → END
# START 是内置的入口节点，END 是内置的出口节点
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# 编译图，得到可运行的应用
app = builder.compile()


# ============================================================
# Step 5：运行聊天机器人
#
# app.invoke({"messages": [HumanMessage(...)]})
#   ─ 传入初始 State（包含用户消息）
#   ─ 图自动按 START → chatbot → END 执行
#   ─ 返回最终 State（包含完整对话历史）
#
# 多轮对话的关键：
#   每次把上一轮的完整历史（state["messages"]）传给下一轮
#   LangGraph 的 add_messages reducer 会自动合并，不会重复
# ============================================================

def run_demo():
    print("=" * 60)
    print("LangGraph 聊天机器人 Demo")
    print("=" * 60)
    print()

    # ──────────────────────────────────────────────────────────
    # Part 1：单轮对话
    # 最简单的用法：传入一条消息，得到一条回复
    # ──────────────────────────────────────────────────────────
    print("【Part 1】单轮对话")
    print("-" * 40)

    # 调用图：传入初始状态
    result = app.invoke({
        "messages": [HumanMessage(content="你好！请用一句话介绍一下自己。")]
    })

    # result 就是最终的 State
    # result["messages"] 包含：[HumanMessage, AIMessage]
    last_message = result["messages"][-1]   # 取最后一条（AI 的回复）
    print(f"用户：你好！请用一句话介绍一下自己。")
    print(f"助手：{last_message.content}")
    print()


    # ──────────────────────────────────────────────────────────
    # Part 2：多轮对话
    #
    # LangGraph 本身不自动记忆历史（默认无持久化）。
    # 多轮对话的做法：把上一轮 result["messages"] 保留，
    # 下一轮追加新的 HumanMessage 再传入。
    # ──────────────────────────────────────────────────────────
    print("【Part 2】多轮对话")
    print("-" * 40)

    # 第一轮
    question1 = "LangGraph 和 LangChain 有什么区别？"
    state = app.invoke({"messages": [HumanMessage(content=question1)]})
    print(f"用户：{question1}")
    print(f"助手：{state['messages'][-1].content}")
    print()

    # 第二轮：在上一轮 state["messages"] 基础上追加新问题
    question2 = "那我什么时候该用 LangGraph？"
    state = app.invoke({
        "messages": state["messages"] + [HumanMessage(content=question2)]
    })
    print(f"用户：{question2}")
    print(f"助手：{state['messages'][-1].content}")
    print()

    # 第三轮：继续追加
    question3 = "能给我一个最简单的 LangGraph 代码示例吗？"
    state = app.invoke({
        "messages": state["messages"] + [HumanMessage(content=question3)]
    })
    print(f"用户：{question3}")
    print(f"助手：{state['messages'][-1].content}")
    print()

    # 查看完整对话历史
    print("-" * 40)
    print(f"完整对话历史共 {len(state['messages'])} 条消息：")
    for i, msg in enumerate(state["messages"]):
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        preview = str(msg.content)[:40].replace("\n", " ")
        print(f"  [{i+1}] {role}：{preview}...")
    print()


    # ──────────────────────────────────────────────────────────
    # Part 3：流式输出（Streaming）
    #
    # app.stream() 与 app.invoke() 用法相同，
    # 区别在于它逐步返回每个节点执行后的 State 快照。
    # ──────────────────────────────────────────────────────────
    print("【Part 3】流式输出（逐节点返回状态快照）")
    print("-" * 40)

    print("用户：用一句话解释什么是「图」（Graph）在 LangGraph 中的含义？")
    print("助手（流式）：", end="", flush=True)

    for chunk in app.stream({
        "messages": [HumanMessage(content="用一句话解释什么是「图」（Graph）在 LangGraph 中的含义？")]
    }):
        # chunk 是一个 dict，key 是节点名，value 是该节点返回的状态更新
        # 格式：{"chatbot": {"messages": [AIMessage(...)]}}
        if "chatbot" in chunk:
            ai_msg = chunk["chatbot"]["messages"][-1]
            print(ai_msg.content, end="", flush=True)

    print()
    print()


    # ──────────────────────────────────────────────────────────
    # Part 4：可视化图结构（ASCII）
    #
    # 用 app.get_graph().draw_ascii() 打印图的结构，
    # 直观看到节点和边的连接关系。
    # ──────────────────────────────────────────────────────────
    print("【Part 4】图结构可视化")
    print("-" * 40)
    print("当前图的结构（ASCII 示意）：")
    print()
    try:
        print(app.get_graph().draw_ascii())
    except Exception:
        # 部分环境可能缺少依赖，手动画出结构
        print("  [__start__]")
        print("       ↓")
        print("   [chatbot]")
        print("       ↓")
        print("  [__end__]")
    print()


    print("=" * 60)
    print("总结：LangGraph 核心概念")
    print("=" * 60)
    print("""
  State   —— 图中流转的共享状态（TypedDict），本 demo 是消息列表
  Node    —— 处理单元（Python 函数），接收 State，返回状态增量
  Edge    —— 连接节点的箭头，决定执行顺序
  Graph   —— 节点 + 边组成的完整工作流，compile() 后可运行

  add_messages reducer：
    让 messages 字段只追加，不覆盖，是多轮对话的关键

  invoke()  —— 同步运行图，返回最终 State
  stream()  —— 流式运行图，逐步返回每个节点的状态快照

  扩展方向：
    ✦ 加入 MemorySaver —— 持久化对话历史，跨会话记忆
    ✦ 加入工具节点    —— 让聊天机器人能调用外部工具（见后续 demo）
    ✦ 加入条件边      —— 根据模型意图动态路由到不同节点
""")


if __name__ == "__main__":
    run_demo()

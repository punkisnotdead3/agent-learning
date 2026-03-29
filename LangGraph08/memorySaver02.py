import os
import sys

sys.stdout.reconfigure(encoding="utf-8") # type: ignore

# ============================================================
# LangGraph 记忆功能（Memory & Checkpointing）是什么？
#
# 在 demo01 中，我们展示了通过手动把上一轮的 history 传给图来实现多轮对话。
# 但在真实的业务场景中（比如基于 Web 的应用），每次用户的请求都是独立的，
# 我们不能指望前端一直把所有历史记录传来传去。
#
# LangGraph 提供了 Checkpointer（检查点保存器）机制，用于自动持久化 State。
# 
# 核心概念：
#
#   MemorySaver（内存保存器）
#     ─ 这是 LangGraph 内置的、最简单的 Checkpointer。
#     ─ 它会将每一步的状态（State）保存在内存中。
#     ─ 实际生产中可替换为 SqliteSaver、PostgresSaver、RedisSaver 等。
#
#   thread_id（线程 ID）
#     ─ 既然要把状态保存下来，就需要一个标识符来区分不同用户、不同会话。
#     ─ `thread_id` 就是这个标识符。当你运行图时，带上相同的 `thread_id`，
#       LangGraph 就会自动帮你把对应的历史 State 取出来继续执行。
#
# 本 demo 实现的流程：
#
#   1. 初始化一个 MemorySaver。
#   2. 在 compile() 图的时候，把它作为 checkpointer 传入。
#   3. 使用不同的 thread_id 测试跨会话的状态隔离。
# ============================================================

from typing import Annotated
from typing_extensions import TypedDict

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 【新增】引入内存检查点
from langgraph.checkpoint.memory import MemorySaver


# ============================================================
# Step 1：定义 State（状态）
# （与 demo01 完全一致）
# ============================================================
class State(TypedDict):
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
# （与 demo01 完全一致）
# ============================================================
SYSTEM_PROMPT = SystemMessage(
    content="你是一个友好、专业的 AI 助手。回答要简洁清晰。"
)

def chatbot(state: State) -> dict:
    all_messages = [SYSTEM_PROMPT] + state["messages"]
    response = model.invoke(all_messages)
    return {"messages": [response]}


# ============================================================
# Step 4：构建图并加入记忆机制（Checkpointer）
# ============================================================
builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# 【核心差异 1】初始化一个 Checkpointer（这里使用放在内存里的 MemorySaver）
memory = MemorySaver()

# 【核心差异 2】在编译时，将 checkpointer 传入
app = builder.compile(checkpointer=memory)


# ============================================================
# Step 5：运行带记忆的聊天机器人
# ============================================================
def run_demo():
    print("=" * 60)
    print("LangGraph 记忆功能 (MemorySaver) Demo")
    print("=" * 60)
    print()

    # ──────────────────────────────────────────────────────────
    # 会话 1：用户 A 开始聊天
    # ──────────────────────────────────────────────────────────
    print("【会话 1】标识符：user_a_session")
    print("-" * 40)
    
    # 定义配置，指定当前的 thread_id
    config_a = {"configurable": {"thread_id": "user_a_session"}}

    # 第 1 轮对话（用户 A 告知名字）
    question_a1 = "你好，我叫张三，是一个程序员。"
    print(f"用户 A：{question_a1}")
    
    # 只需要传本次的新消息即可，历史记录完全交给 checkpointer 托管！
    events_a1 = app.invoke({"messages": [HumanMessage(content=question_a1)]}, config=config_a)
    print(f"助手：{events_a1['messages'][-1].content}")
    print()

    # 第 2 轮对话（检查是否记住名字）
    question_a2 = "你还记得我叫什么名字，以及我的职业吗？"
    print(f"用户 A：{question_a2}")
    
    # 依然只需传新消息，注意一定要带上 config_a 才能触发相同的 thread_id
    events_a2 = app.invoke({"messages": [HumanMessage(content=question_a2)]}, config=config_a)
    print(f"助手：{events_a2['messages'][-1].content}")
    print()

    # ──────────────────────────────────────────────────────────
    # 会话 2：用户 B 开始聊天
    # ──────────────────────────────────────────────────────────
    print("【会话 2】标识符：user_b_session")
    print("-" * 40)
    
    # 使用一个全新的 thread_id
    config_b = {"configurable": {"thread_id": "user_b_session"}}

    # 第 1 轮对话（检查用户 B 能不能看到用户 A 的信息）
    question_b1 = "你好，你知道我是谁吗？"
    print(f"用户 B：{question_b1}")
    
    events_b1 = app.invoke({"messages": [HumanMessage(content=question_b1)]}, config=config_b)
    print(f"助手：{events_b1['messages'][-1].content}")
    print()


    # ──────────────────────────────────────────────────────────
    # 会话 3：回到用户 A
    # ──────────────────────────────────────────────────────────
    print("【会话 3】重新切回标识符：user_a_session")
    print("-" * 40)

    question_a3 = "我回来了。我们刚才聊到哪了？"
    print(f"用户 A：{question_a3}")
    
    # 复用最初的 config_a
    events_a3 = app.invoke({"messages": [HumanMessage(content=question_a3)]}, config=config_a)
    print(f"助手：{events_a3['messages'][-1].content}")
    print()


    print("=" * 60)
    print("总结：Checkpointer 的优势")
    print("=" * 60)
    print("""
  1. 状态自动化：无需在外部手动拼接 list(旧历史) + 新消息。每次调用只需传入新内容。
  2. 会话隔离：通过 `thread_id`，轻松实现多用户、多会话并发互不干扰。
  3. 支持持久化：将 MemorySaver 换成 PostgresSaver 或 SqliteSaver，
     即可实现把对话历史永久存入数据库，重启后依然能够恢复。
""")

if __name__ == "__main__":
    run_demo()
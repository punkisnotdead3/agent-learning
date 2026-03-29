import os
import sys
from typing import Annotated, Literal

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

# 设置输出编码，防止 Windows 下乱码
sys.stdout.reconfigure(encoding="utf-8") # type: ignore

# ============================================================
# ToolNode 是什么？
#
# 在 LangGraph 中，ToolNode 是一个内置的节点，专门用来执行模型请求调用的工具。
#
# 核心工作流：
# 1. 聊天节点（chatbot）：模型决定是否需要调用工具（通过 tool_calls 字段）。
# 2. 条件边（conditional edge）：根据模型回复判断，如果有 tool_calls 则跳转到 tools 节点，否则结束。
# 3. 工具节点（tools）：ToolNode 接收包含 tool_calls 的消息，执行对应的 Python 函数，并返回 ToolMessage。
# 4. 回到聊天节点：模型根据工具执行结果，生成最终回复。
# ============================================================

# --- Step 1: 定义工具 ---

@tool
def search(query: str):
    """查询天气、新闻等实时信息。"""
    # 这是一个模拟工具，实际中可以调用搜索 API
    if "天气" in query:
        return "北京今天晴，气温 15°C 到 25°C。"
    return "搜索结果：关于 " + query + " 的相关内容。"

tools = [search]
# 创建预置的工具节点
tool_node = ToolNode(tools)

# --- Step 2: 定义状态 ---

class State(TypedDict):
    # 使用 add_messages 确保消息是追加而非覆盖
    messages: Annotated[list[BaseMessage], add_messages]

# --- Step 3: 定义模型并绑定工具 ---

model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    temperature=0, # 建议设为 0，让工具调用更稳定
).bind_tools(tools)

# --- Step 4: 定义逻辑节点 ---

def chatbot(state: State):
    """
    模型节点：决定是否需要工具调用。
    """
    return {"messages": [model.invoke(state["messages"])]}

def should_continue(state: State) -> Literal["tools", END]:
    """
    条件边逻辑：判断下一步是去执行工具还是结束。
    """
    messages = state["messages"]
    last_message = messages[-1]
    # 如果最后一条消息包含工具调用请求，则前往 "tools" 节点
    if last_message.tool_calls:
        return "tools"
    # 否则，结束流程
    return END

# --- Step 5: 构建图 ---

workflow = StateGraph(State)

# 添加节点
workflow.add_node("agent", chatbot)
workflow.add_node("tools", tool_node)

# 设置入口
workflow.add_edge(START, "agent")

# 设置条件边
# 从 "agent" 出发，根据 should_continue 的返回值决定去 "tools" 还是 END
workflow.add_conditional_edges(
    "agent",
    should_continue,
)

# 从 "tools" 执行完后，必须回到 "agent" 让模型根据工具结果说话
workflow.add_edge("tools", "agent")

# 编译
app = workflow.compile()

# --- Step 6: 运行测试 ---

def run_demo():
    print("=" * 60)
    print("LangGraph ToolNode 示例 (工具调用流)")
    print("=" * 60)
    print()

    # 场景 1：不需要工具的问题
    print("【测试 1：普通闲聊】")
    inputs = {"messages": [HumanMessage(content="你好，请介绍一下你自己。")]}
    for chunk in app.stream(inputs, stream_mode="values"):
        last_msg = chunk["messages"][-1]
        role = "助手" if last_msg.type == "ai" else "用户"
        if last_msg.content:
            print(f"{role}: {last_msg.content}")
    print("\n" + "-"*40 + "\n")

    # 场景 2：需要触发工具的问题
    print("【测试 2：工具调用】")
    inputs = {"messages": [HumanMessage(content="帮我查一下北京的天气。")]}
    
    # 我们使用 stream 模式来观察节点流转
    for output in app.stream(inputs):
        # output 是一个字典，key 是当前执行完的节点名
        for node_name, state_update in output.items():
            print(f"--- 节点执行完: {node_name} ---")
            # 打印该节点产生的新消息
            for msg in state_update.get("messages", []):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    print(f"模型请求调用工具: {msg.tool_calls[0]['name']}({msg.tool_calls[0]['args']})")
                elif msg.type == "tool":
                    print(f"工具执行结果: {msg.content}")
                else:
                    print(f"回复内容: {msg.content}")
    
    print()
    print("=" * 60)
    print("图结构可视化：")
    try:
        print(app.get_graph().draw_ascii())
    except:
        print(" (agent) -> [conditional] -> (tools)")
        print("    ^                           |")
        print("    └---------------------------┘")

if __name__ == "__main__":
    run_demo()

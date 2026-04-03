import os
import sys
from typing import Annotated, Literal

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# 设置输出编码，防止 Windows 下乱码
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

# ============================================================
# LangGraph 反思机制（Reflection）Demo
#
# 什么是 Reflection（反思机制）？
#
# 很多 Agent 不是“一次生成就结束”，而是：
#   1. 先产出一个初稿
#   2. 再由另一个节点检查这份初稿哪里不够好
#   3. 根据反馈继续修改
#   4. 达到满意条件后结束
#
# 这就是一种非常常见的 Agent 模式：Generate -> Reflect -> Revise
#
# 典型应用场景：
#   - 写作：先写文章，再审稿，再润色
#   - 代码：先生成代码，再 review，再修正
#   - 方案设计：先给方案，再自我评估，再补充风险与细节
#
# 本 demo 的教学目标：
#   - 理解“反思节点”在 LangGraph 中如何建模
#   - 理解“循环边 + 条件边”如何控制多轮修订
#   - 理解状态里如何保存草稿、反馈、轮次计数
#
# 本 demo 的流程：
#
#      START
#        ↓
#    generate  —— 先生成初稿
#        ↓
#    reflect   —— 审查初稿，给出反馈，并判断是否通过
#        ↓
#   [条件边]
#    ├─ 通过 → END
#    └─ 未通过 → revise —— 根据反馈改写
#                       ↓
#                    reflect
#
# ============================================================


# ============================================================
# Step 1：定义 State（状态）
#
# 除了 messages 之外，我们额外保存：
#   - task: 用户真正要完成的任务
#   - draft: 当前版本的草稿
#   - reflection: 最近一次反思意见
#   - revision_count: 当前已经修订了几轮
#   - is_approved: 是否已经达到可接受质量
# ============================================================
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    task: str
    draft: str
    reflection: str
    revision_count: int
    is_approved: bool


# ============================================================
# Step 2：初始化模型
#
# 为了教学简单，这里使用同一个模型完成：
#   - 生成初稿
#   - 反思评审
#   - 根据反馈修订
#
# 在真实项目里，也可以拆成两个模型：
#   - generator 负责创作
#   - critic/evaluator 负责评估
# ============================================================
model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    temperature=0.7,
)

MAX_REVISIONS = 2


# ============================================================
# Step 3：定义系统提示词
# ============================================================
GENERATOR_SYSTEM_PROMPT = SystemMessage(
    content="""你是一位教学型写作助手。
你的任务是根据用户要求，产出一版清晰、准确、适合教学演示的内容。
要求：
- 内容尽量结构化
- 表达清晰，避免空话
- 先给出可读性好的初稿"""
)

REFLECTOR_SYSTEM_PROMPT = SystemMessage(
    content="""你是一位严格但建设性的审稿人。
请根据用户任务检查当前草稿，并从以下角度评估：
1. 是否准确回应了任务
2. 是否结构清晰
3. 是否适合教学演示
4. 是否有明显遗漏

输出必须严格使用下面格式：
DECISION: approve 或 revise
FEEDBACK: 具体反馈内容

要求：
- 如果草稿已经足够好，可输出 approve
- 如果还有明显问题，输出 revise，并给出具体可执行建议
- 不要输出多余内容"""
)

REVISION_SYSTEM_PROMPT = SystemMessage(
    content="""你是一位擅长改稿的助手。
你会根据“原草稿 + 审稿反馈”生成更好的新版本。
要求：
- 必须响应反馈中的关键问题
- 保留原文中正确的部分
- 改得更清晰、更完整、更适合教学"""
)


# ============================================================
# Step 4：定义节点函数
# ============================================================
def generate_draft(state: State) -> dict:
    """
    第一次生成草稿。
    """
    task = state["task"]
    response = model.invoke([
        GENERATOR_SYSTEM_PROMPT,
        HumanMessage(content=f"请完成这个任务：{task}")
    ])

    print("  [generate] 已生成初稿")

    return {
        "draft": str(response.content),
        "messages": [AIMessage(content=f"初稿：\n{response.content}")],
    }



def reflect_draft(state: State) -> dict:
    """
    审查当前草稿，给出通过/修改决定。
    """
    task = state["task"]
    draft = state["draft"]

    response = model.invoke([
        REFLECTOR_SYSTEM_PROMPT,
        HumanMessage(content=f"用户任务：{task}\n\n当前草稿：\n{draft}")
    ])

    content = str(response.content).strip()
    lowered = content.lower()

    approved = "decision: approve" in lowered
    feedback = content

    print(f"  [reflect] 审查结果：{'通过' if approved else '需要修改'}")

    return {
        "reflection": feedback,
        "is_approved": approved,
        "messages": [AIMessage(content=f"反思意见：\n{feedback}")],
    }



def revise_draft(state: State) -> dict:
    """
    根据反思意见修订草稿。
    """
    task = state["task"]
    draft = state["draft"]
    reflection = state["reflection"]
    revision_count = state["revision_count"] + 1

    response = model.invoke([
        REVISION_SYSTEM_PROMPT,
        HumanMessage(
            content=(
                f"用户任务：{task}\n\n"
                f"原草稿：\n{draft}\n\n"
                f"审稿反馈：\n{reflection}\n\n"
                f"请输出一版改进后的新草稿。"
            )
        )
    ])

    print(f"  [revise] 已完成第 {revision_count} 轮修订")

    return {
        "draft": str(response.content),
        "revision_count": revision_count,
        "messages": [AIMessage(content=f"第 {revision_count} 轮修订稿：\n{response.content}")],
    }


# ============================================================
# Step 5：定义条件边逻辑
#
# 什么时候结束？
#   - 审稿通过：结束
#   - 或者修订次数达到上限：结束
#   - 否则继续去 revise 节点
# ============================================================
def should_continue(state: State) -> Literal["revise", END]:
    if state["is_approved"]:
        return END

    if state["revision_count"] >= MAX_REVISIONS:
        return END

    return "revise"


# ============================================================
# Step 6：构建图
# ============================================================
workflow = StateGraph(State)

workflow.add_node("generate", generate_draft)
workflow.add_node("reflect", reflect_draft)
workflow.add_node("revise", revise_draft)

workflow.add_edge(START, "generate")
workflow.add_edge("generate", "reflect")
workflow.add_conditional_edges("reflect", should_continue)
workflow.add_edge("revise", "reflect")

app = workflow.compile()


# ============================================================
# Step 7：运行 Demo
# ============================================================
def run_demo():
    print("=" * 70)
    print("LangGraph Reflection 反思机制 Demo")
    print("=" * 70)
    print()

    task = "请写一段面向初学者的说明，解释什么是 LangGraph 的条件边，并给一个简短例子。"

    print("【用户任务】")
    print(task)
    print()

    initial_state = {
        "messages": [HumanMessage(content=task)],
        "task": task,
        "draft": "",
        "reflection": "",
        "revision_count": 0,
        "is_approved": False,
    }

    final_state = None

    print("【执行过程】")
    print("-" * 70)
    for event in app.stream(initial_state):
        for node_name, state_update in event.items():
            print(f"--- 节点执行完: {node_name} ---")

            if "draft" in state_update:
                preview = state_update["draft"][:160].replace("\n", " ")
                print(f"当前草稿预览: {preview}...")

            if "reflection" in state_update:
                preview = state_update["reflection"][:160].replace("\n", " ")
                print(f"反思意见预览: {preview}...")

            if "revision_count" in state_update:
                print(f"修订轮次: {state_update['revision_count']}")

            final_state = state_update
        print()

    result = app.invoke(initial_state)

    print("【最终结果】")
    print("-" * 70)
    print(result["draft"])
    print()
    print(f"是否通过审查: {result['is_approved']}")
    print(f"总修订轮次: {result['revision_count']}")
    print()

    print("=" * 70)
    print("总结：Reflection 模式的关键点")
    print("=" * 70)
    print("""
1. 生成节点（generate）
   - 先产出一个初稿

2. 反思节点（reflect）
   - 不直接生成最终答案，而是专门负责“挑问题”
   - 本质上相当于一个 reviewer / critic

3. 修订节点（revise）
   - 读取当前草稿和反馈，生成新版本

4. 条件边（conditional edges）
   - 控制是否继续循环
   - 这就是 LangGraph 做 Agent 工作流很强的地方

5. 轮次上限（MAX_REVISIONS）
   - 非常重要
   - 否则模型可能一直反思下去，无法停止
""")

    print("图结构可视化：")
    try:
        print(app.get_graph().draw_ascii())
    except Exception:
        print("START -> generate -> reflect -> [approve: END / revise: revise -> reflect]")


if __name__ == "__main__":
    run_demo()

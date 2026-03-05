import os
import sys
from typing import Literal

sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# demo04_human_in_the_loop.py -- Tool 开发最佳实践：Human-in-the-loop
#
# 【核心知识点】
# 1. ask_human：Agent 无法自主决策时，主动暂停并请求用户帮助
# 2. require_confirmation：危险操作执行前必须通过人工确认
# 3. 把关键行为的决策权和责任交还给用户，而不是让 AI 自作主张
#
# 【何时需要 Human-in-the-loop？】
#   - 不可逆操作（删除数据、发送消息、扣款等）
#   - 涉及外部可见效果（发邮件、发 Slack、推送通知）
#   - 多个选项等价，需要用户偏好来决定
#   - 用户意图不明确，存在歧义
#
# 【实现模式】
#   ask_human()           → 通用求助工具，Agent 主动发起询问
#   confirmed 参数        → 危险工具要求调用方先确认再执行
#   prompt 中写安全规则  → 在系统提示中明确告知 Agent 何时必须确认
# ============================================================

from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate


# ============================================================
# ask_human：Agent 的"求助"工具
#
# 设计要点：
#   - Docstring 的"使用时机"要写得够具体，让 LLM 知道什么情况下该主动问
#   - 参数 question 要求 Agent 组织清晰的问题，而不是含糊地发问
#   - 返回格式统一加"用户回答："前缀，方便 Agent 解析用户意图
# ============================================================
@tool
def ask_human(question: str) -> str:
    """
    当遇到无法自动决策的情况时，向用户提问并等待回答。

    使用时机（以下情况必须调用）：
    - 即将执行不可逆操作（删除数据、发送消息、扣款）前，需要用户授权
    - 有多个等价选项，需要用户偏好来决定
    - 用户意图不明确，存在歧义，需要澄清
    - 缺少关键信息，无法自动继续任务

    参数：question - 向用户提出的问题，要清晰具体，说明背景和选项

    示例：
      ask_human("即将删除 2024Q1 的全部 3 条订单，此操作不可恢复。确认继续吗？(是/否)")
      ask_human("找到两件跑鞋：Nike ¥799 和 Adidas ¥699，你更倾向于哪个品牌？")
    """
    print(f"\n  [Agent 请求确认]")
    print(f"  {question}")
    answer = input("  你的回答：").strip()
    return f"用户回答：{answer}"


# ============================================================
# 模拟数据
# ============================================================
ORDERS_DB = {
    "2024Q1": ["订单-001", "订单-002", "订单-003"],
    "2024Q2": ["订单-004", "订单-005"],
    "2024Q3": ["订单-006", "订单-007", "订单-008", "订单-009"],
}

EMAIL_GROUPS = {
    "all_users":      50000,
    "vip_users":      2000,
    "inactive_users": 8000,
}


# ============================================================
# 危险工具 1：批量删除订单
#
# 设计要点：
#   - confirmed=False 默认值：强制调用方先确认
#   - Docstring 里写明"必须先用 ask_human() 获得确认"
#   - 两个步骤写进示例流程，降低 LLM 跳步的风险
# ============================================================
@tool
def list_order_batches() -> str:
    """
    列出所有订单批次及订单数量。

    在执行任何订单操作前，先调用此工具了解数据规模。
    """
    lines = ["当前订单批次："]
    for batch, orders in ORDERS_DB.items():
        lines.append(f"  - {batch}：{len(orders)} 条订单")
    return "\n".join(lines)


@tool
def delete_order_batch(batch_id: str, confirmed: bool = False) -> str:
    """
    删除指定批次的全部订单数据（不可恢复！）。

    安全规则（必须遵守）：
    1. 先调用 list_order_batches() 了解数据规模
    2. 再调用 ask_human() 让用户明确确认，获得"是/确认/yes"类回答
    3. 用户同意 → confirmed=True 调用；用户拒绝 → confirmed=False 调用

    示例流程：
      list_order_batches()
      → ask_human("确认删除 2024Q1 的 3 条订单吗？不可恢复！(是/否)")
      → delete_order_batch("2024Q1", confirmed=True 或 False)
    """
    if batch_id not in ORDERS_DB:
        return f"批次 '{batch_id}' 不存在，请先用 list_order_batches() 确认批次名称"

    count = len(ORDERS_DB[batch_id])

    if not confirmed:
        return (
            f"操作已取消：删除 '{batch_id}' 的 {count} 条订单属于高危操作。\n"
            f"请先调用 ask_human() 获得用户的明确同意，再以 confirmed=True 重新调用"
        )

    del ORDERS_DB[batch_id]
    return f"已删除批次 '{batch_id}' 的 {count} 条订单"


# ============================================================
# 危险工具 2：群发邮件
#
# 设计要点：
#   - 列出各 recipient_group 对应的受众规模，让 Agent 在确认时告知用户影响范围
#   - Docstring 明确说明"必须告知用户会影响多少人"
# ============================================================
@tool
def send_bulk_email(
    recipient_group: Literal["all_users", "vip_users", "inactive_users"],
    subject: str,
    body_preview: str,
) -> str:
    """
    向指定用户群发送批量邮件（高风险，发出后无法撤回）。

    发送前必须：
    1. 用 ask_human() 告知用户：收件人群体、预计发送人数、邮件主题
    2. 获得用户明确同意后才执行

    受众规模参考（ask_human 时需要告知用户）：
    - all_users：约 50,000 人
    - vip_users：约 2,000 人
    - inactive_users：约 8,000 人

    参数：
    - recipient_group：收件人群体
    - subject：邮件主题
    - body_preview：邮件正文前 100 字的预览
    """
    count = EMAIL_GROUPS[recipient_group]
    print(f"\n  [邮件系统] 已群发至 {recipient_group}（{count:,} 人）：{subject}")
    return f"邮件已发送：主题「{subject}」→ {recipient_group}（{count:,} 人）"


# ============================================================
# 运行演示（交互式，运行后会出现输入提示）
# ============================================================
def run_demo():
    model = ChatDeepSeek(
        model="deepseek-chat",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        temperature=0,
    )

    prompt = PromptTemplate.from_template(
        """你是一个数据管理助手。

安全规则（不可违反）：
1. 执行任何删除、发送消息等不可逆操作前，必须先调用 ask_human() 获得用户明确确认
2. ask_human 时要说清楚：要做什么、影响范围、是否可恢复
3. 用户回答"否/取消/no/不"时，立即停止，不得执行操作
4. 意图不明确时，主动用 ask_human() 澄清

可用工具：
{tools}

工具名称：{tool_names}

规则：按照 Thought/Action/Action Input/Observation/Final Answer 格式。

Question: {input}
Thought:{agent_scratchpad}"""
    )

    tools_list = [ask_human, list_order_batches, delete_order_batch, send_bulk_email]
    agent = create_react_agent(llm=model, tools=tools_list, prompt=prompt)
    executor = AgentExecutor(
        agent=agent, tools=tools_list,
        verbose=True, max_iterations=12, handle_parsing_errors=True,
    )

    print("=" * 60)
    print("【场景】Agent 执行危险操作前自动请求用户确认")
    print("（运行后会出现交互提示，请按提示输入 是/否）")
    print("=" * 60)

    result = executor.invoke({
        "input": "帮我把 2024Q1 的历史订单数据清理掉"
    })
    print(f"\n最终结果：{result['output']}")


if __name__ == "__main__":
    run_demo()

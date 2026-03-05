import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# demo02_llm_friendly_interface.py -- Tool 开发最佳实践：LLM 友好的接口设计
#
# 【核心知识点】
# 1. 花 50% 的时间打磨 Docstring，用自然语言 + Examples 引导准确传参
# 2. 单一责任原则：一个工具只做一件事，不做组合式"万能工具"
# 3. 在 Docstring 中写清楚"何时用"和"用完之后建议做什么"
#
# 【对比实验】
#   反例：一个 handle_user() 承包一切 → LLM 搞不清 action 参数的含义
#   正例：3 个职责清晰的小工具 → LLM 按顺序调用，结果更可靠
#
# 【为什么单一责任更好？】
#   - LLM 选工具时靠工具名 + Docstring，职责越单一越好选
#   - 参数越少，LLM 传参出错的概率越低
#   - 每个工具独立可测试、独立可复用
# ============================================================

from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate


# ============================================================
# 反例：单一职责违反 + 文档稀少的"万能工具"
#
# 问题清单：
#   1. action 参数的合法值是什么？LLM 完全靠猜
#   2. extra 参数什么时候需要填？不知道
#   3. 三件事混在一个工具里，LLM 会选错时机调用
#   4. 没有示例，没有后续操作建议
# ============================================================
@tool
def handle_user_bad(user_id: str, action: str, extra: str = "") -> str:
    """处理用户相关操作"""
    if action == "prefs":
        return f"用户 {user_id} 的偏好：喜欢运动装备，预算 1000 元，已开启通知"
    elif action == "recommend":
        return f"为用户 {user_id} 推荐：Nike 跑鞋 ¥799、优衣库外套 ¥299"
    elif action == "notify":
        return f"已发送通知给用户 {user_id}：{extra}"
    return "未知 action，支持：prefs / recommend / notify"


# ============================================================
# 正例：三个职责清晰的小工具
#
# 关键 Docstring 设计技巧：
#   1. 第一行：一句话说清楚这个工具做什么
#   2. "使用时机"：告诉 LLM 什么场景下调用我
#   3. "常见后续操作"：引导 LLM 知道拿到结果后该做什么
#   4. "示例"：真实的输入/输出示例，帮 LLM 对齐预期
# ============================================================
@tool
def get_user_preferences(user_id: str) -> dict:
    """
    获取用户的偏好设置和历史行为数据。

    使用时机：
    - 在推荐商品之前，先调用此工具了解用户喜好
    - 在发送通知之前，先检查 notification_enabled 字段

    常见后续操作：
    - 需要推荐商品 → 把返回的 liked_categories 和 budget 传给 recommend_products()
    - 需要发通知   → 检查 notification_enabled 为 True 才调用 send_notification()

    示例：
      get_user_preferences("u001")
      → {"liked_categories": ["clothing", "electronics"], "budget": 1000, "notification_enabled": True}
    """
    mock_db = {
        "u001": {"liked_categories": ["clothing", "electronics"], "budget": 1000, "notification_enabled": True},
        "u002": {"liked_categories": ["food"], "budget": 200, "notification_enabled": False},
        "u003": {"liked_categories": ["electronics"], "budget": 3000, "notification_enabled": True},
    }
    return mock_db.get(user_id, {"liked_categories": [], "budget": 500, "notification_enabled": False})


@tool
def recommend_products(liked_categories: list, budget: int) -> str:
    """
    根据用户的偏好类别和预算推荐合适的商品。

    注意：此工具只负责推荐，不负责查询用户信息。
    如果还不知道用户偏好，请先调用 get_user_preferences() 获取。

    参数说明：
    - liked_categories：用户喜欢的类别列表，如 ["clothing", "electronics"]
    - budget：用户预算（人民币整数），只推荐价格不超过此值的商品

    使用时机：
    - 已通过 get_user_preferences() 拿到用户偏好之后调用

    示例：
      recommend_products(["clothing"], 800)
      → "推荐商品：Nike 跑鞋 ¥799（符合服装偏好，在预算内）"
    """
    catalog = {
        "electronics": [("Sony 耳机", 1999), ("小米手环", 199), ("iPad mini", 3999)],
        "clothing":    [("Nike 跑鞋", 799), ("优衣库外套", 299), ("Adidas 运动裤", 399)],
        "food":        [("进口坚果礼盒", 128), ("有机燕麦", 59)],
    }

    results = []
    for cat in liked_categories:
        for name, price in catalog.get(cat, []):
            if price <= budget:
                results.append(f"{name} ¥{price}（类别：{cat}）")

    if not results:
        return f"在预算 ¥{budget} 内，{liked_categories} 类别暂无合适商品"
    return "为你推荐：\n" + "\n".join(f"  - {r}" for r in results)


@tool
def send_notification(user_id: str, message: str) -> str:
    """
    向用户发送一条消息通知。

    重要规则：发送前必须先用 get_user_preferences() 确认
    notification_enabled=True，为 False 则不应调用此工具。

    参数说明：
    - user_id：目标用户的 ID
    - message：通知内容，用自然语言描述，系统会自动格式化

    使用时机：
    - 推荐完商品，且用户开启了通知时，把推荐结果发给用户

    示例：
      send_notification("u001", "你关注的 Nike 跑鞋降价了，现在只要 ¥699！")
      → "通知已发送给用户 u001"
    """
    print(f"\n  [通知系统] 发送给 {user_id}：{message}")
    return f"通知已成功发送给用户 {user_id}"


# ============================================================
# 对比运行：同一个任务，用"坏工具"和"好工具"各跑一次
# 观察两者在调用链路和结果质量上的差异
# ============================================================
def run_comparison():
    model = ChatDeepSeek(
        model="deepseek-chat",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        temperature=0,
    )

    prompt = PromptTemplate.from_template(
        """你是一个电商购物助手。

可用工具：
{tools}

工具名称：{tool_names}

规则：按照 Thought/Action/Action Input/Observation/Final Answer 格式执行。

Question: {input}
Thought:{agent_scratchpad}"""
    )

    task = "用户 u001 想要商品推荐，如果他开启了通知功能，请把推荐结果发给他"

    # 反例：万能工具
    print("=" * 60)
    print("【反例】职责混乱、文档稀少的万能工具")
    print("=" * 60)
    bad_tools = [handle_user_bad]
    agent_bad = create_react_agent(llm=model, tools=bad_tools, prompt=prompt)
    exec_bad = AgentExecutor(
        agent=agent_bad, tools=bad_tools, verbose=True,
        max_iterations=6, handle_parsing_errors=True,
    )
    result_bad = exec_bad.invoke({"input": task})
    print(f"\n结果：{result_bad['output']}")

    # 正例：职责清晰的小工具
    print("\n\n" + "=" * 60)
    print("【正例】单一职责、文档详细的工具组合")
    print("=" * 60)
    good_tools = [get_user_preferences, recommend_products, send_notification]
    agent_good = create_react_agent(llm=model, tools=good_tools, prompt=prompt)
    exec_good = AgentExecutor(
        agent=agent_good, tools=good_tools, verbose=True,
        max_iterations=8, handle_parsing_errors=True,
    )
    result_good = exec_good.invoke({"input": task})
    print(f"\n结果：{result_good['output']}")


if __name__ == "__main__":
    run_comparison()

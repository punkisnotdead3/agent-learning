import os
import sys
import json

sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# demo01_type_safety.py -- Tool 开发最佳实践：类型安全与自动化
#
# 【核心知识点】
# 1. 用 Pydantic BaseModel 定义 Tool 的输入参数
# 2. 用 Literal 限制枚举值，减少模型出错概率
# 3. 用 Field(description=...) 引导 LLM 正确传参
# 4. 类型系统自动生成 JSON Schema，LLM 依赖它理解如何传参
#
# 【对比实验】
#   没有 Pydantic：LLM 自己猜 category / sort_by 的值，经常传错
#   有了 Pydantic ：LLM 看到枚举列表，准确率大幅提升
#
# 【关键设计决策】
#   - Literal["a", "b"] 比 str 好：明确告诉 LLM 只能填这几个值
#   - Optional[int] = None    ：有默认值的参数不强制填写
#   - Field(description=...)  ：核心！描述比参数名本身更重要
# ============================================================

from typing import Literal, Optional
from pydantic import BaseModel, Field, ValidationError
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate


# ============================================================
# 反例：没有类型约束的工具
#
# 问题：
#   - category 可以填任意字符串（"手机"、"3C"、"电子产品" 都可能出现）
#   - sort_by 的合法值是什么？LLM 完全不知道
#   - price_max 应该是数字还是字符串？不清楚
# ============================================================
@tool
def search_products_bad(query: str, category: str, sort_by: str) -> str:
    """搜索产品"""
    return f"搜索：query={query}, category={category}, sort_by={sort_by}"


# ============================================================
# 正例：用 Pydantic + Literal + Field 定义类型安全的工具
#
# 设计要点：
#   1. BaseModel 负责：自动生成 JSON Schema + 自动验证输入
#   2. Literal    负责：把枚举值写进 Schema，LLM 只能从中选一个
#   3. Field      负责：写清楚每个参数的含义和使用规则
#   4. Optional   负责：可选参数有明确的 None 默认值
# ============================================================
class SearchProductsInput(BaseModel):
    query: str = Field(
        description="用户的自然语言搜索意图。例如：'适合跑步的防水鞋'、'便宜的蓝牙耳机'"
    )
    category: Literal["electronics", "clothing", "food"] = Field(
        description=(
            "产品类别，只能从以下三个值中选一个：\n"
            "- electronics：电子产品（手机、电脑、耳机等）\n"
            "- clothing：服装（鞋子、衣服、包包等）\n"
            "- food：食品（零食、饮料、生鲜等）"
        )
    )
    price_max: Optional[int] = Field(
        None,
        description="最高价格（人民币整数）。只有当用户明确提到预算或价格上限时才填写，否则留 null"
    )
    sort_by: Literal["relevance", "price_asc", "rating"] = Field(
        default="relevance",
        description=(
            "排序方式，默认 relevance：\n"
            "- relevance：按相关性（默认，大多数情况用这个）\n"
            "- price_asc：按价格从低到高（用户说'便宜的'/'最低价'时使用）\n"
            "- rating：按评分从高到低（用户说'好评的'/'评分高的'时使用）"
        )
    )


@tool(args_schema=SearchProductsInput)
def search_products(
    query: str,
    category: Literal["electronics", "clothing", "food"],
    price_max: Optional[int] = None,
    sort_by: Literal["relevance", "price_asc", "rating"] = "relevance",
) -> str:
    """
    搜索商品，支持按类别、最高价格、排序方式筛选。

    示例：
    - 用户说"帮我找500以内的蓝牙耳机"
      → query="蓝牙耳机", category="electronics", price_max=500
    - 用户说"找评分最高的跑步鞋"
      → query="跑步鞋", category="clothing", sort_by="rating"
    - 用户说"推荐便宜的零食"
      → query="零食", category="food", sort_by="price_asc"
    """
    mock_catalog = [
        {"name": "Sony WH-1000XM5 耳机", "price": 2499, "rating": 4.8, "category": "electronics"},
        {"name": "Apple AirPods Pro",     "price": 1799, "rating": 4.9, "category": "electronics"},
        {"name": "小米蓝牙耳机 4 Pro",    "price": 299,  "rating": 4.5, "category": "electronics"},
        {"name": "Nike Air Zoom 跑鞋",    "price": 899,  "rating": 4.7, "category": "clothing"},
        {"name": "Adidas 运动T恤",        "price": 199,  "rating": 4.2, "category": "clothing"},
        {"name": "进口坚果礼盒",           "price": 128,  "rating": 4.6, "category": "food"},
        {"name": "有机燕麦片",             "price": 59,   "rating": 4.3, "category": "food"},
    ]

    results = [r for r in mock_catalog if r["category"] == category]

    if price_max is not None:
        results = [r for r in results if r["price"] <= price_max]

    if sort_by == "price_asc":
        results.sort(key=lambda x: x["price"])
    elif sort_by == "rating":
        results.sort(key=lambda x: x["rating"], reverse=True)

    if not results:
        price_hint = f"，价格≤¥{price_max}" if price_max else ""
        return f"未找到符合条件的商品（类别：{category}{price_hint}）"

    lines = [f"找到 {len(results)} 件商品（类别：{category}，排序：{sort_by}）："]
    for r in results:
        lines.append(f"  - {r['name']}：¥{r['price']}，评分 {r['rating']}")
    return "\n".join(lines)


# ============================================================
# 演示 1：Pydantic 自动数据验证
# 展示类型系统在"工具被调用前"就能拦截错误参数
# ============================================================
def demo_validation():
    print("=" * 60)
    print("【演示 1】Pydantic 自动数据验证")
    print("=" * 60)

    # 合法输入
    print("\n合法输入：")
    valid = SearchProductsInput(
        query="蓝牙耳机",
        category="electronics",
        price_max=500,
        sort_by="price_asc",
    )
    print(f"  解析成功: {valid.model_dump()}")

    # 非法 category
    print("\n非法输入（category='appliances' 不在 Literal 允许值内）：")
    try:
        SearchProductsInput(query="洗衣机", category="appliances")
    except ValidationError as e:
        print(f"  自动拦截: {e.errors()[0]['msg']}")

    # 非法 sort_by
    print("\n非法输入（sort_by='newest' 不在 Literal 允许值内）：")
    try:
        SearchProductsInput(query="耳机", category="electronics", sort_by="newest")
    except ValidationError as e:
        print(f"  自动拦截: {e.errors()[0]['msg']}")

    # 查看自动生成的 JSON Schema（LLM 实际看到的参数说明）
    print("\nJSON Schema（这就是 LLM 调用工具时看到的参数说明）：")
    schema = SearchProductsInput.model_json_schema()
    print(json.dumps(schema, ensure_ascii=False, indent=2))


# ============================================================
# 演示 2：Agent 使用类型安全工具
# 观察 LLM 是否能根据 Schema 正确传参
# ============================================================
def demo_agent():
    print("\n" + "=" * 60)
    print("【演示 2】Agent 使用类型安全工具")
    print("=" * 60)

    model = ChatDeepSeek(
        model="deepseek-chat",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        temperature=0,
    )

    prompt = PromptTemplate.from_template(
        """你是一个购物助手，帮助用户找到合适的商品。

可用工具：
{tools}

工具名称：{tool_names}

执行规则：严格按照 Thought/Action/Action Input/Observation/Final Answer 格式。

Question: {input}
Thought:{agent_scratchpad}"""
    )

    agent = create_react_agent(llm=model, tools=[search_products], prompt=prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=[search_products],
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
    )

    # 观察 LLM 是否正确推断出：sort_by="rating"，category="clothing"
    result = executor.invoke({"input": "我想买评分最高的跑步鞋，预算 800 元以内"})
    print(f"\n最终回答：{result['output']}")


if __name__ == "__main__":
    demo_validation()
    print("\n\n")
    demo_agent()

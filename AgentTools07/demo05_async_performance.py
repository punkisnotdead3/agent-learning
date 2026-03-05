import os
import sys
import time
import asyncio
from typing import Optional

sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# demo05_async_performance.py -- Tool 开发最佳实践：性能优化与上下文管理
#
# 【核心知识点】
# 1. 异步并行工具调用：多个独立 IO 操作从串行改为并行，大幅提速
#    串行：task_a(2s) + task_b(2s) + task_c(2s) = 6s
#    并行：asyncio.gather(a, b, c) ≈ 2s
#
# 2. max_results 限制返回数量：防止 LLM Context 被海量数据撑爆
#    - 设置合理默认值（3条），并用 le=5 限制上限
#
# 3. 摘要 vs 全文分离：先返回摘要，按需获取全文
#    - search_knowledge_base() → 只返回前 200 字 + full_text_available 标志
#    - get_document_full_text() → 仅在摘要不够时才调用
#
# 【为什么摘要分离很重要？】
#   假设每篇文档 5000 字，搜索返回 5 篇 = 25000 字进入 context
#   如果只返回摘要（200字/篇）= 1000 字，节省 96% 的 token
#   LLM 通常看摘要就能回答 80% 的问题，不需要全文
# ============================================================

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate


# ============================================================
# 模拟知识库（文档内容故意设计得很长，突出摘要分离的价值）
# ============================================================
KNOWLEDGE_BASE = {
    "doc001": {
        "title": "Python 在 AI 领域的应用",
        "content": (
            "Python 是 AI 和机器学习领域的首选语言，生态无可替代。"
            "主要框架包括 PyTorch、TensorFlow、scikit-learn、Hugging Face Transformers。"
            "Python 语法简洁，与数学公式天然契合，研究人员可以快速将论文思路转化为代码。"
            "Jupyter Notebook 成为科研标配，让数据探索和可视化极为方便。"
        ) * 20,  # 故意重复让内容变长，模拟真实长文档
        "tags": ["python", "ai", "machine-learning"],
        "score": 0.95,
    },
    "doc002": {
        "title": "Python 性能优化指南",
        "content": (
            "Python 性能优化常用方案：NumPy 向量化计算、Cython 编译扩展、"
            "asyncio 异步 IO、多进程 multiprocessing 绕过 GIL、"
            "使用 PyPy 替代 CPython、profile 分析热点代码。"
            "大多数 Python 性能问题源于 IO 等待，而非 CPU 计算。"
        ) * 15,
        "tags": ["python", "performance", "optimization"],
        "score": 0.88,
    },
    "doc003": {
        "title": "FastAPI 快速入门",
        "content": (
            "FastAPI 基于 Python 3.7+ 的类型注解，自动生成 OpenAPI 文档。"
            "性能已接近 Node.js 和 Go，适合构建高并发 API 服务。"
            "内置支持 async/await，依赖注入系统清晰，测试友好。"
        ) * 18,
        "tags": ["python", "web", "fastapi", "api"],
        "score": 0.82,
    },
    "doc004": {
        "title": "数据科学工具链",
        "content": (
            "数据科学标准工具链：Jupyter Notebook（探索）、Pandas（数据处理）、"
            "Matplotlib/Seaborn（可视化）、Scikit-learn（传统ML）、"
            "PyTorch/TensorFlow（深度学习）。"
        ) * 12,
        "tags": ["data-science", "python", "pandas", "visualization"],
        "score": 0.79,
    },
    "doc005": {
        "title": "Python vs Go 语言对比",
        "content": (
            "Python 优势：开发效率高、AI 生态无可替代、语法简洁。"
            "Go 优势：编译型、高并发、内存占用低、部署简单（单一二进制）。"
            "选型建议：AI/数据用 Python，高并发微服务考虑 Go。"
        ) * 10,
        "tags": ["python", "golang", "comparison"],
        "score": 0.75,
    },
}


# ============================================================
# 工具 1：分页+摘要优先的知识库搜索
#
# 关键设计：
#   - max_results 有默认值(3)且有上限(5)，防止返回过多内容
#   - 只返回摘要（200字），不返回全文
#   - full_text_available=True 提示 Agent 有全文可获取
# ============================================================
class SearchKBInput(BaseModel):
    query: str = Field(
        description="搜索查询，描述想找什么内容，例如：'Python 性能优化方法'"
    )
    max_results: int = Field(
        default=3,
        ge=1,
        le=5,
        description=(
            "返回结果数量，默认 3 条已足够。"
            "只有用户明确要求'更多结果'时才增加，最多 5 条"
        ),
    )
    tag_filter: Optional[str] = Field(
        None,
        description="按标签过滤，如 'python'/'ai'/'web'。不确定时留空"
    )


@tool(args_schema=SearchKBInput)
def search_knowledge_base(
    query: str,
    max_results: int = 3,
    tag_filter: Optional[str] = None,
) -> list:
    """
    搜索知识库，返回相关文档的摘要信息（非全文）。

    重要：此工具只返回每篇文档的前 200 字摘要，不返回完整内容。
    如果摘要不够用，再对最相关的 1-2 篇调用 get_document_full_text()。

    使用建议：
    - 先用默认 max_results=3 搜索，通常足够
    - 拿到结果后先判断哪篇最相关，再按需获取全文
    - 80% 的问题看摘要就能回答，不需要全文
    """
    time.sleep(0.05)  # 模拟查询延迟

    results = []
    for doc_id, doc in KNOWLEDGE_BASE.items():
        if tag_filter and tag_filter not in doc["tags"]:
            continue
        # 简单关键词匹配
        query_words = query.replace("，", " ").replace(",", " ").split()
        if any(word in doc["title"] + doc["content"] for word in query_words):
            results.append((doc_id, doc))

    results.sort(key=lambda x: x[1]["score"], reverse=True)
    results = results[:max_results]

    # 只返回摘要，不返回全文（核心！）
    return [
        {
            "doc_id": doc_id,
            "title": doc["title"],
            "summary": doc["content"][:200] + "...",   # 只返回前 200 字
            "tags": doc["tags"],
            "relevance_score": doc["score"],
            "full_text_available": True,                # 提示可以获取全文
            "full_text_length": len(doc["content"]),   # 让 LLM 知道全文有多长
        }
        for doc_id, doc in results
    ]


@tool
def get_document_full_text(doc_id: str) -> str:
    """
    获取指定文档的完整内容。

    使用时机（只在以下情况调用，避免不必要的 token 消耗）：
    - search_knowledge_base 返回的摘要不足以回答问题
    - 用户明确要求"完整内容"或"详细介绍"
    - 需要引用文档中的具体细节或数据

    注意：完整内容较长，会占用较多 context，非必要不调用。

    参数：doc_id - 文档 ID，从 search_knowledge_base 结果的 doc_id 字段获取
    """
    if doc_id not in KNOWLEDGE_BASE:
        return f"文档 '{doc_id}' 不存在"
    doc = KNOWLEDGE_BASE[doc_id]
    # 实际使用时返回完整内容，这里截断展示
    preview = doc["content"][:500]
    return f"【{doc['title']}】完整内容（共 {len(doc['content'])} 字）：\n{preview}..."


# ============================================================
# 工具 2：多源并行搜索（内部使用 asyncio.gather）
#
# 关键设计：
#   - 对 LLM 来说是普通的同步工具（一次调用）
#   - 内部透明地并行查询多个来源，结果聚合后返回
#   - LLM 不需要知道并行细节，只需要知道"一次调用拿多源结果"
# ============================================================
async def _query_source(source: str, query: str, delay: float) -> dict:
    """模拟带网络延迟的异步数据源查询"""
    await asyncio.sleep(delay)
    return {
        "source": source,
        "results": [
            f"[{source}] 关于 '{query}' 的发现：示例结论 A",
            f"[{source}] 关于 '{query}' 的补充：示例数据 B",
        ],
    }


@tool
def search_multi_source(query: str) -> str:
    """
    同时从知识库、技术新闻、开发者论坛三个来源并行搜索，速度远快于逐一查询。

    适合需要多角度信息时使用（如：既要官方文档又要社区实践经验）。
    内部使用 asyncio.gather 并发查询，耗时约等于单个来源的查询时间。

    参数：query - 搜索查询，描述想了解什么
    """
    print(f"\n  [并行搜索] 同时查询 3 个来源：知识库、技术新闻、开发者论坛")
    start = time.time()

    async def gather():
        return await asyncio.gather(
            _query_source("knowledge_base", query, delay=1.5),
            _query_source("tech_news",      query, delay=1.2),
            _query_source("dev_forum",      query, delay=1.8),
        )

    all_results = asyncio.run(gather())
    elapsed = time.time() - start
    print(f"  [并行搜索] 3 个来源完成，耗时 {elapsed:.2f}s（串行约需 {1.5+1.2+1.8:.1f}s）")

    lines = [f"多源搜索结果（并行耗时 {elapsed:.2f}s）："]
    for r in all_results:
        lines.append(f"\n  [{r['source']}]")
        for item in r["results"]:
            lines.append(f"    - {item}")
    return "\n".join(lines)


# ============================================================
# 演示 1：串行 vs 并行性能对比（不需要 LLM，纯代码演示）
# ============================================================
def demo_performance_comparison():
    print("=" * 60)
    print("【演示 1】串行 vs 并行工具调用性能对比")
    print("=" * 60)

    async def serial():
        r1 = await _query_source("source_A", "Python", 1.5)
        r2 = await _query_source("source_B", "Python", 1.2)
        r3 = await _query_source("source_C", "Python", 1.8)
        return [r1, r2, r3]

    async def parallel():
        return await asyncio.gather(
            _query_source("source_A", "Python", 1.5),
            _query_source("source_B", "Python", 1.2),
            _query_source("source_C", "Python", 1.8),
        )

    print("\n串行执行（逐一等待）：")
    t = time.time()
    asyncio.run(serial())
    serial_time = time.time() - t
    print(f"  串行耗时：{serial_time:.2f}s")

    print("\n并行执行（asyncio.gather）：")
    t = time.time()
    asyncio.run(parallel())
    parallel_time = time.time() - t
    print(f"  并行耗时：{parallel_time:.2f}s")

    print(f"\n速度提升：{serial_time / parallel_time:.1f}x（理论上限：3x）")


# ============================================================
# 演示 2：摘要分离策略（不需要 LLM）
# ============================================================
def demo_summary_separation():
    print("\n" + "=" * 60)
    print("【演示 2】摘要分离策略对比 token 消耗")
    print("=" * 60)

    # 如果返回全文
    full_text_total = sum(len(doc["content"]) for doc in KNOWLEDGE_BASE.values())
    print(f"\n如果搜索返回全文（5篇）：约 {full_text_total:,} 字进入 context")

    # 如果只返回摘要
    summary_total = sum(min(len(doc["content"]), 200) for doc in KNOWLEDGE_BASE.values())
    print(f"如果只返回摘要（200字/篇）：约 {summary_total:,} 字进入 context")
    print(f"节省 token：{(1 - summary_total / full_text_total) * 100:.0f}%")
    print(f"\n80% 的问题看摘要就能回答，剩余 20% 按需获取 1-2 篇全文即可")


# ============================================================
# 演示 3：Agent 使用摘要优先策略
# ============================================================
def demo_agent():
    print("\n" + "=" * 60)
    print("【演示 3】Agent 摘要优先 + 按需获取全文")
    print("=" * 60)

    model = ChatDeepSeek(
        model="deepseek-chat",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        temperature=0,
    )

    prompt = PromptTemplate.from_template(
        """你是一个知识库检索助手。

性能优化原则（必须遵守）：
1. 先用 search_knowledge_base 获取摘要（max_results 默认 3 即可）
2. 判断哪篇最相关后，只对最相关的 1 篇调用 get_document_full_text
3. 如果摘要已能回答问题，不需要获取全文
4. 需要多来源信息时，用 search_multi_source（内部已并行处理）

可用工具：
{tools}

工具名称：{tool_names}

规则：按照 Thought/Action/Action Input/Observation/Final Answer 格式。

Question: {input}
Thought:{agent_scratchpad}"""
    )

    tools_list = [search_knowledge_base, get_document_full_text, search_multi_source]
    agent = create_react_agent(llm=model, tools=tools_list, prompt=prompt)
    executor = AgentExecutor(
        agent=agent, tools=tools_list,
        verbose=True, max_iterations=8, handle_parsing_errors=True,
    )

    result = executor.invoke({
        "input": "帮我搜索 Python 相关的技术文档，简单总结有哪些内容"
    })
    print(f"\n最终结果：{result['output']}")


if __name__ == "__main__":
    demo_performance_comparison()
    demo_summary_separation()
    print("\n\n")
    demo_agent()

"""
04_semantic_search.py
=====================
目标：读取含 Embedding 向量的 CSV 文件，对用户输入的自然语言查询
      进行语义搜索，返回语义最相似的 Top-K 条评论。

核心原理：
  1. 将所有评论的向量加载到内存，拼成一个 (N, 768) 的矩阵
  2. 对查询文本同样生成一个 768 维向量
  3. 用矩阵乘法一次性计算查询向量与所有行的余弦相似度
  4. 取相似度最高的 K 行输出

为什么不需要向量数据库？
  1000 条 × 768 维 ≈ 6 MB 内存，numpy 矩阵运算在毫秒级完成，
  完全不需要引入额外的基础设施。

输入：data/reviews_with_embeddings.csv（由 03_embed_and_store.py 生成）

运行方式：
  python 04_semantic_search.py
"""

import json
import sys
import numpy as np
import pandas as pd
from openai import OpenAI

# ── 配置参数 ──────────────────────────────────────────────────────────────────

INPUT_PATH = "data/reviews_with_embeddings.csv"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
EMBED_MODEL     = "nomic-embed-text"

# 默认返回最相似的前 K 条结果
DEFAULT_TOP_K = 5

# 打印评论正文时，截取的最大字符数（避免输出太长）
PREVIEW_CHARS = 200


# ── 初始化客户端 ──────────────────────────────────────────────────────────────

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
)


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_data(path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """
    加载 CSV 文件，解析 embedding 列，返回 DataFrame 和向量矩阵。

    返回：
        df            : 原始 DataFrame（含 Id, Score, content 等列）
        embed_matrix  : shape (N, D) 的 numpy 浮点矩阵，D 为向量维度（768）
    """
    print(f"正在加载 {path} ...")
    df = pd.read_csv(path)

    if "embedding" not in df.columns:
        raise ValueError("CSV 中缺少 embedding 列，请先运行 03_embed_and_store.py")

    # 将每一行的 JSON 字符串还原为 list[float]，再堆叠成二维 numpy 矩阵
    # np.stack：把 N 个长度为 D 的列表竖向叠放，得到 (N, D) 矩阵
    vectors = [json.loads(row) for row in df["embedding"]]
    embed_matrix = np.stack(vectors).astype(np.float32)   # float32 节省内存、加速计算

    print(f"加载完成：{len(df)} 条评论，向量矩阵形状 {embed_matrix.shape}")
    return df, embed_matrix


# ── 向量归一化 ────────────────────────────────────────────────────────────────

def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    对矩阵的每一行做 L2 归一化（使每行向量的模为 1）。

    为什么要归一化？
        归一化后，余弦相似度可以简化为点积：
            cos(a, b) = a_norm · b_norm
        这样对整个矩阵只需一次矩阵乘法，效率极高。

    参数：
        matrix: shape (N, D)

    返回：
        shape (N, D)，每行的 L2 范数为 1
    """
    # keepdims=True：保持维度 (N, 1)，方便广播除法
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # 加上极小值防止除以零（理论上不会发生，但作为保护）
    return matrix / (norms + 1e-10)


# ── 语义搜索 ──────────────────────────────────────────────────────────────────

def semantic_search(
    query: str,
    df: pd.DataFrame,
    norm_matrix: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
    score_filter: int | None = None,
) -> pd.DataFrame:
    """
    对自然语言查询执行语义搜索，返回最相似的 Top-K 条评论。

    参数：
        query        : 用户输入的查询字符串
        df           : 评论 DataFrame
        norm_matrix  : 已归一化的向量矩阵，shape (N, 768)
        top_k        : 返回结果数量
        score_filter : 若不为 None，只在指定评分（1-5）的评论中搜索

    返回：
        包含 [score, similarity, content] 列的 DataFrame，按相似度降序排列
    """
    # 步骤 A：生成查询向量
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=[query],
    )
    query_vec = np.array(response.data[0].embedding, dtype=np.float32)

    # 步骤 B：归一化查询向量（与矩阵归一化方式相同）
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)

    # 步骤 C：计算相似度
    # norm_matrix @ query_norm：矩阵-向量乘法
    # 结果 scores 形状为 (N,)，scores[i] 就是第 i 条评论与查询的余弦相似度
    scores = norm_matrix @ query_norm

    # 步骤 D：（可选）按评分过滤
    # 如果指定了 score_filter，将不符合评分的行的相似度设为 -1（排在最后）
    if score_filter is not None:
        mask = df["Score"].values != score_filter   # 不符合评分的位置为 True
        scores[mask] = -1.0                         # 强制置为最低分，不会被选中

    # 步骤 E：取 Top-K
    # np.argsort 返回从小到大的排序索引，[::-1] 反转为从大到小
    top_indices = np.argsort(scores)[::-1][:top_k]

    # 步骤 F：构造结果 DataFrame
    results = df.iloc[top_indices].copy()
    results["similarity"] = scores[top_indices]

    return results[["Score", "similarity", "content"]].reset_index(drop=True)


# ── 结果打印 ──────────────────────────────────────────────────────────────────

def print_results(query: str, results: pd.DataFrame) -> None:
    """格式化打印搜索结果。"""
    print(f"\n{'─' * 60}")
    print(f"查询：{query!r}")
    print(f"Top-{len(results)} 相似评论：")
    print(f"{'─' * 60}")

    for rank, (_, row) in enumerate(results.iterrows(), 1):
        # 将星级数字转成星号符号，更直观
        stars = "★" * int(row["Score"]) + "☆" * (5 - int(row["Score"]))
        preview = row["content"][:PREVIEW_CHARS]
        if len(row["content"]) > PREVIEW_CHARS:
            preview += "..."

        print(f"\n#{rank}  相似度: {row['similarity']:.4f}  评分: {stars}")
        print(f"    {preview}")

    print(f"\n{'─' * 60}\n")


# ── 预置演示查询 ──────────────────────────────────────────────────────────────

DEMO_QUERIES = [
    # 查询语义：咖啡好喝
    "great chocolate taste but too sweet",
    # 查询语义：宠物喜欢这个食品
    "my dog loves this food",
    # 查询语义：差评，浪费钱
    "terrible product waste of money",
    # 查询语义：咖啡风味绝佳
    "best coffee I ever tried, amazing aroma",
]


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    # ── 加载数据 ─────────────────────────────────────────────────────────────
    try:
        df, embed_matrix = load_data(INPUT_PATH)
    except FileNotFoundError:
        print(f"[错误] 找不到文件：{INPUT_PATH}")
        print("请先运行 03_embed_and_store.py 生成含向量的 CSV 文件。")
        sys.exit(1)

    # ── 预处理：对整个矩阵做一次归一化，后续搜索无需重复计算 ─────────────────
    print("正在对向量矩阵做 L2 归一化（只需一次）...")
    norm_matrix = normalize_matrix(embed_matrix)
    print("归一化完成。\n")

    # ── 模式选择 ─────────────────────────────────────────────────────────────
    print("请选择运行模式：")
    print("  1. 运行预置演示查询（4 个示例）")
    print("  2. 进入交互式搜索（自己输入查询语句）")

    choice = input("输入选项（1 或 2）：").strip()

    if choice == "1":
        # ── 演示模式：依次运行预置查询 ─────────────────────────────────────
        print(f"\n运行 {len(DEMO_QUERIES)} 个预置查询...\n")
        for query in DEMO_QUERIES:
            results = semantic_search(query, df, norm_matrix, top_k=DEFAULT_TOP_K)
            print_results(query, results)

    elif choice == "2":
        # ── 交互模式：循环等待用户输入 ─────────────────────────────────────
        print("\n进入交互式搜索模式（输入 'quit' 或 'q' 退出）\n")
        print("提示：可以在查询前加 [N星] 前缀来过滤评分，例如：")
        print("  [5星] best coffee ever")
        print("  [1星] worst product\n")

        while True:
            query = input("请输入查询：").strip()

            if query.lower() in ("quit", "q", "exit", ""):
                print("退出搜索。")
                break

            # 解析评分过滤前缀，例如 "[5星] best coffee"
            score_filter = None
            import re
            match = re.match(r"^\[(\d)星\]\s*", query)
            if match:
                score_filter = int(match.group(1))
                query = query[match.end():]    # 去掉前缀，保留真正的查询内容
                print(f"（已启用评分过滤：只在 {score_filter} 星评论中搜索）")

            if not query:
                print("查询内容为空，请重新输入。")
                continue

            results = semantic_search(
                query, df, norm_matrix,
                top_k=DEFAULT_TOP_K,
                score_filter=score_filter,
            )
            print_results(query, results)

    else:
        print("无效选项，退出。")


if __name__ == "__main__":
    main()

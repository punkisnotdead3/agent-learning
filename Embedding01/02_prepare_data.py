"""
02_prepare_data.py
==================
目标：加载亚马逊美食评论原始数据集，进行清洗和裁剪，
      输出一个干净的小规模子集，供后续 Embedding 实验使用。

数据集说明：
  - 来源：Kaggle - Amazon Fine Food Reviews
  - 下载地址：https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
  - 文件名：Reviews.csv（约 290 MB，568,454 条评论）
  - 原始列：Id, ProductId, UserId, ProfileName, HelpfulnessNumerator,
            HelpfulnessDenominator, Score, Time, Summary, Text

运行前提：
  - 安装 kagglehub（pip install kagglehub）
  - 若是首次运行，kagglehub 会自动下载数据集到本地缓存，
    并将 Reviews.csv 复制到 data/ 目录。

运行方式：
  python 02_prepare_data.py
"""

import os
import shutil
import pandas as pd

# ── 配置参数 ──────────────────────────────────────────────────────────────────

# 输入文件路径（原始数据集）
INPUT_PATH = "data/Reviews.csv"

# 输出文件路径（清洗后的子集）
OUTPUT_PATH = "data/reviews_clean.csv"

# 取前 N 条作为学习子集。
# 1000 条在普通电脑上生成 Embedding 约需 1-3 分钟，适合入门学习。
# 如果想覆盖更多数据，可以调大这个值（5000、10000 等）。
SAMPLE_SIZE = 1000


def ensure_dataset(dest: str) -> None:
    """
    确保 Reviews.csv 存在于 dest 路径。若不存在，则通过 kagglehub 自动下载。

    kagglehub 工作原理：
      - 首次调用时从 Kaggle 下载数据集压缩包并解压到本地缓存目录
        （通常为 ~/.cache/kagglehub/datasets/snap/amazon-fine-food-reviews/）
      - 再次调用时直接使用缓存，不会重复下载
      - 不需要手动登录网站，但首次运行时会通过浏览器或环境变量完成认证
    """
    if os.path.exists(dest):
        print(f"[跳过下载] {dest} 已存在")
        return

    print("未找到 Reviews.csv，正在通过 kagglehub 自动下载...")
    print("（首次下载约 290 MB，请耐心等待）\n")

    try:
        import kagglehub
    except ImportError:
        print("[错误] 未安装 kagglehub，请先执行：pip install kagglehub")
        raise

    # dataset_download 返回数据集缓存目录的路径
    cache_path = kagglehub.dataset_download("snap/amazon-fine-food-reviews")
    print(f"下载完成，缓存目录：{cache_path}")

    # 在缓存目录中递归查找 Reviews.csv
    found = None
    for root, _, files in os.walk(cache_path):
        if "Reviews.csv" in files:
            found = os.path.join(root, "Reviews.csv")
            break

    if not found:
        raise FileNotFoundError(
            f"缓存目录 {cache_path} 中未找到 Reviews.csv，"
            "请检查 kagglehub 下载是否完整。"
        )

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(found, dest)
    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"已复制到：{dest}（{size_mb:.1f} MB）\n")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    # ── 步骤 1：确保数据集存在（自动下载） ───────────────────────────────────
    ensure_dataset(INPUT_PATH)

    # ── 步骤 2：加载原始数据 ──────────────────────────────────────────────────
    print(f"正在读取 {INPUT_PATH} ...")
    # low_memory=False：让 pandas 一次性推断列类型，避免混合类型警告
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    print(f"原始数据：{len(df):,} 条评论，{df.shape[1]} 列")
    print(f"列名：{list(df.columns)}\n")

    # ── 步骤 3：只保留需要的列 ────────────────────────────────────────────────
    # 我们只需要 Id、ProductId、Score（评分）、Summary（标题）、Text（正文）
    df = df[["Id", "ProductId", "UserId", "Score", "Summary", "Text"]]

    # ── 步骤 4：去除重复评论 ──────────────────────────────────────────────────
    # 同一个用户对同一个商品可能多次评论，保留最后一条（Time 最晚的）。
    # 原始数据已按 Time 升序排列，所以 keep="last" 即保留最新的。
    before = len(df)
    df = df.drop_duplicates(subset=["ProductId", "UserId"], keep="last")
    after = len(df)
    print(f"步骤 4 去重：删除 {before - after:,} 条重复评论，剩余 {after:,} 条")

    # ── 步骤 5：过滤空文本 ────────────────────────────────────────────────────
    # Summary 或 Text 为空（NaN 或纯空白）的行，无法生成有意义的 Embedding
    before = len(df)
    df = df.dropna(subset=["Summary", "Text"])          # 删除 NaN
    df = df[df["Summary"].str.strip() != ""]            # 删除纯空白
    df = df[df["Text"].str.strip() != ""]
    after = len(df)
    print(f"步骤 5 过滤空文本：删除 {before - after:,} 条，剩余 {after:,} 条")

    # ── 步骤 6：合并 Summary 和 Text 为单一 content 字段 ──────────────────────
    # Embedding 模型接收单段文本，我们把标题和正文拼接，让语义信息更完整。
    # 格式："{Summary}. {Text}"
    # 例如："Great coffee. This is the best coffee I have ever tasted..."
    df["content"] = df["Summary"].str.strip() + ". " + df["Text"].str.strip()

    # ── 步骤 7：文本截断（防止超出模型 Token 限制） ───────────────────────────
    # nomic-embed-text 最大支持 8192 tokens，大约对应 6000 个英文字符。
    # 超长文本会被模型自动截断，但提前截断可以节省推理时间。
    MAX_CHARS = 2000    # 保守值，足够覆盖绝大多数评论的关键信息
    df["content"] = df["content"].str[:MAX_CHARS]
    long_count = (df["content"].str.len() == MAX_CHARS).sum()
    if long_count > 0:
        print(f"步骤 7 截断：{long_count} 条文本超过 {MAX_CHARS} 字符，已截断")

    # ── 步骤 8：取前 N 条子集 ─────────────────────────────────────────────────
    # reset_index：重置行索引，使其从 0 开始连续，drop=True 不把旧索引保留为列
    df = df.head(SAMPLE_SIZE).reset_index(drop=True)
    print(f"步骤 8 取子集：保留前 {len(df):,} 条")

    # ── 步骤 9：查看评分分布 ──────────────────────────────────────────────────
    # Score 是 1-5 的整数，理想情况下各分值都有覆盖
    print("\n评分分布（Score 列）：")
    score_dist = df["Score"].value_counts().sort_index()
    for score, count in score_dist.items():
        bar = "█" * (count // 5)    # 简单的 ASCII 条形图，每 5 条评论画一格
        print(f"  {score} 星：{count:4d} 条  {bar}")

    # ── 步骤 10：保存结果 ─────────────────────────────────────────────────────
    # 只输出后续需要的列
    output_df = df[["Id", "ProductId", "Score", "content"]]
    os.makedirs("data", exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\n清洗完成，已保存至：{OUTPUT_PATH}")
    print(f"输出列：{list(output_df.columns)}")
    print(f"文件大小：{os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")

    # 打印前 2 条，方便确认内容格式
    print("\n前 2 条样本预览：")
    for i, row in output_df.head(2).iterrows():
        print(f"\n  [第 {i} 条] Score={row['Score']} | Id={row['Id']}")
        print(f"  content: {row['content'][:120]}...")


if __name__ == "__main__":
    main()

"""
01_embedding_basics.py
======================
目标：理解 Embedding 的基本概念，验证 Ollama 接口连通性。

运行前提：
  1. Ollama 已安装并运行（在命令行执行 `ollama serve`）
  2. 已拉取模型（`ollama pull nomic-embed-text`）
  3. 已安装依赖（`pip install openai numpy matplotlib`）

运行方式：
  python 01_embedding_basics.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from openai import OpenAI

# Windows 终端默认 GBK 编码不支持部分 Unicode 字符，强制设为 UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ── 配置 ────────────────────────────────────────────────────────────────────

# Ollama 在本地启动后，会在 11434 端口暴露一个与 OpenAI 完全兼容的 HTTP 接口。
# 只需把 base_url 指向本地，api_key 填任意非空字符串即可（Ollama 不做鉴权）。
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",           # 任意字符串，Ollama 不校验
)

# 使用的 Embedding 模型名称，需与 ollama pull 时的名称一致
EMBED_MODEL = "nomic-embed-text"


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    """
    调用 Ollama（OpenAI 兼容接口）为单条文本生成 Embedding 向量。

    参数：
        text: 需要向量化的文本字符串

    返回：
        一个 float 列表，长度为模型的向量维度（nomic-embed-text 为 768）

    底层原理：
        文本经过 Transformer 模型后，取 [CLS] token 或对所有 token 做平均池化，
        得到一个固定长度的浮点数向量，代表文本在语义空间中的位置。
    """
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text],           # 接口接受列表，这里传入单元素列表
    )
    # response.data 是一个列表，每个元素对应输入列表中的一条文本
    # .embedding 属性就是向量本体，类型为 list[float]
    return response.data[0].embedding


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    计算两个向量的余弦相似度。

    公式：
        cos(a, b) = (a · b) / (||a|| × ||b||)

    值域：[-1, 1]
        - 1.0  表示方向完全相同（语义极度相似）
        - 0.0  表示互相垂直（语义无关）
        - -1.0 表示方向完全相反（语义相反，实践中极少出现）

    为什么用余弦而不是欧氏距离？
        Embedding 向量的长度（模）可能受文本长短影响，
        余弦相似度只关注方向，对长度不敏感，更适合语义比较。
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    # np.dot：点积（内积）
    # np.linalg.norm：计算向量的 L2 范数（即欧氏长度）
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ── 实验一：验证接口连通性 ───────────────────────────────────────────────────

def experiment_1_connectivity():
    """实验一：用最简单的文本验证 Ollama 接口是否正常工作。"""
    print("=" * 60)
    print("实验一：验证接口连通性")
    print("=" * 60)

    test_text = "Hello, Embedding!"
    print(f"输入文本：{test_text!r}")

    vec = get_embedding(test_text)

    print(f"向量维度：{len(vec)}")                    # 应为 768
    print(f"前 5 个值：{[round(v, 4) for v in vec[:5]]}")
    print("[OK] 接口连通正常\n")


# ── 实验二：相似 vs 不相关句子的相似度对比 ──────────────────────────────────

def experiment_2_similarity_comparison():
    """
    实验二：通过具体数值，直观感受"语义相似 → 向量距离近"。

    我们准备三组句子：
        - 锚句（anchor）：作为基准
        - 相似句（similar）：与锚句语义接近
        - 不相关句（unrelated）：与锚句语义无关
    """
    print("=" * 60)
    print("实验二：相似句子 vs 不相关句子的余弦相似度")
    print("=" * 60)

    # 三组实验，每组包含：锚句、相似句、不相关句
    groups = [
        {
            "anchor":    "This coffee has a rich and smooth flavor.",
            "similar":   "The coffee tastes great, very balanced and aromatic.",
            "unrelated": "My dog loves to play fetch in the park.",
        },
        {
            "anchor":    "The product arrived damaged and the packaging was broken.",
            "similar":   "Item was received in poor condition, box was crushed.",
            "unrelated": "Sunny weather is perfect for a picnic.",
        },
    ]

    for i, group in enumerate(groups, 1):
        print(f"\n--- 第 {i} 组 ---")
        print(f"锚   句: {group['anchor']}")
        print(f"相似句: {group['similar']}")
        print(f"无关句: {group['unrelated']}")

        # 分别生成三个向量
        vec_anchor    = get_embedding(group["anchor"])
        vec_similar   = get_embedding(group["similar"])
        vec_unrelated = get_embedding(group["unrelated"])

        # 计算相似度
        sim_similar   = cosine_similarity(vec_anchor, vec_similar)
        sim_unrelated = cosine_similarity(vec_anchor, vec_unrelated)

        print(f"\n  锚句 <-> 相似句  余弦相似度: {sim_similar:.4f}  {'[高]' if sim_similar > 0.7 else ''}")
        print(f"  锚句 <-> 无关句  余弦相似度: {sim_unrelated:.4f}  {'（预期明显低于上面）' if sim_unrelated < sim_similar else ''}")

    print()


# ── 实验三：多句子相似度热力图 ───────────────────────────────────────────────

def experiment_3_heatmap():
    """
    实验三：对一组句子两两计算相似度，用热力图可视化。

    热力图解读：
        - 颜色越深（越接近黄色/白色）：两句话语义越相似
        - 对角线永远是 1.0（自身与自身完全相同）
        - 同主题句子之间应该形成"高亮块"
    """
    print("=" * 60)
    print("实验三：多句子相似度热力图")
    print("=" * 60)

    sentences = [
        # 咖啡主题
        "This coffee is delicious and has a great aroma.",
        "I love the rich flavor of this coffee blend.",
        "Best coffee I have ever tasted, smooth and bold.",
        # 差评主题
        "Terrible product, completely waste of money.",
        "I am very disappointed, this is not what I expected.",
        "Poor quality, will never buy again.",
        # 宠物食品主题
        "My dog absolutely loves this food.",
        "Great pet food, my cat eats it every day.",
    ]

    n = len(sentences)
    print(f"共 {n} 个句子，正在生成 Embedding（需要几秒钟）...")

    # 生成所有句子的向量
    embeddings = [get_embedding(s) for s in sentences]

    # 构建 N×N 相似度矩阵
    # sim_matrix[i][j] = sentences[i] 与 sentences[j] 的余弦相似度
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i][j] = cosine_similarity(embeddings[i], embeddings[j])

    # ── 绘制热力图 ──────────────────────────────────────────────────────────
    # 使用支持中文显示的字体（若系统无 SimHei 则退回英文标签）
    try:
        matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    # 缩短标签，避免太长
    labels = [s[:30] + "..." if len(s) > 30 else s for s in sentences]

    fig, ax = plt.subplots(figsize=(10, 8))

    # imshow 将矩阵渲染为颜色图，vmin/vmax 固定色条范围
    im = ax.imshow(sim_matrix, cmap="YlOrRd", vmin=0, vmax=1)

    # 设置坐标轴刻度和标签
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    # 在每个格子中显示具体数值
    for i in range(n):
        for j in range(n):
            val = sim_matrix[i][j]
            # 根据背景色深浅选择文字颜色，保证可读性
            text_color = "white" if val > 0.75 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=text_color)

    # 添加颜色条（图例）
    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    ax.set_title("句子间余弦相似度热力图\n（同主题句子应聚集成高亮块）", fontsize=12)

    plt.tight_layout()

    # 保存到文件（避免需要 GUI 环境）
    output_path = "data/heatmap.png"
    import os
    os.makedirs("data", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"热力图已保存至：{output_path}")

    # 如果有显示环境，也弹出窗口
    try:
        plt.show()
    except Exception:
        pass

    print()


# ── 主程序入口 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n【Embedding 基础概念验证】")
    print(f"使用模型：{EMBED_MODEL}")
    print(f"Ollama 地址：http://localhost:11434/v1\n")

    experiment_1_connectivity()
    experiment_2_similarity_comparison()
    experiment_3_heatmap()

    print("=" * 60)
    print("全部实验完成！")
    print("建议观察：")
    print("  1. 相似句子的余弦相似度是否明显高于不相关句子？")
    print("  2. 热力图中同主题的句子是否形成了高亮的色块？")
    print("=" * 60)

"""
03_embed_and_store.py
=====================
目标：读取清洗后的评论数据，批量调用 Ollama Embedding 接口，
      将生成的向量保存到 CSV 文件中。

输入：data/reviews_clean.csv（由 02_prepare_data.py 生成）
输出：data/reviews_with_embeddings.csv
      新增列 embedding：JSON 字符串，存储 768 维浮点向量

运行方式：
  python 03_embed_and_store.py
"""

import os
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ── 配置 ──────────────────────────────────────────────────────────────────────

INPUT_PATH  = "data/reviews_clean.csv"
OUTPUT_PATH = "data/reviews_with_embeddings.csv"
EMBED_MODEL = "nomic-embed-text"

# 每次批量发送给 Ollama 的文本数量
# 批量请求比逐条请求效率更高，16 是经验值
BATCH_SIZE = 16

# Ollama 本地服务，与 OpenAI SDK 完全兼容，只需改 base_url
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",   # Ollama 不做鉴权，填任意字符串即可
)

# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    # 读取清洗后的数据
    df = pd.read_csv(INPUT_PATH)
    print(f"已加载 {len(df)} 条评论，开始生成 Embedding...")

    texts = df["content"].tolist()
    all_embeddings = []

    # 分批处理，tqdm 显示进度条
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="生成 Embedding", unit="批"):
        batch = texts[i : i + BATCH_SIZE]

        # 调用 Ollama Embedding 接口
        # input 传入列表，response.data 按顺序返回对应向量
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)

        for item in response.data:
            # 将 list[float] 序列化为 JSON 字符串，方便存入 CSV
            # 读取时用 json.loads() 还原
            all_embeddings.append(json.dumps(item.embedding))

    # 写入新列并保存
    df["embedding"] = all_embeddings
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\n完成！已保存至 {OUTPUT_PATH}（{os.path.getsize(OUTPUT_PATH)/1024/1024:.1f} MB）")
    print(f"向量维度：{len(json.loads(df['embedding'].iloc[0]))}")


if __name__ == "__main__":
    main()

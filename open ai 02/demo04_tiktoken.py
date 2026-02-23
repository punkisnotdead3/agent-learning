import sys
import tiktoken

sys.stdout.reconfigure(encoding="utf-8")


# ============================================================
# tiktoken 是 OpenAI 开源的分词库，用于将文本切分成 token
# 安装：pip install tiktoken
#
# 核心概念：
#   - 编码（encoding）：一套词表 + 分词规则，不同模型用不同编码
#   - token：文本被切分后的最小单元，可以是一个词、半个词、一个字符
#     英文大约 1 token ≈ 4 个字符 ≈ 0.75 个单词
#     中文大约 1 token ≈ 1~2 个汉字（视词表而定）
# ============================================================

# ============================================================
# 1. 获取编码器
# ============================================================

# 方式一：按编码名称获取
#   OpenAI 目前主要有三套编码：
#   - cl100k_base  → GPT-4 / GPT-3.5-turbo / text-embedding-ada-002
#   - o200k_base   → GPT-4o 系列（更新的词表，词汇量 20 万）
#   - p50k_base    → 旧版 GPT-3（text-davinci-003 等，已弃用）
enc = tiktoken.get_encoding("cl100k_base")

# 方式二：按模型名称自动选择对应编码（推荐用于 OpenAI 模型）
# enc = tiktoken.encoding_for_model("gpt-4o")

print("=" * 50)
print("编码器名称:", enc.name)
print("词表大小  :", enc.n_vocab)   # 词表中 token 的总数量

# ============================================================
# 2. encode：文本 → token ID 列表
# ============================================================
text = "Hello, how are you? 你好，今天天气怎么样？"

tokens = enc.encode(text)

print()
print("=" * 50)
print("原始文本 :", text)
print("token 数量:", len(tokens))
print("token IDs :", tokens)

# ============================================================
# 3. decode：token ID 列表 → 文本（还原）
# ============================================================
restored = enc.decode(tokens)
print()
print("=" * 50)
print("还原文本  :", restored)

# ============================================================
# 4. 逐个查看每个 token 对应的原始字节/文本
# ============================================================
print()
print("=" * 50)
print("逐 token 拆解:")
for token_id in tokens:
    # decode_single_token_bytes 返回该 token 对应的原始字节
    token_bytes = enc.decode_single_token_bytes(token_id)
    try:
        token_str = token_bytes.decode("utf-8")
    except UnicodeDecodeError:
        token_str = str(token_bytes)   # 部分 token 是半个 UTF-8 字符
    print(f"  ID={token_id:6d}  bytes={token_bytes}  text={repr(token_str)}")

# ============================================================
# 5. 计算 chat 请求的实际 token 数
#    （OpenAI 格式：每条消息除内容外还有固定的结构开销）
# ============================================================
def count_tokens_for_messages(messages: list, model: str = "gpt-4o") -> int:
    """
    参照 OpenAI 官方公式计算 messages 列表的 token 数。
    每条消息固定额外消耗：
      - 3 token（消息结构开销：<|im_start|> role \n content <|im_end|>）
      - role 字段本身的 token
      - content 字段本身的 token
    整个请求末尾还有 3 token 的固定开销（<|im_start|>assistant）
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens_per_message = 3
    total = 0
    for message in messages:
        total += tokens_per_message
        for value in message.values():
            total += len(encoding.encode(value))
    total += 3   # 末尾固定开销
    return total

sample_messages = [
    {"role": "system",    "content": "你是一个简洁的助手。"},
    {"role": "user",      "content": "用一句话解释什么是向量数据库。"},
]

print()
print("=" * 50)
print("messages token 估算（OpenAI 公式）:", count_tokens_for_messages(sample_messages))

# ============================================================
# ⚠️  重要说明：tiktoken 对 DeepSeek 不准确
# ============================================================
# tiktoken 使用的是 OpenAI 的词表（cl100k_base / o200k_base）。
# DeepSeek 使用自己的词表（基于 HuggingFace tokenizers），
# 两者词表不同，分词结果会有差异。
#
# 用 tiktoken 估算 DeepSeek 的 token 数：
#   - 英文：误差较小（英文词汇分词规则相近）
#   - 中文：误差可能较大（中文 token 粒度不同）
#
# 获取 DeepSeek 精确 token 数的唯一可靠方式：
#   → 直接读取 API 返回的 response.usage 字段
#     response.usage.prompt_tokens     ← 输入精确值
#     response.usage.completion_tokens ← 输出精确值
#
# 如需离线精确计算 DeepSeek token，需安装 transformers 并加载
# DeepSeek 官方 tokenizer：
#   from transformers import AutoTokenizer
#   tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
#   tokens = tokenizer.encode(text)
# 但该模型文件体积较大，且需要 HuggingFace 网络访问。
# ============================================================
print()
print("=" * 50)
print("⚠️  tiktoken 对 DeepSeek 仅供估算，精确值请读 response.usage")

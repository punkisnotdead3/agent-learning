import os
import sys
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# ============================================================
# chat.completions.create 参数说明
# ============================================================
response = client.chat.completions.create(
    # model: 指定使用的模型名称
    model="deepseek-chat",

    # messages: 对话历史，列表中每条是一个字典
    #   role 可以是:
    #     "system"    - 系统提示，设定模型的行为/角色
    #     "user"      - 用户输入
    #     "assistant" - 模型之前的回复（多轮对话时用）
    #   content: 该角色说的内容
    messages=[
        {
            "role": "system",
            "content": "你是一个简洁的助手，回答尽量控制在两句话以内。"
        },
        {
            "role": "user",
            "content": "用一句话解释什么是向量数据库。"
        },
    ],

    # max_tokens: 限制模型最多生成多少个 token（不含输入部分）
    # 不设置则由模型自行决定，设置太小会截断回复
    max_tokens=200,

    # temperature: 控制输出的随机性，范围 0~2
    #   0   → 确定性最强，每次结果几乎一样（适合代码/事实问答）
    #   1   → 默认平衡值
    #   >1  → 更随机、更有创意（适合写作/头脑风暴）
    temperature=1.0,

    # top_p: 核采样，范围 0~1
    #   模型只从累计概率达到 top_p 的 token 集合中采样
    #   与 temperature 二选一调节即可，不建议同时大幅修改两者
    top_p=1.0,

    # n: 一次请求生成几条候选回复，默认 1
    #   生成多条时 choices 列表会有多个元素
    n=1,

    # stream: 是否流式返回（像打字机一样逐字输出），默认 False
    #   True 时返回的是迭代器，每次 yield 一个 chunk
    stream=False,

    # stop: 遇到该字符串时提前停止生成，可以是字符串或列表
    #   例如 stop="\n" 表示生成到换行符就停
    # stop=None,

    # presence_penalty: 范围 -2~2，正值惩罚已出现过的 token
    #   鼓励模型谈论新话题，避免重复
    presence_penalty=0,

    # frequency_penalty: 范围 -2~2，正值惩罚高频 token
    #   降低模型逐字重复同一内容的概率
    frequency_penalty=0,
)

# ============================================================
# 打印完整原始 JSON，查看所有字段
# ============================================================
print("=== 原始 JSON ===")
print(response.model_dump_json(indent=2))

# ============================================================
# 返回 JSON 主要字段含义
# ============================================================
# {
#   "id": "chatcmpl-xxx",          // 本次请求的唯一 ID
#   "object": "chat.completion",   // 固定值，表示对象类型
#   "created": 1700000000,         // Unix 时间戳，请求创建时间
#   "model": "deepseek-chat",      // 实际使用的模型
#
#   "choices": [                   // 候选回复列表（n=1 时只有一条）
#     {
#       "index": 0,                // 候选序号
#       "message": {
#         "role": "assistant",     // 固定为 assistant
#         "content": "..."         // 模型生成的回复文本
#       },
#       "finish_reason": "stop"    // 停止原因:
#                                  //   "stop"         - 正常生成完毕
#                                  //   "length"       - 达到 max_tokens 被截断
#                                  //   "content_filter" - 触发内容过滤
#     }
#   ],
#
#   "usage": {                     // token 用量统计（计费依据）
#     "prompt_tokens": 30,         // 输入消耗的 token 数
#     "completion_tokens": 40,     // 输出消耗的 token 数
#     "total_tokens": 70           // 合计
#   }
# }

print()
print("=== 解析关键字段 ===")
print("请求 ID        :", response.id)
print("使用模型       :", response.model)
print("停止原因       :", response.choices[0].finish_reason)
print("输入 tokens    :", response.usage.prompt_tokens)
print("输出 tokens    :", response.usage.completion_tokens)
print("总计 tokens    :", response.usage.total_tokens)
print()
print("=== 模型回复 ===")
print(response.choices[0].message.content)


# ============================================================
# completions.create —— 旧版「文本补全」API
# ============================================================
# 与 chat.completions.create 的核心区别：
#   - chat 版：输入是 messages 列表（多角色对话）
#   - 旧版：输入是一段纯文本 prompt，模型直接续写
#
# 注意：DeepSeek 官方对旧版 API 的支持有限，
#       部分参数（如 echo、best_of）可能不生效。
# ============================================================
# ============================================================
# completions.create —— 旧版「文本补全」API
# ============================================================
# 与 chat.completions.create 的核心区别：
#   - chat 版：输入是 messages 列表（多角色对话）
#   - 旧版：输入是一段纯文本 prompt，模型直接续写
#
# 注意：DeepSeek 的旧版 API 需要使用 beta 端点，
#       因此这里单独创建一个指向 /beta 的 client。
# ============================================================
client_beta = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/beta",  # ← beta 端点才支持旧版补全
)

response2 = client_beta.completions.create(
    # model: 同 chat 版，指定模型
    model="deepseek-chat",

    # prompt: 纯文本输入，模型会在此基础上续写
    #   可以是字符串，也可以是字符串列表（批量补全）
    prompt="向量数据库的核心作用是",

    # max_tokens: 最多续写多少 token
    max_tokens=100,

    # temperature / top_p / n / stream / stop
    # 含义与 chat 版完全相同，不再赘述
    temperature=1.0,

    # suffix: 指定续写内容的结尾，模型会在 prompt 和 suffix 之间填充
    #   例如 prompt="def hello():", suffix="hello()"
    #   常用于代码填充场景（FIM, Fill In the Middle）
    suffix=None,

    # echo: 是否在返回内容中把 prompt 也一并带回，默认 False
    echo=False,

    # logprobs: 返回每个 token 的 top-N 对数概率，默认 None（不返回）
    logprobs=None,
)

print()
print("=" * 50)
print("=== completions.create 原始 JSON ===")
print(response2.model_dump_json(indent=2))

# ============================================================
# 返回 JSON 与 chat 版的差异
# ============================================================
# {
#   "id": "cmpl-xxx",           // 同 chat 版
#   "object": "text_completion", // 注意：不是 chat.completion
#   "created": 1700000000,
#   "model": "deepseek-chat",
#
#   "choices": [
#     {
#       "index": 0,
#       "text": "...",           // ★ 旧版用 text 字段，不是 message.content
#       "logprobs": null,
#       "finish_reason": "stop"
#     }
#   ],
#
#   "usage": { ... }            // 同 chat 版
# }

print()
print("=== completions 解析关键字段 ===")
print("停止原因  :", response2.choices[0].finish_reason)
print("续写内容  :", response2.choices[0].text)

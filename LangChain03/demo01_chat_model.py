import os
import sys
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# LangChain 的 ChatModel 是什么？
#
# LangChain 对各家大模型做了统一封装，
# 不管你用 OpenAI、DeepSeek 还是其他模型，
# 调用方式都是一样的——这就是 ChatModel 的核心价值。
#
# 消息类型（3 种角色）：
#   SystemMessage  —— 系统提示，定义模型的行为和性格
#   HumanMessage   —— 用户说的话
#   AIMessage      —— 模型回复的话（多轮对话时需要带上历史）
#
# 基本流程：
#   1. 创建 ChatModel 实例（配置模型参数）
#   2. 构建消息列表 [SystemMessage, HumanMessage, ...]
#   3. 调用 model.invoke(messages) 得到回复
# ============================================================


# ============================================================
# Step 1：初始化 ChatModel
#
# ChatDeepSeek 是 LangChain 官方提供的 DeepSeek 封装。
# 如果将来想换成 OpenAI，只需改成：
#   from langchain_openai import ChatOpenAI
#   model = ChatOpenAI(model="gpt-4o")
# 下面的所有代码完全不需要改动——这就是 LangChain 的优势。
# ============================================================
model = ChatDeepSeek(
    model="deepseek-chat",                          # 使用 DeepSeek-V3
    api_key=os.environ.get("DEEPSEEK_API_KEY"),     # 从环境变量读取 key
    temperature=0.7,                                # 创造性程度：0=严谨，1=发散
    # max_tokens=1024,                              # 可选：限制回复最大长度
)


# ============================================================
# Part 1：最简单的单轮对话
#
# invoke() 接收一个消息列表，返回一个 AIMessage 对象。
# 消息列表里至少要有一条 HumanMessage。
# ============================================================
print("=" * 55)
print("Part 1：单轮对话")
print("=" * 55)

messages = [
    # SystemMessage 定义模型的"角色设定"，相当于给员工的工作说明
    SystemMessage(content="你是一个编程导师，专门帮助初学者学习 Python。回答要简洁，多用例子。"),

    # HumanMessage 就是用户的提问
    HumanMessage(content="列表和元组有什么区别？"),
]

# invoke() 发送消息并等待回复
# 返回值是一个 AIMessage 对象
response = model.invoke(messages)

# AIMessage 的 .content 属性就是模型回复的文本
print("模型回复：")
print(response.content)
print()

# 查看完整的 AIMessage 对象（包含 token 用量等信息）
print("完整响应对象（AIMessage）：")
print(f"  类型: {type(response)}")
print(f"  token 用量: {response.response_metadata.get('token_usage', '无')}")
print()


# ============================================================
# Part 2：多轮对话
#
# LangChain 的 ChatModel 本身是无状态的——
# 它不会自动记住上一轮说了什么。
#
# 要实现多轮对话，需要手动把历史消息都带上：
#   [System, Human1, AI1, Human2, AI2, Human3, ...]
#
# 每次发送时，把之前所有的消息都带上，
# 模型才能"记住"之前说了什么。
# ============================================================
print("=" * 55)
print("Part 2：多轮对话（手动维护历史）")
print("=" * 55)

# 初始化对话历史，只放系统提示
history = [
    SystemMessage(content="你是一个友好的 Python 编程导师。"),
]

def chat(user_input: str) -> str:
    """
    发送一条消息，自动维护对话历史，返回模型回复文本。

    每次调用时：
      1. 把用户消息加入历史
      2. 把完整历史发给模型
      3. 把模型回复也加入历史（下一轮需要带上）
      4. 返回回复文本
    """
    # 把用户消息加入历史
    history.append(HumanMessage(content=user_input))

    # 把完整历史发给模型
    response = model.invoke(history)

    # 把模型回复也加入历史，下一轮对话时带上
    history.append(AIMessage(content=response.content))

    return response.content


# 第一轮
print("用户：我想学习 Python，从哪里开始？")
reply1 = chat("我想学习 Python，从哪里开始？")
print(f"助手：{reply1}")
print()

# 第二轮（模型知道上文说的是"学 Python"）
print("用户：变量怎么定义？")
reply2 = chat("变量怎么定义？")
print(f"助手：{reply2}")
print()

# 第三轮（模型知道上文说的是"变量定义"）
print("用户：能给我一个例子吗？")
reply3 = chat("能给我一个例子吗？")
print(f"助手：{reply3}")
print()

# 查看当前对话历史里有多少条消息
print(f"当前历史消息数量：{len(history)} 条")
print("历史消息类型列表：")
for i, msg in enumerate(history):
    role = type(msg).__name__   # SystemMessage / HumanMessage / AIMessage
    preview = msg.content[:30].replace("\n", " ")  # 只显示前30字
    print(f"  [{i}] {role}: {preview}...")
print()


# ============================================================
# Part 3：流式输出（Streaming）
#
# 默认的 invoke() 是等模型全部生成完才返回。
# stream() 是逐字/逐块返回，用户体验更好（像打字机效果）。
# ============================================================
print("=" * 55)
print("Part 3：流式输出（打字机效果）")
print("=" * 55)

print("助手（流式）：", end="", flush=True)

# stream() 返回一个生成器，每次 yield 一个文本块
for chunk in model.stream([
    SystemMessage(content="你是一个助手。"),
    HumanMessage(content="用三句话介绍一下 LangChain 是什么。"),
]):
    # chunk.content 是这一块的文本片段
    print(chunk.content, end="", flush=True)

print()  # 换行
print()

print("=" * 55)
print("总结")
print("=" * 55)
print("""
LangChain ChatModel 核心要点：

1. 统一接口：换模型只改 import 和初始化，其他不变
2. 消息类型：SystemMessage / HumanMessage / AIMessage
3. 无状态：模型本身不记忆，多轮对话需手动带上历史
4. invoke()：发送消息，等待完整回复
5. stream()：发送消息，流式逐块返回（打字机效果）
""")

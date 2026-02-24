import os
import sys
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# 为什么需要 Memory（对话记忆）？
#
# 大模型本身是完全无状态的：
#   你发一条消息，它回一条消息，然后"忘得一干二净"。
#   下次你再发消息，它完全不知道之前说过什么。
#
# 这就像每次打电话给客服，对方都是新的，
#   你每次都得重新介绍自己。
#
# Memory 的作用：在多轮对话中让模型"记住"之前说过的话。
#
# LangChain 中实现 Memory 的两种方式：
#   方式一：手动维护消息列表（简单直接，demo01 已演示过）
#   方式二：用 RunnableWithMessageHistory（更规范，适合生产环境）
#
# Memory 的三种常见策略：
#   1. 全量记忆   —— 保留所有对话历史（会越来越长）
#   2. 窗口记忆   —— 只保留最近 N 轮对话（控制长度）
#   3. 摘要记忆   —— 用模型对历史做摘要，压缩后保留（节省 token）
# ============================================================

model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    temperature=0.7,
)


# ============================================================
# Part 1：无记忆的问题演示
#
# 先感受一下没有记忆时的痛点。
# ============================================================
print("=" * 55)
print("Part 1：无记忆的问题演示")
print("=" * 55)

no_memory_chain = (
    ChatPromptTemplate.from_messages([("human", "{input}")])
    | model
    | StrOutputParser()
)

print("用户：我叫小明，我在学 Python")
r1 = no_memory_chain.invoke({"input": "我叫小明，我在学 Python"})
print(f"助手：{r1}\n")

print("用户：我叫什么名字？")
r2 = no_memory_chain.invoke({"input": "我叫什么名字？"})
print(f"助手：{r2}")
print("（模型不记得了！每次都是全新对话）")
print()


# ============================================================
# Part 2：手动维护消息列表（最简单的记忆实现）
#
# 原理：把每一轮的 HumanMessage 和 AIMessage 都存到列表里，
#       下次调用时把完整列表带上。
#
# 优点：简单直观，完全自己控制
# 缺点：对话越长，发送的 token 越多；需要自己管理列表
# ============================================================
print("=" * 55)
print("Part 2：手动维护消息列表（全量记忆）")
print("=" * 55)

# 初始化历史，放系统提示
chat_history = [
    SystemMessage(content="你是一个友好的编程助手，记住用户说过的信息。"),
]

def chat_with_memory(user_input: str) -> str:
    """带记忆的对话函数"""
    # 把用户消息加入历史
    chat_history.append(HumanMessage(content=user_input))
    # 把完整历史发给模型
    response = model.invoke(chat_history)
    # 把模型回复也存入历史（下一轮需要带上）
    chat_history.append(AIMessage(content=response.content))
    return response.content

print("用户：我叫小红，最近在学 LangChain")
r1 = chat_with_memory("我叫小红，最近在学 LangChain")
print(f"助手：{r1}\n")

print("用户：我在学什么？")
r2 = chat_with_memory("我在学什么？")
print(f"助手：{r2}\n")

print("用户：我叫什么名字？")
r3 = chat_with_memory("我叫什么名字？")
print(f"助手：{r3}\n")

print(f"当前历史消息数：{len(chat_history)} 条")
print()


# ============================================================
# Part 3：RunnableWithMessageHistory（现代 LCEL 写法）
#
# 这是 LangChain 推荐的生产环境写法：
#   - 用 InMemoryChatMessageHistory 存储历史（可换成数据库）
#   - 用 RunnableWithMessageHistory 把历史注入 chain
#   - 通过 session_id 区分不同用户的对话
#
# session_id 的作用：
#   就像每个用户有自己的"聊天室 ID"，
#   不同用户的历史记录互不干扰。
# ============================================================
print("=" * 55)
print("Part 3：RunnableWithMessageHistory（推荐写法）")
print("=" * 55)

# 用字典模拟"数据库"，key 是 session_id，value 是该用户的历史
# 真实项目中这里可以换成 Redis、数据库等持久化存储
session_store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    根据 session_id 获取（或创建）对话历史。
    RunnableWithMessageHistory 每次调用时会自动调用这个函数。
    """
    if session_id not in session_store:
        # 第一次对话，创建新的历史记录
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# 定义带 MessagesPlaceholder 的提示词模板
# MessagesPlaceholder 是历史消息的占位符，历史会自动填入这里
prompt_with_history = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的编程助手，会记住用户说的信息。"),
    MessagesPlaceholder(variable_name="history"),  # 历史消息插入这里
    ("human", "{input}"),                          # 当前用户输入
])

# 基础 chain（不带记忆）
base_chain = prompt_with_history | model | StrOutputParser()

# 用 RunnableWithMessageHistory 包装，赋予记忆能力
chain_with_memory = RunnableWithMessageHistory(
    base_chain,
    get_session_history,            # 传入获取历史的函数
    input_messages_key="input",     # 告诉它哪个字段是用户输入
    history_messages_key="history", # 告诉它把历史放到哪个占位符
)

# 调用时通过 config 传入 session_id
# 同一个 session_id = 同一个对话，历史自动保留
config_user_a = {"configurable": {"session_id": "user_A"}}
config_user_b = {"configurable": {"session_id": "user_B"}}

print("--- 用户A 的对话 ---")
print("用户A：我是小强，我在做一个电商项目")
r = chain_with_memory.invoke({"input": "我是小强，我在做一个电商项目"}, config=config_user_a)
print(f"助手：{r}\n")

print("用户A：我的项目是做什么的？")
r = chain_with_memory.invoke({"input": "我的项目是做什么的？"}, config=config_user_a)
print(f"助手：{r}\n")

print("--- 用户B 的对话（和用户A 完全隔离）---")
print("用户B：我叫小李")
r = chain_with_memory.invoke({"input": "我叫小李"}, config=config_user_b)
print(f"助手：{r}\n")

print("用户B：你知道小强是谁吗？")
r = chain_with_memory.invoke({"input": "你知道小强是谁吗？"}, config=config_user_b)
print(f"助手：{r}")
print("（用户B 看不到用户A 的历史）\n")

# 查看历史存储情况
print(f"当前 session 数量：{len(session_store)}")
for sid, history in session_store.items():
    print(f"  {sid}：{len(history.messages)} 条消息")
print()


# ============================================================
# Part 4：窗口记忆（只保留最近 N 轮）
#
# 全量记忆的问题：对话越长，token 消耗越多，成本越高。
# 解决方案：只保留最近 N 轮对话，旧的丢掉。
#
# 这里手动实现窗口记忆逻辑，直观易懂。
# ============================================================
print("=" * 55)
print("Part 4：窗口记忆（只保留最近 N 轮）")
print("=" * 55)

WINDOW_SIZE = 2  # 只保留最近 2 轮对话（4 条消息：2 Human + 2 AI）

window_history = []

def chat_with_window(user_input: str) -> str:
    """只保留最近 N 轮的对话函数"""
    window_history.append(HumanMessage(content=user_input))
    response = model.invoke([
        SystemMessage(content="你是一个助手。"),
        *window_history,    # 展开当前窗口内的历史
    ])
    window_history.append(AIMessage(content=response.content))

    # 超出窗口大小就删掉最老的一轮（2条：1 Human + 1 AI）
    if len(window_history) > WINDOW_SIZE * 2:
        window_history.pop(0)   # 删最老的 HumanMessage
        window_history.pop(0)   # 删最老的 AIMessage

    return response.content

turns = [
    "第一轮：我在学 Python",
    "第二轮：我在学 LangChain",
    "第三轮：我在学 RAG",
    "你记得我第一轮说的是什么吗？",  # 第一轮已被窗口丢弃
]

for msg in turns:
    print(f"用户：{msg}")
    reply = chat_with_window(msg)
    print(f"助手：{reply}")
    print(f"（当前窗口消息数：{len(window_history)}）\n")


print("=" * 55)
print("总结：三种记忆策略对比")
print("=" * 55)
print("""
全量记忆（Part 2/3）：
  原理：保留所有历史消息
  优点：什么都记得
  缺点：对话越长 token 越多，成本越高
  适合：短对话、对记忆完整性要求高的场景

窗口记忆（Part 4）：
  原理：只保留最近 N 轮
  优点：token 消耗可控
  缺点：会遗忘早期内容
  适合：大多数聊天机器人场景

摘要记忆（未演示，进阶）：
  原理：用模型把旧历史压缩成摘要，摘要 + 近期历史一起带上
  优点：兼顾记忆完整性和 token 控制
  缺点：需要额外调用模型生成摘要，增加成本
  适合：需要长期记忆且对成本敏感的场景

生产环境推荐：
  用 RunnableWithMessageHistory（Part 3）
  + 把 InMemoryChatMessageHistory 换成 Redis 或数据库
  + 根据需要加窗口或摘要逻辑
""")

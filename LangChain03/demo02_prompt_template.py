import os
import sys
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# PromptTemplate 是什么？为什么需要它？
#
# 没有 PromptTemplate 时，你的代码可能长这样：
#
#   city = "北京"
#   language = "Python"
#   prompt = f"请用 {language} 写一段查询 {city} 天气的代码"
#
# 这样写有几个问题：
#   1. 提示词散落在代码各处，难以维护
#   2. 变量替换全靠手动 f-string，容易出错
#   3. 没法复用——换个场景又要重写
#
# PromptTemplate 解决了这些问题：
#   - 把提示词模板集中管理
#   - 用 {变量名} 占位，统一替换
#   - 模板可复用，只需传入不同参数
#
# LangChain 有两种常用模板：
#   1. PromptTemplate       —— 用于普通字符串提示词
#   2. ChatPromptTemplate   —— 用于对话格式（有角色区分），更常用
# ============================================================

model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    temperature=0.7,
)


# ============================================================
# Part 1：PromptTemplate（普通字符串模板）
#
# 适合简单场景，生成一段纯文本提示词。
# 用 {变量名} 作为占位符，调用时传入具体值。
# ============================================================
print("=" * 55)
print("Part 1：PromptTemplate 基本用法")
print("=" * 55)

# 定义模板，{topic} 和 {level} 是占位符
template = PromptTemplate(
    input_variables=["topic", "level"],     # 声明模板里用到的变量名
    template="请用适合{level}的语言，解释一下「{topic}」是什么，100字以内。",
)

# 方法一：用 .format() 填入变量，得到完整的字符串
filled_prompt = template.format(topic="递归", level="初学者")
print("填充后的提示词：")
print(filled_prompt)
print()

# 直接把填充好的字符串发给模型
response = model.invoke(filled_prompt)
print("模型回复：")
print(response.content)
print()

# 换个参数复用同一个模板
filled_prompt2 = template.format(topic="闭包", level="有一定基础的程序员")
response2 = model.invoke(filled_prompt2)
print("换个参数（闭包 + 有基础）：")
print(response2.content)
print()


# ============================================================
# Part 2：ChatPromptTemplate（对话模板）
#
# 实际开发中更常用这个，因为它支持多角色消息。
# 格式：[(角色, 模板内容), ...]
#   "system"  —— 系统提示
#   "human"   —— 用户消息
#   "ai"      —— 模型历史回复（多轮对话时用）
# ============================================================
print("=" * 55)
print("Part 2：ChatPromptTemplate（对话模板）")
print("=" * 55)

chat_template = ChatPromptTemplate.from_messages([
    # 系统提示模板，{style} 是占位符
    ("system", "你是一个{style}风格的编程导师，用简洁的语言回答问题。"),
    # 用户消息模板，{question} 是占位符
    ("human", "{question}"),
])

# .invoke() 传入字典，填充所有占位符
# 返回的是一个消息列表（ChatPromptValue），可以直接发给模型
messages = chat_template.invoke({
    "style": "幽默有趣",
    "question": "for 循环和 while 循环有什么区别？",
})

# 查看填充后的消息内容
print("填充后的消息列表：")
for msg in messages.messages:
    print(f"  [{type(msg).__name__}] {msg.content}")
print()

response3 = model.invoke(messages)
print("模型回复：")
print(response3.content)
print()


# ============================================================
# Part 3：用 | 把模板和模型串联起来（Chain）
#
# LangChain 的核心特性之一：用管道符 | 把组件串成链
# 写法：chain = 模板 | 模型
# 调用：chain.invoke({"变量": "值"})
#
# 好处：代码更简洁，组件可以随意替换和拼接
# 这就是 LangChain 名字中 "Chain" 的由来
# ============================================================
print("=" * 55)
print("Part 3：Chain（模板 | 模型）")
print("=" * 55)

# 定义一个代码生成模板
code_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个 {language} 专家，只输出代码，不要解释，代码要有注释。"),
    ("human", "写一个函数：{task}"),
])

# 用 | 把模板和模型串起来，形成一个 chain
# chain 的调用方式和单独调用模型一样，但内部自动完成"填充模板→发给模型"
chain = code_template | model

# 直接传变量字典调用
result = chain.invoke({
    "language": "Python",
    "task": "判断一个数是否为质数",
})
print("生成的代码：")
print(result.content)
print()

# 换个语言，复用同一个 chain
result2 = chain.invoke({
    "language": "JavaScript",
    "task": "计算数组中所有数字的平均值",
})
print("JavaScript 版本：")
print(result2.content)
print()


# ============================================================
# Part 4：多轮对话模板（带历史消息占位符）
#
# MessagesPlaceholder 是一个特殊占位符，
# 专门用来在模板里预留一个位置，放入历史消息列表。
# ============================================================
print("=" * 55)
print("Part 4：带历史的对话模板")
print("=" * 55)

from langchain_core.prompts import MessagesPlaceholder

history_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手。"),
    # MessagesPlaceholder 会把传入的消息列表原样插入这里
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# 模拟一段历史对话
fake_history = [
    HumanMessage(content="我叫小明"),
    SystemMessage(content="你好，小明！有什么我可以帮你的吗？"),
]

messages = history_template.invoke({
    "history": fake_history,                # 插入历史消息
    "input": "你还记得我叫什么名字吗？",    # 当前问题
})

response4 = model.invoke(messages)
print("用户：你还记得我叫什么名字吗？")
print(f"助手：{response4.content}")
print()

print("=" * 55)
print("总结")
print("=" * 55)
print("""
PromptTemplate 核心要点：

1. PromptTemplate     —— 简单字符串模板，用 {变量} 占位
2. ChatPromptTemplate —— 对话模板，支持 system/human/ai 多角色
3. MessagesPlaceholder —— 在模板里预留位置插入历史消息列表
4. 管道符 |           —— 把模板和模型串成 chain，一步调用

使用模板的好处：
  ✅ 提示词集中管理，便于修改
  ✅ 参数化复用，不重复写代码
  ✅ 配合 | 构成 chain，代码更简洁
""")

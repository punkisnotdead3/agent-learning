import os
import sys
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,    # 把输入原样传递
    RunnableLambda,         # 把普通 Python 函数包装成可以接入管道的组件
    RunnableParallel,       # 并行运行多个分支
    RunnableBranch,         # 条件分支路由
)

sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# LCEL 是什么？（LangChain Expression Language）
#
# LCEL 是 LangChain 在 2023 年推出的核心语法，
# 核心就是用 | 管道符把各种组件串联起来。
#
# 设计思想来自 Unix 管道：
#   cat file.txt | grep "error" | sort | uniq
#
# LangChain 里：
#   prompt | model | parser
#
# 好处：
#   1. 写法简洁，逻辑清晰
#   2. 所有组件接口统一（都有 invoke / stream / batch）
#   3. 组件可以随意替换、组合
#   4. 自动支持流式输出、批量处理、异步
#
# 能接入管道的组件（统称 Runnable）：
#   - ChatPromptTemplate    模板
#   - ChatModel             模型
#   - OutputParser          解析器
#   - RunnablePassthrough   原样透传
#   - RunnableLambda        自定义函数
#   - RunnableParallel      并行分支
#   - RunnableBranch        条件路由
# ============================================================

model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    temperature=0.7,
)
parser = StrOutputParser()


# ============================================================
# Part 1：基础管道（你已经见过的）
#
# prompt | model | parser 的执行过程：
#   1. prompt.invoke({"input": "..."})  → 生成消息列表
#   2. model.invoke(消息列表)           → 生成 AIMessage
#   3. parser.invoke(AIMessage)         → 提取纯文本字符串
#
# 每一步的输出，自动成为下一步的输入。
# ============================================================
print("=" * 55)
print("Part 1：基础管道 prompt | model | parser")
print("=" * 55)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个简洁的助手，回答不超过30字。"),
    ("human", "{input}"),
])

# 三个组件用 | 串起来
basic_chain = prompt | model | parser

result = basic_chain.invoke({"input": "Python 是什么？"})
print(f"类型：{type(result)}")   # str，不是 AIMessage
print(f"结果：{result}")
print()


# ============================================================
# Part 2：RunnablePassthrough（透传输入）
#
# 有时候你既想把数据传给模型处理，
# 又想在后续步骤里保留原始输入。
# RunnablePassthrough 就做这件事：把输入原样传递下去。
#
# 常见用法：配合 RunnableParallel 同时保留原始数据和处理结果
# ============================================================
print("=" * 55)
print("Part 2：RunnablePassthrough（保留原始输入）")
print("=" * 55)

# RunnableParallel 并行执行多个分支，结果合并成字典
# 这里同时做两件事：
#   "original" 分支：原样保留输入
#   "processed" 分支：经过模型处理
parallel_chain = RunnableParallel({
    "original": RunnablePassthrough(),       # 原样透传
    "processed": basic_chain,                # 经过模型处理
})

result = parallel_chain.invoke({"input": "什么是递归？"})
print(f"原始输入：{result['original']}")
print(f"模型处理：{result['processed']}")
print()


# ============================================================
# Part 3：RunnableLambda（把普通函数接入管道）
#
# 有时候你需要在管道中间做一些自定义处理，
# 比如：格式化数据、过滤内容、记录日志。
# RunnableLambda 可以把任意 Python 函数包装成管道组件。
# ============================================================
print("=" * 55)
print("Part 3：RunnableLambda（自定义函数接入管道）")
print("=" * 55)

# 定义几个普通 Python 函数
def add_prefix(text: str) -> str:
    """在文本前加前缀"""
    return f"【AI回答】{text}"

def count_chars(text: str) -> str:
    """统计字数并追加到末尾"""
    return f"{text}\n（共 {len(text)} 字）"

def log_output(text: str) -> str:
    """记录日志，同时把数据原样传下去"""
    print(f"  [日志] 当前管道输出长度：{len(text)} 字")
    return text  # 必须返回数据，让管道继续

# 用 RunnableLambda 包装函数，接入管道
lambda_chain = (
    prompt
    | model
    | parser
    | RunnableLambda(log_output)     # 先记录日志
    | RunnableLambda(add_prefix)     # 再加前缀
    | RunnableLambda(count_chars)    # 最后统计字数
)

result = lambda_chain.invoke({"input": "列举三种排序算法"})
print("最终结果：")
print(result)
print()


# ============================================================
# Part 4：RunnableParallel（并行执行）
#
# 同时运行多个独立的链，最后汇总结果。
# 好处：并行执行比串行更快（多个 API 调用同时发出）
#
# 应用场景：
#   - 同时生成多个角度的回答
#   - 同时做翻译和总结
#   - 同时查询多个数据源
# ============================================================
print("=" * 55)
print("Part 4：RunnableParallel（并行执行多个链）")
print("=" * 55)

# 定义三个不同角度的提示词
prompt_simple = ChatPromptTemplate.from_messages([
    ("human", "用一句话（10字内）解释：{topic}"),
])
prompt_detail = ChatPromptTemplate.from_messages([
    ("human", "用三点解释{topic}，每点15字内"),
])
prompt_example = ChatPromptTemplate.from_messages([
    ("human", "用一个生活中的比喻解释{topic}，30字内"),
])

# 三个链并行运行
parallel_explain = RunnableParallel({
    "简单版": prompt_simple | model | parser,
    "详细版": prompt_detail | model | parser,
    "比喻版": prompt_example | model | parser,
})

print("同时从三个角度解释「变量」：")
results = parallel_explain.invoke({"topic": "变量"})
for key, value in results.items():
    print(f"\n【{key}】")
    print(value)
print()


# ============================================================
# Part 5：RunnableBranch（条件路由）
#
# 根据输入内容，动态选择走哪条链。
# 类似 if/elif/else，但以管道形式表达。
#
# 语法：RunnableBranch(
#   (条件函数1, 链1),
#   (条件函数2, 链2),
#   默认链,              # 所有条件都不满足时走这里
# )
# ============================================================
print("=" * 55)
print("Part 5：RunnableBranch（条件路由）")
print("=" * 55)

# 三种不同风格的回答链
chain_formal = (
    ChatPromptTemplate.from_messages([("human", "请用正式专业的语言回答：{question}")])
    | model | parser
)
chain_casual = (
    ChatPromptTemplate.from_messages([("human", "请用轻松口语的方式回答：{question}")])
    | model | parser
)
chain_simple = (
    ChatPromptTemplate.from_messages([("human", "请用小学生能理解的方式回答：{question}")])
    | model | parser
)

# 根据输入里的 style 字段选择链
branch_chain = RunnableBranch(
    # (条件, 对应的链)：条件是一个返回 True/False 的函数
    (lambda x: x.get("style") == "formal", chain_formal),
    (lambda x: x.get("style") == "casual", chain_casual),
    # 默认链（没有条件，直接放最后）
    chain_simple,
)

question = "黑洞是什么？"

print(f"问题：{question}\n")

print("[正式风格]")
print(branch_chain.invoke({"question": question, "style": "formal"}))
print()

print("[口语风格]")
print(branch_chain.invoke({"question": question, "style": "casual"}))
print()

print("[默认（简单）风格]")
print(branch_chain.invoke({"question": question, "style": "other"}))
print()


# ============================================================
# Part 6：invoke / stream / batch 三种调用方式
#
# 所有 LCEL 链都自动支持三种调用方式，不需要额外代码：
#   invoke()  —— 等待完整结果（最常用）
#   stream()  —— 流式逐块返回（打字机效果）
#   batch()   —— 批量处理多个输入（自动并发）
# ============================================================
print("=" * 55)
print("Part 6：invoke / stream / batch 三种调用方式")
print("=" * 55)

simple_chain = (
    ChatPromptTemplate.from_messages([("human", "{input}")])
    | model
    | parser
)

# 方式一：invoke（单次调用，等待完整结果）
print("【invoke】单次调用：")
result = simple_chain.invoke({"input": "用一句话介绍 LangChain"})
print(result)
print()

# 方式二：stream（流式输出，逐字打印）
print("【stream】流式输出：", end="", flush=True)
for chunk in simple_chain.stream({"input": "用一句话介绍人工智能"}):
    print(chunk, end="", flush=True)
print("\n")

# 方式三：batch（批量处理，内部自动并发）
print("【batch】批量处理三个问题：")
inputs = [
    {"input": "Python 的优点"},
    {"input": "JavaScript 的优点"},
    {"input": "Go 语言的优点"},
]
results = simple_chain.batch(inputs)
for i, r in enumerate(results):
    print(f"  问题{i+1}：{r}")
print()


print("=" * 55)
print("总结：LCEL 核心组件速查")
print("=" * 55)
print("""
管道符 |               把组件串联：prompt | model | parser

RunnablePassthrough    原样透传输入，常配合 RunnableParallel 使用
RunnableLambda(fn)     把普通 Python 函数接入管道
RunnableParallel({})   并行执行多个链，结果合并成字典
RunnableBranch(...)    条件路由，根据输入选择不同的链

三种调用方式（所有链通用）：
  chain.invoke(input)   单次调用
  chain.stream(input)   流式输出
  chain.batch([...])    批量处理
""")

import os
import sys
import csv

sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# Data Connection（数据连接）是什么？
#
# 大模型训练完之后，知识就"冻结"了。
# 它不知道你公司的内部文档、最新新闻、私有数据库。
#
# Data Connection 解决的问题：
#   如何把外部文档"喂给"模型，让它能基于这些文档回答问题。
#
# LangChain Data Connection 的完整流程（也是 RAG 的基础）：
#
#   外部文档（PDF/网页/CSV/数据库）
#       ↓  【Document Loaders】加载器：把各种格式统一成 Document 对象
#   Document 对象列表
#       ↓  【Text Splitters】切割器：把长文档切成小块
#   小块列表（Chunks）
#       ↓  【Embeddings】向量化：把文字转成数字向量（RAG 阶段）
#   向量
#       ↓  【Vector Stores】向量库：存储和检索相似内容（RAG 阶段）
#   检索结果
#       ↓  发给模型回答问题
#
# 本 demo 重点介绍前两步：Document Loaders + Text Splitters
# Embeddings 和 Vector Stores 留给 RAG demo 专门讲
# ============================================================

# ============================================================
# 先创建一些示例文件，用于后面的加载演示
# ============================================================
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))

# 创建示例 txt 文件
txt_path = os.path.join(DEMO_DIR, "sample.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("""LangChain 是什么？

LangChain 是一个用于开发大语言模型应用的开源框架，由 Harrison Chase 于 2022 年创建。
它提供了一套标准化的工具和接口，帮助开发者更轻松地构建基于 LLM 的应用程序。

LangChain 的核心功能包括：
1. 模型集成：支持 OpenAI、Anthropic、DeepSeek 等几十种大模型
2. 提示词管理：通过 PromptTemplate 统一管理提示词
3. 链式调用：用管道符 | 把多个组件串联起来
4. 记忆管理：帮助模型在多轮对话中记住上下文
5. 数据连接：加载 PDF、网页、数据库等外部数据源
6. Agent：让模型自主决策，调用工具完成复杂任务

LangChain 的设计哲学是"组合优于继承"，所有组件都遵循统一的 Runnable 接口，
可以灵活地组合和替换，大大降低了开发 LLM 应用的复杂度。

LangChain 在 2024 年进行了大规模重构，推出了 LCEL（LangChain Expression Language），
用管道符 | 取代了老版本的 Chain 类，使代码更加简洁和直观。
""")

# 创建示例 CSV 文件
csv_path = os.path.join(DEMO_DIR, "sample.csv")
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["姓名", "职位", "技能", "年限"])
    writer.writerow(["张三", "后端工程师", "Python, Django, PostgreSQL", "5"])
    writer.writerow(["李四", "前端工程师", "React, TypeScript, CSS", "3"])
    writer.writerow(["王五", "AI工程师", "LangChain, PyTorch, RAG", "2"])
    writer.writerow(["赵六", "全栈工程师", "Vue, FastAPI, Docker", "4"])


# ============================================================
# Part 1：Document 对象是什么？
#
# LangChain 用 Document 对象统一表示所有外部内容。
# 不管是 PDF、网页还是 CSV，加载后都变成 Document。
#
# Document 只有两个字段：
#   page_content  —— 文本内容（字符串）
#   metadata      —— 元数据（字典，记录来源、页码等信息）
# ============================================================
print("=" * 55)
print("Part 1：Document 对象结构")
print("=" * 55)

from langchain_core.documents import Document

# 手动创建一个 Document（了解结构）
doc = Document(
    page_content="这是文档的正文内容，可以是任何文字。",
    metadata={
        "source": "example.pdf",    # 来源文件
        "page": 1,                  # 页码（PDF 场景）
        "author": "张三",           # 自定义元数据
    }
)
print(f"page_content：{doc.page_content}")
print(f"metadata：{doc.metadata}")
print()


# ============================================================
# Part 2：TextLoader（加载纯文本文件）
#
# 最简单的加载器，把 .txt 文件读成 Document 列表。
# 一个文件 = 一个 Document。
# ============================================================
print("=" * 55)
print("Part 2：TextLoader（加载 txt 文件）")
print("=" * 55)

from langchain_community.document_loaders import TextLoader

loader = TextLoader(txt_path, encoding="utf-8")

# load() 返回 Document 列表
docs = loader.load()

print(f"加载了 {len(docs)} 个 Document")
print(f"metadata：{docs[0].metadata}")
print(f"内容前100字：{docs[0].page_content[:100]}...")
print()


# ============================================================
# Part 3：CSVLoader（加载 CSV 文件）
#
# 把 CSV 的每一行变成一个 Document。
# ============================================================
print("=" * 55)
print("Part 3：CSVLoader（加载 CSV 文件）")
print("=" * 55)

from langchain_community.document_loaders.csv_loader import CSVLoader

csv_loader = CSVLoader(csv_path, encoding="utf-8")
csv_docs = csv_loader.load()

print(f"加载了 {len(csv_docs)} 个 Document（每行一个）")
for doc in csv_docs:
    print(f"\n  内容：{doc.page_content}")
    print(f"  来源行：{doc.metadata}")
print()


# ============================================================
# Part 4：WebBaseLoader（加载网页）
#
# 抓取网页内容，自动去除 HTML 标签，只保留文字。
# 需要安装 beautifulsoup4。
# ============================================================
print("=" * 55)
print("Part 4：WebBaseLoader（加载网页）")
print("=" * 55)

from langchain_community.document_loaders import WebBaseLoader

# 加载一个真实网页
url = "https://python.langchain.com/docs/introduction/"
print(f"正在加载网页：{url}")

web_loader = WebBaseLoader(url)
try:
    web_docs = web_loader.load()
    print(f"加载成功！共 {len(web_docs)} 个 Document")
    print(f"内容前200字：\n{web_docs[0].page_content[:200].strip()}...")
    print(f"metadata：{web_docs[0].metadata}")
except Exception as e:
    print(f"加载失败（可能是网络问题）：{e}")
print()


# ============================================================
# Part 5：Text Splitters（文本切割器）
#
# 为什么要切割？
#   1. 模型有 context window 限制，长文档放不进去
#   2. 检索时，小块比大块更精准（RAG 场景）
#   3. 每次只发相关的小块，节省 token
#
# 最常用：RecursiveCharacterTextSplitter
#   按优先级尝试这些分隔符：["\n\n", "\n", " ", ""]
#   先按段落分，段落太长再按行分，行太长再按词分
#   尽量保证切割后的块在语义上相对完整
# ============================================================
print("=" * 55)
print("Part 5：RecursiveCharacterTextSplitter（文本切割）")
print("=" * 55)

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,         # 每个块最多 200 个字符
    chunk_overlap=30,       # 相邻块之间重叠 30 个字符（保证上下文连贯）
    # chunk_overlap 的作用：
    # 如果一句话被切在了两个块的边界，
    # 重叠部分确保两个块都包含这句话，不会丢失信息
)

# 对刚才加载的 txt 文档进行切割
chunks = splitter.split_documents(docs)

print(f"原始文档：1 个，{len(docs[0].page_content)} 字")
print(f"切割后：{len(chunks)} 个块")
print()

for i, chunk in enumerate(chunks):
    print(f"【块 {i+1}】{len(chunk.page_content)} 字")
    print(f"  内容：{chunk.page_content[:60].strip()}...")
    print(f"  metadata：{chunk.metadata}")
print()


# ============================================================
# Part 6：直接切割纯文本（不经过 Loader）
#
# 有时候你已经有了字符串，不需要从文件加载，
# 直接用 split_text() 切割成字符串列表。
# ============================================================
print("=" * 55)
print("Part 6：直接切割纯文本字符串")
print("=" * 55)

long_text = """
人工智能（AI）是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。
机器学习是 AI 的核心技术之一，它让计算机通过数据自动学习规律，而无需显式编程。
深度学习是机器学习的子集，使用多层神经网络处理复杂数据，在图像识别和自然语言处理领域表现出色。
大语言模型（LLM）是深度学习的最新成果，通过在海量文本数据上训练，获得了强大的语言理解和生成能力。
LangChain 作为 LLM 应用开发框架，让开发者能够方便地把这些强大的模型集成到实际产品中。
"""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

# split_text() 直接接受字符串，返回字符串列表（不是 Document）
text_chunks = text_splitter.split_text(long_text.strip())

print(f"原始文本：{len(long_text.strip())} 字")
print(f"切割成 {len(text_chunks)} 块：")
for i, chunk in enumerate(text_chunks):
    print(f"\n  块{i+1}（{len(chunk)}字）：{chunk}")
print()


# ============================================================
# 清理示例文件
# ============================================================
os.remove(txt_path)
os.remove(csv_path)

print("=" * 55)
print("总结：Data Connection 完整流程")
print("=" * 55)
print("""
第一步：选择合适的 Loader 加载文档
  TextLoader          → txt 文件
  CSVLoader           → csv 文件
  WebBaseLoader       → 网页
  PyPDFLoader         → PDF（需装 pypdf）
  DirectoryLoader     → 整个文件夹
  ...还有几十种 Loader

第二步：用 Text Splitter 切割成小块
  RecursiveCharacterTextSplitter  → 最常用，智能按段落/行/词切割
  CharacterTextSplitter           → 按单个分隔符切割
  TokenTextSplitter               → 按 token 数切割

第三步（RAG，下个 demo 讲）：
  Embeddings    → 把文字块转成向量
  VectorStore   → 存储向量，支持相似度搜索
  Retriever     → 根据问题检索最相关的块
  Chain         → 把检索结果 + 问题发给模型回答

核心数据结构：Document
  doc.page_content   文本内容
  doc.metadata       来源信息（文件名、页码、URL 等）
""")

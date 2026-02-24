import os
import sys
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, CommaSeparatedListOutputParser
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# OutputParser 是什么？为什么需要它？
#
# 模型返回的永远是一个 AIMessage 对象，
# 里面的内容是一段纯文本字符串。
#
# 但实际开发中，我们往往需要的不是字符串，而是：
#   - 一个 Python 字符串（去掉 AIMessage 包装）
#   - 一个列表 ["苹果", "香蕉", "橙子"]
#   - 一个字典 {"name": "张三", "age": 18}
#   - 一个 Python 对象（有类型、有校验）
#
# OutputParser 的作用：
#   把模型输出的原始文本 → 转换成你想要的 Python 数据类型
#
# 常用的 Parser：
#   1. StrOutputParser          —— 最简单，AIMessage → 纯字符串
#   2. CommaSeparatedListOutputParser —— 文本 → Python 列表
#   3. JsonOutputParser         —— JSON 文本 → Python 字典
#   4. PydanticOutputParser     —— JSON 文本 → Pydantic 对象（有类型校验）
# ============================================================

model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    temperature=0.7,
)


# ============================================================
# Part 1：StrOutputParser（最常用）
#
# 模型返回的是 AIMessage 对象，
# 但大多数时候你只需要里面的文本内容。
# StrOutputParser 就做这一件事：把 AIMessage → 纯字符串。
# ============================================================
print("=" * 55)
print("Part 1：StrOutputParser（AIMessage → 字符串）")
print("=" * 55)

# 没有 Parser 时
response = model.invoke("用一句话介绍 Python")
print(f"没有 Parser，返回类型：{type(response)}")         # <class 'AIMessage'>
print(f"需要 .content 才能拿到文本：{response.content}")
print()

# 用了 StrOutputParser
prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}")
])
parser = StrOutputParser()

# 用 | 把三者串成 chain：模板 → 模型 → 解析器
chain = prompt | model | parser

result = chain.invoke({"input": "用一句话介绍 Python"})
print(f"有 Parser，返回类型：{type(result)}")             # <class 'str'>
print(f"直接就是字符串：{result}")
print()


# ============================================================
# Part 2：CommaSeparatedListOutputParser（文本 → 列表）
#
# 让模型返回逗号分隔的内容，Parser 自动拆成列表。
# 适合：生成标签、关键词、推荐列表等场景。
# ============================================================
print("=" * 55)
print("Part 2：CommaSeparatedListOutputParser（文本 → 列表）")
print("=" * 55)

list_parser = CommaSeparatedListOutputParser()

# get_format_instructions() 生成一段说明文字，告诉模型应该用什么格式输出
# 直接塞到提示词里，引导模型按格式回答
format_instructions = list_parser.get_format_instructions()
print("格式说明（会插入提示词）：")
print(format_instructions)
print()

list_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{domain}领域的专家。"),
    ("human", "列举5个{topic}，{instructions}"),
])

list_chain = list_prompt | model | list_parser

result_list = list_chain.invoke({
    "domain": "编程",
    "topic": "Python 常用的内置函数",
    "instructions": format_instructions,    # 把格式要求插入提示词
})

print(f"返回类型：{type(result_list)}")     # <class 'list'>
print(f"列表内容：{result_list}")
print(f"第一个元素：{result_list[0]}")
print(f"元素数量：{len(result_list)}")
print()


# ============================================================
# Part 3：JsonOutputParser（文本 → 字典）
#
# 让模型返回 JSON 格式的文本，Parser 自动解析成 Python 字典。
# 适合：需要结构化数据、多个字段的场景。
# ============================================================
print("=" * 55)
print("Part 3：JsonOutputParser（文本 → 字典）")
print("=" * 55)

json_parser = JsonOutputParser()

json_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个数据提取助手，只返回 JSON 格式，不要有任何多余文字。"),
    ("human", "提取以下文本中的人物信息，返回包含 name、age、city 字段的 JSON：\n{text}"),
])

json_chain = json_prompt | model | json_parser

result_dict = json_chain.invoke({
    "text": "小明今年25岁，住在上海，是一名软件工程师。"
})

print(f"返回类型：{type(result_dict)}")     # <class 'dict'>
print(f"字典内容：{result_dict}")
print(f"姓名：{result_dict.get('name')}")
print(f"年龄：{result_dict.get('age')}")
print(f"城市：{result_dict.get('city')}")
print()


# ============================================================
# Part 4：PydanticOutputParser（最强大，带类型校验）
#
# 在 JsonOutputParser 基础上更进一步：
#   - 定义一个 Pydantic 模型，明确每个字段的类型和说明
#   - 模型输出会被校验和转换成这个对象
#   - 字段类型不对会报错，更安全
#
# 适合：正式项目、需要严格数据类型的场景。
# ============================================================
print("=" * 55)
print("Part 4：PydanticOutputParser（文本 → Pydantic 对象）")
print("=" * 55)

# 第一步：定义数据结构（像写数据库表结构一样）
class MovieReview(BaseModel):
    title: str = Field(description="电影名称")
    score: float = Field(description="评分，0到10分")
    genre: str = Field(description="电影类型，如：动作、喜剧、科幻")
    summary: str = Field(description="一句话简评")
    recommend: bool = Field(description="是否推荐，true 或 false")

# 第二步：创建 Parser，传入你定义的数据结构
pydantic_parser = PydanticOutputParser(pydantic_object=MovieReview)

# get_format_instructions() 会根据你的类结构，自动生成详细的格式要求
format_instructions = pydantic_parser.get_format_instructions()
print("自动生成的格式说明（前200字）：")
print(format_instructions[:200] + "...")
print()

pydantic_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的电影评论家。"),
    ("human", "请评价电影《{movie}》\n\n{format_instructions}"),
])

pydantic_chain = pydantic_prompt | model | pydantic_parser

result_obj = pydantic_chain.invoke({
    "movie": "泰坦尼克号",
    "format_instructions": format_instructions,
})

print(f"返回类型：{type(result_obj)}")          # <class 'MovieReview'>
print(f"电影名：{result_obj.title}")
print(f"评分：{result_obj.score}")              # 自动是 float 类型
print(f"类型：{result_obj.genre}")
print(f"简评：{result_obj.summary}")
print(f"是否推荐：{result_obj.recommend}")      # 自动是 bool 类型
print()
# Pydantic 对象可以直接转成字典
print(f"转成字典：{result_obj.dict()}")
print()


print("=" * 55)
print("总结")
print("=" * 55)
print("""
OutputParser 选择指南：

  只需要文本字符串         → StrOutputParser          （最常用）
  需要一个列表             → CommaSeparatedListOutputParser
  需要字典/多个字段        → JsonOutputParser
  需要类型校验+对象        → PydanticOutputParser      （正式项目推荐）

使用套路（固定写法）：
  chain = prompt | model | parser
  result = chain.invoke({...})

Parser 的两个职责：
  1. 告诉模型该输出什么格式（get_format_instructions()）
  2. 把模型输出的文本转成对应的 Python 类型
""")

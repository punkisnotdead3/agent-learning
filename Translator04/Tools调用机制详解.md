# Tools 调用机制详解

## 一句话总结

> **LLM 负责"想"，Tools 负责"做"，LLM 再负责"说"。**

---

## 完整流程

```
用户输入
   ↓
LLM 理解意图
"我需要查天气，应该调用 get_weather 这个 tool"
   ↓
Tool 执行（真正的代码逻辑）
get_weather("上海") → "小雨，12°C"
   ↓
LLM 润色结果
"上海今天小雨，气温12度，建议带伞出门"
   ↓
返回给用户
```

---

## LLM 在整个过程中做了两件事

| 阶段 | LLM 做的事 | 类比 |
|------|-----------|------|
| 调用前 | 理解意图，选择 tool，决定参数 | 大脑决策 |
| 调用后 | 把 tool 的原始结果润色成自然语言 | 嘴巴表达 |

**LLM 不执行 tool，只是"指挥"。真正执行的是你写的 Python 函数。**

---

## 关键细节：docstring 决定 tool 的调用质量

LLM 完全靠读 tool 的 docstring 来决定：
- 要不要调用这个 tool
- 传什么参数进去

```python
@tool
def get_weather(city: str) -> str:
    """
    查询指定城市的今日天气信息。
    当用户询问某个城市的天气、温度、是否需要带伞时使用此工具。
    输入：城市名称，例如：北京、上海
    """
    ...
```

docstring 写得模糊 → LLM 选错 tool 或传错参数
docstring 写得清晰 → LLM 精准调用，结果准确

---

## 本项目（翻译 Agent）的两条 Chain

```
用户输入："帮我把人工智能正在改变世界翻译成英语"
         ↓
  【Chain 1 解析链】
  parse_prompt | model | PydanticOutputParser
         ↓ 输出结构化对象
  source_text    = "人工智能正在改变世界"
  target_language = "英语"
         ↓
  【RunnableLambda 转换】
  ParseResult 对象 → 字典
         ↓
  【Chain 2 翻译链】
  translate_prompt | model | StrOutputParser
         ↓
  "Artificial intelligence is changing the world"
```

Chain 1 让 LLM "想清楚"要翻译什么、翻译成什么语言。
Chain 2 让 LLM 真正执行翻译任务。
两者通过 `RunnableLambda` 串联，数据格式在中间完成转换。

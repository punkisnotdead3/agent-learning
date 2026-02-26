import os
import sys
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# 翻译 Agent
#
# 功能：用户输入一段文本 + 目标语言，Agent 调用 LLM 完成翻译
#
# 流程：
#   用户输入（原文 + 目标语言）
#       ↓
#   PromptTemplate 组装提示词
#       ↓
#   DeepSeek LLM 执行翻译
#       ↓
#   StrOutputParser 提取纯文本
#       ↓
#   返回翻译结果
# ============================================================

# 初始化模型
model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    temperature=0.3,    # 翻译场景用低温度，保证准确性，减少"发挥"
)

# 翻译提示词模板
translate_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个专业翻译，精通各国语言。"
        "请将用户提供的文本准确翻译成目标语言。"
        "只输出翻译结果，不要加任何解释、注释或多余文字。"
    ),
    (
        "human",
        "请将以下文本翻译成{target_language}：\n\n{text}"
    ),
])

# 构建翻译 chain：提示词 → 模型 → 解析器
translate_chain = translate_prompt | model | StrOutputParser()


def translate(text: str, target_language: str) -> str:
    """
    翻译函数
    :param text: 需要翻译的原文
    :param target_language: 目标语言，例如：英语、日语、法语、西班牙语
    :return: 翻译结果
    """
    result = translate_chain.invoke({
        "text": text,
        "target_language": target_language,
    })
    return result


def run_interactive():
    """交互模式：循环接收用户输入，输入 quit 退出"""
    print("=" * 50)
    print("       翻译 Agent（输入 quit 退出）")
    print("=" * 50)
    print("支持任意语言，例如：英语、日语、法语、韩语、西班牙语等")
    print()

    while True:
        # 输入原文
        text = input("请输入要翻译的文本：").strip()
        if text.lower() == "quit":
            print("已退出。")
            break
        if not text:
            print("文本不能为空，请重新输入。\n")
            continue

        # 输入目标语言
        target_language = input("请输入目标语言：").strip()
        if target_language.lower() == "quit":
            print("已退出。")
            break
        if not target_language:
            print("目标语言不能为空，请重新输入。\n")
            continue

        # 执行翻译
        print("\n翻译中...\n")
        result = translate(text, target_language)

        print("-" * 50)
        print(f"原文（→ {target_language}）：")
        print(text)
        print(f"\n译文：")
        print(result)
        print("-" * 50)
        print()


if __name__ == "__main__":
    run_interactive()

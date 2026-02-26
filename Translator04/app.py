import os
import sys
import gradio as gr
from pydantic import BaseModel, Field
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableLambda

sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# 整体设计思路（两条 Chain 串联）
#
# 用户输入："帮我把'人工智能正在改变世界'翻译成日语"
#
#   【Chain 1：解析链】
#   用户自然语言输入
#       ↓ parse_prompt
#       ↓ model
#       ↓ PydanticOutputParser   ← 把输出结构化成 ParseResult 对象
#   ParseResult(source_text="人工智能正在改变世界", target_language="日语")
#
#   【Chain 2：翻译链】
#   ParseResult 对象
#       ↓ RunnableLambda         ← 把对象转成字典，传给下一个 prompt
#       ↓ translate_prompt
#       ↓ model
#       ↓ StrOutputParser
#   "人工知能は世界を変えています"
#
#   完整 Pipeline：
#   parse_chain | RunnableLambda(转换) | translate_chain
# ============================================================

model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    temperature=0,
)


# ============================================================
# Step 1：定义结构化输出的数据模型
#
# 用 Pydantic 定义解析结果的数据结构：
#   source_text     —— 用户想翻译的原文
#   target_language —— 目标语言
# ============================================================
class ParseResult(BaseModel):
    source_text: str = Field(description="用户想要翻译的原始文本内容")
    target_language: str = Field(description="目标语言，例如：英语、日语、法语")


# ============================================================
# Chain 1：解析链
#
# 职责：理解用户的自然语言指令，提取出"原文"和"目标语言"
# 输入：用户的一句话，例如 "帮我把xxx翻译成英语"
# 输出：ParseResult 对象（有 source_text 和 target_language 两个字段）
# ============================================================
parse_parser = PydanticOutputParser(pydantic_object=ParseResult)

parse_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个指令解析器，专门从用户的翻译请求中提取关键信息。\n"
        "请从用户输入中识别：1）需要翻译的原文  2）目标语言\n\n"
        "{format_instructions}"        # PydanticOutputParser 自动生成的格式说明
    ),
    (
        "human",
        "{user_input}"
    ),
])

# 解析链：提示词 → 模型 → 结构化解析器
parse_chain = parse_prompt | model | parse_parser


# ============================================================
# Chain 2：翻译链
#
# 职责：拿到结构化的原文和目标语言，执行翻译
# 输入：包含 source_text 和 target_language 的字典
# 输出：翻译结果字符串
# ============================================================
translate_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个专业翻译，精通各国语言。"
        "只输出翻译结果，不要加任何解释或多余文字。"
    ),
    (
        "human",
        "请将以下文本翻译成{target_language}：\n\n{source_text}"
    ),
])

# 翻译链：提示词 → 模型 → 字符串解析器
translate_chain = translate_prompt | model | StrOutputParser()


# ============================================================
# 串联两条 Chain
#
# parse_chain 输出的是 ParseResult 对象，
# translate_chain 需要的是字典 {"source_text": ..., "target_language": ...}
#
# 用 RunnableLambda 做中间转换：ParseResult 对象 → 字典
# ============================================================
def parse_result_to_dict(result: ParseResult) -> dict:
    """把 ParseResult 对象转换成翻译链需要的字典格式"""
    return {
        "source_text": result.source_text,
        "target_language": result.target_language,
    }

# 完整 pipeline：解析链 → 转换 → 翻译链
full_pipeline = (
    parse_chain
    | RunnableLambda(parse_result_to_dict)
    | translate_chain
)


# ============================================================
# 供 Gradio 调用的主函数
# 同时返回中间解析结果，让用户看到 Agent 的推理过程
# ============================================================
def translate(user_input: str):
    if not user_input.strip():
        return "请输入翻译指令", ""

    # Step 1：调用解析链，提取原文和目标语言
    parsed: ParseResult = parse_chain.invoke({
        "user_input": user_input,
        "format_instructions": parse_parser.get_format_instructions(),
    })

    # Step 2：调用翻译链，执行翻译
    result = translate_chain.invoke({
        "source_text": parsed.source_text,
        "target_language": parsed.target_language,
    })

    # 把解析过程展示出来，便于理解
    parse_info = (
        f"解析结果：\n"
        f"  原文：{parsed.source_text}\n"
        f"  目标语言：{parsed.target_language}"
    )

    return parse_info, result


# ============================================================
# Gradio 界面
# ============================================================
with gr.Blocks(title="AI 翻译助手") as app:

    gr.Markdown("# AI 翻译助手")
    gr.Markdown(
        "直接输入翻译指令，Agent 会自动解析原文和目标语言，再执行翻译：\n"
        "- `帮我把人工智能正在改变世界翻译成英语`\n"
        "- `把下面这段话翻译成日语：我今天很开心`\n"
        "- `translate hello world to Chinese`"
    )

    with gr.Column():
        input_text = gr.Textbox(
            label="翻译指令",
            placeholder="例如：帮我把人工智能正在改变世界翻译成英语",
            lines=3,
        )
        translate_btn = gr.Button("翻译", variant="primary")

    with gr.Row():
        # 左侧展示解析结果（Chain 1 的输出）
        parse_output = gr.Textbox(
            label="Chain 1 解析结果",
            lines=4,
            interactive=False,
        )
        # 右侧展示翻译结果（Chain 2 的输出）
        translate_output = gr.Textbox(
            label="Chain 2 翻译结果",
            lines=4,
            interactive=False,
        )

    translate_btn.click(
        fn=translate,
        inputs=input_text,
        outputs=[parse_output, translate_output],
    )

    input_text.submit(
        fn=translate,
        inputs=input_text,
        outputs=[parse_output, translate_output],
    )


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )

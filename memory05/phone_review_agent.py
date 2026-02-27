import os
import sys
import gradio as gr
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# 手机评测 Agent（带记忆的多轮对话）
#
# 功能：
#   - 专业手机评测助手，熟悉各品牌手机的参数和特性
#   - 支持多轮对话，记住上下文（例如：问完摄像头再问续航，
#     Agent 知道你还在聊同一款手机）
#   - 支持多用户隔离（每个浏览器 tab 是独立的会话）
#
# 记忆实现方式：RunnableWithMessageHistory
#   - InMemoryChatMessageHistory 存储每个 session 的历史
#   - 通过 session_id 区分不同用户
#   - 每轮对话自动把历史注入 prompt，模型能记住上文
# ============================================================

model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    temperature=0.7,
)

# ============================================================
# 提示词：定义评测助手的"性格"和专业能力
# MessagesPlaceholder 是历史消息的占位符，记忆会自动填入这里
# ============================================================
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一位资深手机评测专家，有超过10年的手机评测经验，"
        "熟悉苹果、三星、华为、小米、OPPO、vivo、一加等各大品牌。\n\n"
        "你的评测风格：\n"
        "- 客观公正，既说优点也指出缺点\n"
        "- 数据说话，引用真实的跑分、参数、实测数据\n"
        "- 结合使用场景给出建议（学生党、商务人士、摄影爱好者等）\n"
        "- 语言简洁易懂，避免过度堆砌专业术语\n"
        "- 每次回答控制在500字以内，言简意赅\n\n"
        "【重要：上下文追踪规则】\n"
        "你必须始终追踪对话中出现过的所有手机型号，建立一个隐式的「品牌→型号」映射。\n"
        "规则如下：\n"
        "1. 用户只说品牌名（如「华为」）时，自动关联该品牌在对话中最近提到的具体型号，"
        "并在回答开头确认，例如：「根据我们之前的对话，您说的华为应该是指华为 Mate 50，以下是对比：」\n"
        "2. 若同一品牌在对话中出现过多个型号，则主动询问：「您说的华为是指 Mate 50 还是其他型号？」\n"
        "3. 绝对不能猜测一个对话中从未出现过的型号。"
    ),
    MessagesPlaceholder(variable_name="history"),   # 历史消息自动填入
    ("human", "{input}"),
])

base_chain = prompt | model | StrOutputParser()

# ============================================================
# 用字典存储所有 session 的对话历史
# key = session_id（每个用户/tab 一个）
# value = InMemoryChatMessageHistory 对象
# ============================================================
session_store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# 用 RunnableWithMessageHistory 包装，赋予记忆能力
agent = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


# ============================================================
# Gradio 对话函数
#
# Gradio Chatbot 组件的消息格式是列表：
#   [ [用户消息, 助手回复], [用户消息, 助手回复], ... ]
#
# 流程：
#   1. 把用户消息发给 Agent（Agent 内部自动读取历史）
#   2. 拿到回复后追加到 Gradio 的 chat_history 列表
#   3. 返回更新后的列表，Gradio 自动刷新界面
# ============================================================
def chat(user_input: str, chat_history: list, session_id: str):
    if not user_input.strip():
        return "", chat_history, ""

    # 调用前先拿到当前历史，构造"实际发给 LLM 的完整消息"用于展示
    history_msgs = get_session_history(session_id).messages

    # 拼接调试信息：展示每一条实际发给 LLM 的消息
    debug_lines = ["# 本次实际发给 LLM 的完整消息\n"]
    debug_lines.append("## System Prompt")
    debug_lines.append(prompt.messages[0].prompt.template)
    debug_lines.append(f"\n## 历史消息（共 {len(history_msgs)} 条）")
    for i, msg in enumerate(history_msgs):
        role = "用户" if msg.type == "human" else "助手"
        debug_lines.append(f"\n**[{i+1}] {role}：**\n{msg.content}")
    debug_lines.append(f"\n## 本轮用户输入")
    debug_lines.append(user_input)
    debug_text = "\n".join(debug_lines)

    # 调用 Agent
    response = agent.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )

    chat_history.append((user_input, response))
    return "", chat_history, debug_text


def clear_history(session_id: str):
    """清空指定 session 的对话历史"""
    if session_id in session_store:
        session_store[session_id].clear()
    return [], ""   # 同时清空聊天记录和调试面板


# ============================================================
# Gradio 界面（两个 Tab）
# ============================================================
with gr.Blocks(title="手机评测助手") as app:

    gr.Markdown("# 手机评测助手")

    import uuid
    session_id = gr.State(value=str(uuid.uuid4()))

    with gr.Tab("对话"):
        gr.Markdown(
            "我是资深手机评测专家，可以帮你分析任何手机的优缺点、横向对比、购买建议。\n"
            "支持多轮对话，问完一个问题可以继续追问，我会记住我们聊过的内容。"
        )

        chatbot = gr.Chatbot(label="对话记录", height=480)

        with gr.Row():
            input_box = gr.Textbox(
                label="",
                placeholder="例如：帮我评测一下 iPhone 16 Pro Max...",
                scale=9,
            )
            send_btn = gr.Button("发送", variant="primary", scale=1)

        clear_btn = gr.Button("清空对话")

        gr.Markdown(
            "**推荐提问方式：**\n"
            "- `iPhone 16 Pro Max 和三星 S25 Ultra 哪个拍照更好？`\n"
            "- `小米15的续航怎么样？`\n"
            "- `（接上文）那散热呢？`  ← Agent 会记得你在聊小米15\n"
            "- `3000元预算买什么手机合适，主要用来打游戏`"
        )

    with gr.Tab("调试：实际发给 LLM 的消息"):
        gr.Markdown(
            "每次发送消息后，这里会展示本轮**完整发给 LLM 的内容**，\n"
            "包括 System Prompt、所有历史消息、以及本轮用户输入。\n"
            "这就是 LLM 实际「看到」的全部信息。"
        )
        debug_panel = gr.Markdown(value="*发送第一条消息后，这里会显示内容...*")

    # 发送消息，同时更新对话和调试面板
    send_btn.click(
        fn=chat,
        inputs=[input_box, chatbot, session_id],
        outputs=[input_box, chatbot, debug_panel],
    )
    input_box.submit(
        fn=chat,
        inputs=[input_box, chatbot, session_id],
        outputs=[input_box, chatbot, debug_panel],
    )

    # 清空对话，同时清空调试面板
    clear_btn.click(
        fn=clear_history,
        inputs=[session_id],
        outputs=[chatbot, debug_panel],
    )


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",   # 监听所有网卡，局域网内可访问
        server_port=7860,
        share=True,
    )

import os
import sys
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# ============================================================
# 多轮对话的核心：用一个列表维护完整的对话历史
# 每次请求都把全部历史发给模型，模型才能"记住"上文
#
# role 标识这条消息是谁说的，共三种取值：
#   "system"    - 系统提示，设定模型的行为规则，由开发者写
#   "user"      - 用户的输入
#   "assistant" - 模型的回复
#
# "assistant" 消息的作用：把模型上一轮的回答告诉模型本身，
# 让它知道自己之前说了什么，从而保持上下文连贯。
# 模型并不会"主动记住"自己说过的话——是我们收到回复后，
# 手动把它塞回 messages 列表，下次请求时一起发过去。
#
# 因此也可以"伪造" assistant 消息：在发请求前手动插入一条
# {"role": "assistant", "content": "..."}, 模型会把它当成
# 自己说过的内容来理解，常用于预填充回复或角色扮演场景。
# ============================================================
messages = [
    {
        "role": "system",
        "content": "你是一个耐心的编程老师，回答简洁，每次不超过三句话。"
    }
]


def chat(user_input: str) -> str:
    """
    发送一轮对话，自动维护 messages 历史。
    1. 把用户输入追加到 messages
    2. 把完整 messages 发给模型
    3. 把模型回复追加到 messages（供下一轮使用）
    4. 返回模型回复文本
    """
    # 第一步：把本轮用户输入加入历史
    messages.append({"role": "user", "content": user_input})

    # 第二步：带着完整历史调用 API
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,      # ← 关键：传入全部历史，而不只是当前输入
        max_tokens=300,
        temperature=1.0,
    )

    # 第三步：取出模型回复
    assistant_message = response.choices[0].message.content

    # 第四步：把模型回复也加入历史，供下一轮参考
    messages.append({"role": "assistant", "content": assistant_message})

    return assistant_message


# ============================================================
# 模拟三轮对话，每轮依赖上一轮的上下文
# ============================================================

print("=" * 50)
print("第 1 轮")
print("=" * 50)
q1 = "什么是闭包？"
print(f"用户: {q1}")
print(f"助手: {chat(q1)}")

print()
print("=" * 50)
print("第 2 轮（引用上文概念，不重复解释）")
print("=" * 50)
q2 = "能给我一个 Python 的例子吗？"   # 没有说"闭包的例子"，模型靠上下文理解
print(f"用户: {q2}")
print(f"助手: {chat(q2)}")

print()
print("=" * 50)
print("第 3 轮（继续追问）")
print("=" * 50)
q3 = "这个例子中，outer 函数执行完后为什么 x 没有被销毁？"
print(f"用户: {q3}")
print(f"助手: {chat(q3)}")

# ============================================================
# 打印当前完整的 messages 历史，直观看到"记忆"的本质
# ============================================================
print()
print("=" * 50)
print("完整 messages 历史（共 %d 条）" % len(messages))
print("=" * 50)
for i, msg in enumerate(messages):
    print(f"[{i}] {msg['role']:10s}: {msg['content'][:60]}{'...' if len(msg['content']) > 60 else ''}")

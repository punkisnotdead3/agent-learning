import os
import sys
import time
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# Assistants API 是什么？和普通 Chat API 有什么区别？
#
# 【普通 Chat API（我们之前用的方式）】
#   - 每次对话你都要手动把全部历史消息带上发给模型
#   - 没有"记忆"，全靠你自己维护消息列表
#   - 适合简单的一问一答，或者自己完全掌控上下文的场景
#
# 【Assistants API（本 demo 演示的方式）】
#   - OpenAI 帮你在云端保存对话历史（Thread）
#   - 你不需要手动维护消息列表，发消息就好
#   - 内置工具：代码解释器、文件搜索、函数调用
#   - 适合需要多轮对话、有状态的助手应用
#
# 核心概念（4 个）：
#   1. Assistant —— 助手配置（类似一个"角色设定"），可复用
#   2. Thread    —— 对话线程，保存该用户的完整聊天记录
#   3. Message   —— 线程里的一条消息（用户说的 / 助手回复的）
#   4. Run       —— 让助手处理线程里的消息，产生回复
#
# 流程图：
#   创建 Assistant（一次）
#       ↓
#   创建 Thread（每个用户/会话一个）
#       ↓
#   用户发消息 → 加入 Thread
#       ↓
#   创建 Run → 等待完成
#       ↓
#   读取 Thread 里最新的助手回复
#       ↓
#   用户继续发消息（Thread 自动记住之前的对话）
# ============================================================

# ⚠️  Assistants API 是 OpenAI 独有功能，DeepSeek 不支持
#     需要去 platform.openai.com 注册并获取 OPENAI_API_KEY
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    # 注意：这里不设置 base_url，使用 OpenAI 官方地址
)


# ============================================================
# Step 1：创建 Assistant（助手配置）
#
# Assistant 就像一个"员工档案"：
#   - 给它一个名字
#   - 告诉它用什么大模型
#   - 用 instructions 定义它的"性格/职责"（相当于 system prompt）
#   - 可选开启内置工具（本 demo 先不用工具，保持简单）
#
# 关键点：Assistant 创建一次后会保存在 OpenAI 云端，
#         可以被反复使用，不需要每次对话都重新创建。
# ============================================================
print("=" * 55)
print("Step 1：创建 Assistant")
print("=" * 55)

assistant = client.beta.assistants.create(
    name="学习小助手",                          # 助手的名字（便于识别）
    model="gpt-4o-mini",                        # 使用的模型（mini 版本更便宜）
    instructions=(                              # 相当于 system prompt，定义助手行为
        "你是一个耐心的学习助手，专门帮助编程新手理解概念。"
        "回答要简洁易懂，可以用比喻和举例。"
        "每次回答后，主动问用户是否还有疑问。"
    ),
)

print(f"✅ Assistant 创建成功！")
print(f"   ID：{assistant.id}")     # 这个 ID 可以下次直接复用这个 assistant
print(f"   名字：{assistant.name}")
print()


# ============================================================
# Step 2：创建 Thread（对话线程）
#
# Thread 就像一个"聊天室"：
#   - 每个用户或每次独立会话，创建一个 Thread
#   - OpenAI 自动帮你存储这个 Thread 里的所有消息
#   - 你只管发新消息，不需要手动维护历史记录
#   - Thread 没有过期时间，可以随时继续之前的对话
# ============================================================
print("=" * 55)
print("Step 2：创建 Thread（对话线程）")
print("=" * 55)

thread = client.beta.threads.create()

print(f"✅ Thread 创建成功！")
print(f"   ID：{thread.id}")       # 这个 ID 代表一次独立的对话会话
print()


# ============================================================
# Step 3：向 Thread 添加用户消息
#
# 用户说的话，以 role="user" 加入 Thread。
# 注意：加消息 ≠ 让助手回复，只是把消息放进去，
#       下一步的 Run 才会触发助手处理消息。
# ============================================================
print("=" * 55)
print("Step 3：用户发第一条消息")
print("=" * 55)

user_message_1 = "你好！能用一个生活中的比喻解释一下「线程」是什么吗？"
print(f"用户：{user_message_1}")
print()

client.beta.threads.messages.create(
    thread_id=thread.id,            # 指定加入哪个 Thread
    role="user",                    # 角色固定是 "user"
    content=user_message_1,         # 消息内容
)


# ============================================================
# Step 4：创建 Run（触发助手处理消息）
#
# Run 是让助手"开始思考并回复"的指令。
# Assistants API 是异步的——你发出 Run 请求后，
# 需要轮询等待，直到状态变成 "completed"。
#
# Run 的状态流转：
#   queued → in_progress → completed（正常情况）
#                       → failed / expired（出错）
#                       → requires_action（需要调用工具时）
# ============================================================
print("=" * 55)
print("Step 4：创建 Run，等待助手回复...")
print("=" * 55)

# create_and_poll 是一个便捷方法，自动帮你轮询直到完成
# 等价于：先 create run，再循环检查状态，直到 completed
run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id=assistant.id,
)

print(f"Run 状态：{run.status}")    # 应该是 "completed"
print()


# ============================================================
# Step 5：读取助手的回复
#
# 读取 Thread 里的消息列表，取最新的助手消息。
# 消息列表默认按时间倒序排列（最新的在前面）。
# ============================================================
print("=" * 55)
print("Step 5：读取助手回复")
print("=" * 55)

if run.status == "completed":
    # 获取 Thread 中的所有消息（默认最新在前）
    messages = client.beta.threads.messages.list(thread_id=thread.id)

    # 第一条就是助手的最新回复
    assistant_reply = messages.data[0].content[0].text.value
    print(f"助手：{assistant_reply}")
else:
    print(f"❌ Run 未完成，状态：{run.status}")
print()


# ============================================================
# Step 6：继续对话（体现 Thread 的核心价值）
#
# 关键！！这里不需要带上之前的历史消息，
# Thread 已经记住了前面的对话内容，
# 助手会自动基于上下文回答。
#
# 对比普通 Chat API：你需要手动把所有历史 messages 拼起来带上
# Assistants API：只发新消息，历史自动保留
# ============================================================
print("=" * 55)
print("Step 6：继续追问（Thread 自动记住上下文）")
print("=" * 55)

user_message_2 = "好的！那「进程」和「线程」有什么区别呢？"
print(f"用户：{user_message_2}")
print()

# 直接加新消息，不需要带历史
client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content=user_message_2,
)

# 再次 Run，助手知道上文说了"线程的比喻"，可以前后呼应
run2 = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id=assistant.id,
)

if run2.status == "completed":
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    assistant_reply_2 = messages.data[0].content[0].text.value
    print(f"助手：{assistant_reply_2}")
print()


# ============================================================
# Step 7：清理资源（可选）
#
# Thread 和 Assistant 会持久保存在 OpenAI 云端，占用存储。
# 如果这是临时测试，用完后删掉比较好。
# 真实应用中，Thread 通常保留（用户下次可以继续聊），
# Assistant 可以长期复用。
# ============================================================
print("=" * 55)
print("Step 7：清理资源")
print("=" * 55)

client.beta.threads.delete(thread.id)
print(f"✅ Thread {thread.id} 已删除")

client.beta.assistants.delete(assistant.id)
print(f"✅ Assistant {assistant.id} 已删除")
print()

print("=" * 55)
print("总结：Assistants API vs 普通 Chat API")
print("=" * 55)
print("""
普通 Chat API：
  ✅ 灵活，完全自己控制
  ✅ 延迟低，简单场景够用
  ❌ 需要手动管理对话历史
  ❌ 没有内置工具支持

Assistants API：
  ✅ 自动管理对话历史（Thread）
  ✅ 内置工具（代码执行、文件搜索等）
  ✅ Assistant 可复用，不用每次重新配置
  ❌ 有额外存储费用
  ❌ 异步机制，延迟比直接调用高一点
  ❌ 只有 OpenAI 支持，DeepSeek 等不支持
""")

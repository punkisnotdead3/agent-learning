import os
import sys
import json
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# ============================================================
# Function Call（工具调用）是什么？
#
# 大模型本身不能执行代码、查数据库、调接口。
# Function Call 让模型可以"请求"调用我们预先定义好的函数，
# 由我们在本地执行后把结果再喂给模型，模型再给出最终回答。
#
# 完整流程：
#   1. 我们把"工具清单"（函数描述+参数 schema）和用户问题一起发给模型
#   2. 模型判断是否需要调用工具，如果需要则返回 tool_calls（而不是直接回答）
#   3. 我们解析 tool_calls，在本地真正执行对应函数
#   4. 把执行结果以 role="tool" 的消息追加到 messages，再次调用模型
#   5. 模型拿到结果后给出最终自然语言回答
# ============================================================


# ============================================================
# Step 1：定义本地函数（模型不会直接执行这些，我们来执行）
# ============================================================

def get_weather(city: str) -> str:
    """模拟查询天气，真实场景中这里会调用天气 API"""
    mock_data = {
        "北京": "晴，气温 -2°C，西北风 3 级",
        "上海": "阴，气温 8°C，东南风 2 级",
        "广州": "多云，气温 18°C，南风 1 级",
    }
    return mock_data.get(city, f"暂无 {city} 的天气数据")


def calculate(expression: str) -> str:
    """安全计算简单数学表达式"""
    try:
        # 只允许数字和基本运算符，防止代码注入
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "表达式包含非法字符"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算失败: {e}"


# ============================================================
# Step 2：把函数描述成模型能理解的 JSON Schema（工具清单）
# ============================================================
tools = [
    {
        "type": "function",           # 固定值，目前只支持 function
        "function": {
            "name": "get_weather",    # 函数名，模型调用时会原样返回
            "description":            # 告诉模型这个函数是干什么的（很重要！）
                "查询指定城市的实时天气信息",
            "parameters": {           # 用 JSON Schema 描述参数
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海"
                    }
                },
                "required": ["city"]  # 必填参数
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式，例如 '3 * (4 + 5)'",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "合法的数学表达式字符串"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# ============================================================
# Step 3：第一次调用——把问题 + 工具清单发给模型
# ============================================================
messages = [
    {"role": "user", "content": "北京今天天气怎么样？顺便帮我算一下 123 * 456 等于多少？"}
]

print("=" * 55)
print("【第一次请求】发送问题 + 工具清单")
print("=" * 55)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    tools=tools,
    # tool_choice 控制模型是否使用工具：
    #   "auto"     → 模型自己决定用不用（默认）
    #   "none"     → 禁止使用工具
    #   {"type": "function", "function": {"name": "xxx"}} → 强制调用指定函数
    tool_choice="auto",
)

print("finish_reason:", response.choices[0].finish_reason)
# finish_reason == "tool_calls" 说明模型要调用工具，而不是直接回答

print("原始响应 JSON:")
print(response.model_dump_json(indent=2))

# ============================================================
# Step 4：解析 tool_calls，在本地执行函数
# ============================================================
response_message = response.choices[0].message

# 把模型的回复（含 tool_calls）加入历史
messages.append(response_message)

# 遍历所有工具调用（模型可能一次请求调用多个工具）
tool_results = []
for tool_call in response_message.tool_calls:
    func_name = tool_call.function.name
    # 模型返回的参数是 JSON 字符串，需要反序列化
    func_args = json.loads(tool_call.function.arguments)

    print()
    print("=" * 55)
    print(f"【本地执行】函数={func_name}，参数={func_args}")
    print("=" * 55)

    # 根据函数名分发调用
    if func_name == "get_weather":
        result = get_weather(**func_args)
    elif func_name == "calculate":
        result = calculate(**func_args)
    else:
        result = f"未知函数: {func_name}"

    print(f"执行结果: {result}")
    tool_results.append((tool_call.id, func_name, result))

# ============================================================
# Step 5：把执行结果发回给模型，获取最终回答
# ============================================================
# 每个工具结果对应一条 role="tool" 的消息
# tool_call_id 用于让模型知道这个结果属于哪次工具调用
for call_id, func_name, result in tool_results:
    messages.append({
        "role": "tool",
        "tool_call_id": call_id,   # 与 tool_call.id 对应
        "content": result,
    })

print()
print("=" * 55)
print("【第二次请求】把工具执行结果发回，获取最终回答")
print("=" * 55)

final_response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    tools=tools,
)

print("最终回答:")
print(final_response.choices[0].message.content)

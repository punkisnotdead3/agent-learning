import os
import sys
from typing import Any, Union

sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# demo03_self_healing.py -- Tool 开发最佳实践：自我修复能力
#
# 【核心知识点】
# 1. 工具出错时不应直接抛异常终止流程，而要返回结构化错误
# 2. ToolError 包含 recovery_suggestion，告诉 Agent 下一步怎么做
# 3. Agent 读到建议后，能自动调整策略重试，而不是直接放弃
# 4. 区分"可恢复"错误（有建议）和"不可恢复"错误（解释原因即可）
#
# 【对比传统方式】
#   传统：raise FileNotFoundError("文件不存在") → Agent 崩溃或无从下手
#   自愈：return ToolError(recovery_suggestion="先调用 list_files() 查看文件列表")
#         → Agent 知道下一步该怎么做，自动恢复
#
# 【设计原则】
#   - 每种错误都要问自己："Agent 拿到这个错误，知道该怎么办吗？"
#   - recovery_suggestion 要具体到"调用哪个工具 + 传什么参数"
#   - 不要用晦涩的技术错误码，用自然语言描述
# ============================================================

from pydantic import BaseModel
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate


# ============================================================
# 通用返回类型
#
# 为什么不直接用 dict？
#   - BaseModel 有字段验证，防止忘记填 recovery_suggestion
#   - __str__ 方法让 LLM 收到的文本结构清晰
#   - 代码可读性更好，一眼看出"这是成功 / 失败"
# ============================================================
class ToolSuccess(BaseModel):
    """工具执行成功"""
    success: bool = True
    data: Any

    def __str__(self) -> str:
        return f"执行成功: {self.data}"


class ToolError(BaseModel):
    """
    工具执行失败（包含修复建议）

    字段说明：
    - error：错误描述，用自然语言，避免技术术语
    - recovery_suggestion：具体的修复建议，要说明"调用哪个工具"或"修改什么参数"
    """
    success: bool = False
    error: str
    recovery_suggestion: str

    def __str__(self) -> str:
        return f"执行失败: {self.error}\n建议: {self.recovery_suggestion}"


# ============================================================
# 模拟文件系统
# ============================================================
MOCK_FILES = {
    "report_2024.pdf": {"owner": "alice", "size_kb": 1024, "type": "pdf"},
    "data.csv":        {"owner": "bob",   "size_kb": 512,  "type": "csv"},
    "public_readme.txt": {"owner": "public", "size_kb": 8, "type": "txt"},
}

CURRENT_USER = "alice"


# ============================================================
# 工具定义：带自愈能力的文件操作
# ============================================================
@tool
def list_files() -> str:
    """
    列出当前所有可用的文件及其基本信息。

    使用时机：
    - 不确定文件名是否存在时，先调用此工具确认
    - 遇到"文件不存在"错误后，调用此工具查看正确的文件名
    """
    lines = ["当前可用文件："]
    for name, info in MOCK_FILES.items():
        lines.append(f"  - {name}（所有者：{info['owner']}，大小：{info['size_kb']}KB）")
    return "\n".join(lines)


@tool
def get_file_permissions(file_id: str) -> str:
    """
    查询指定文件的权限信息，判断当前用户是否有操作权限。

    使用时机：
    - 遇到"权限不足"错误后，用此工具确认文件所有者
    - 需要判断当前用户能否操作某个文件时

    参数：file_id - 要查询的文件名，如 "report_2024.pdf"
    """
    if file_id not in MOCK_FILES:
        return str(ToolError(
            error=f"文件 '{file_id}' 不存在，无法查询权限",
            recovery_suggestion="请先调用 list_files() 查看存在的文件列表",
        ))

    info = MOCK_FILES[file_id]
    can_operate = info["owner"] in (CURRENT_USER, "public")
    return (
        f"文件 '{file_id}' 权限信息：\n"
        f"  所有者：{info['owner']}\n"
        f"  当前用户：{CURRENT_USER}\n"
        f"  是否有权删除：{'是' if can_operate else '否（非文件所有者）'}"
    )


@tool
def delete_file(file_id: str) -> str:
    """
    删除指定文件。只能删除自己拥有的文件或公开文件（owner='public'）。

    如果失败，返回的错误信息会包含具体的恢复建议，请仔细阅读并按建议重试。

    参数：file_id - 要删除的文件名，如 "report_2024.pdf"
    """
    # 场景 1：文件不存在
    if file_id not in MOCK_FILES:
        return str(ToolError(
            error=f"文件 '{file_id}' 不存在",
            recovery_suggestion="请先调用 list_files() 查看所有可用文件，再用正确的文件名重试",
        ))

    # 场景 2：权限不足
    info = MOCK_FILES[file_id]
    if info["owner"] not in (CURRENT_USER, "public"):
        return str(ToolError(
            error=f"权限不足：'{file_id}' 属于用户 '{info['owner']}'，当前用户 '{CURRENT_USER}' 无权删除",
            recovery_suggestion=(
                f"可调用 get_file_permissions('{file_id}') 查看详细权限，"
                f"或联系文件所有者 '{info['owner']}'"
            ),
        ))

    # 成功删除
    del MOCK_FILES[file_id]
    return str(ToolSuccess(data=f"文件 '{file_id}' 已成功删除"))


@tool
def read_file(file_id: str) -> str:
    """
    读取并返回指定文件的内容。

    参数：file_id - 要读取的文件名

    使用时机：需要查看文件内容时调用（先确认文件存在）
    """
    if file_id not in MOCK_FILES:
        return str(ToolError(
            error=f"文件 '{file_id}' 不存在",
            recovery_suggestion="请先调用 list_files() 查看当前存在的文件",
        ))
    info = MOCK_FILES[file_id]
    return f"文件 '{file_id}' 内容（{info['size_kb']}KB）：[模拟内容，类型 {info['type']}]"


# ============================================================
# 演示：Agent 如何根据 recovery_suggestion 自动恢复
# ============================================================
def run_demo():
    model = ChatDeepSeek(
        model="deepseek-chat",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        temperature=0,
    )

    prompt = PromptTemplate.from_template(
        """你是一个文件管理助手。

重要规则：当工具返回"执行失败"时，仔细阅读"建议"字段，按建议调整策略后重试，不要直接放弃。

可用工具：
{tools}

工具名称：{tool_names}

规则：按照 Thought/Action/Action Input/Observation/Final Answer 格式。

Question: {input}
Thought:{agent_scratchpad}"""
    )

    tools_list = [list_files, get_file_permissions, delete_file, read_file]
    agent = create_react_agent(llm=model, tools=tools_list, prompt=prompt)
    executor = AgentExecutor(
        agent=agent, tools=tools_list,
        verbose=True, max_iterations=10, handle_parsing_errors=True,
    )

    # 场景 1：用不存在的文件名 → Agent 读到建议后调用 list_files() 找正确的名字
    print("=" * 60)
    print("【场景 1】用错误文件名删除 → Agent 根据建议自动查找正确文件名")
    print("=" * 60)
    result = executor.invoke({"input": "请帮我删除文件 'old_report.pdf'"})
    print(f"\n最终结果：{result['output']}")

    # 场景 2：删除无权限的文件 → Agent 查询权限后告知用户
    print("\n\n" + "=" * 60)
    print("【场景 2】删除无权限文件 → Agent 查询权限后给出准确说明")
    print("=" * 60)
    result2 = executor.invoke({"input": "请帮我删除文件 'data.csv'"})
    print(f"\n最终结果：{result2['output']}")


if __name__ == "__main__":
    run_demo()

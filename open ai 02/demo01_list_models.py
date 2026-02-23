import os
import sys
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# 列出可用模型
models = client.models.list()

print("返回类型:", type(models))
print()
print("原始 JSON:")
print(models.model_dump_json(indent=2))

"""测试 ① 规划层：本地 Qwen 能否产出合法 pick+place_on 计划。"""
import json
from llm_backend import make_llm_backend
from planner import Planner

llm = make_llm_backend("local")
pl = Planner(llm)

task = "stack the green block on the yellow block"
objs = ["green_block", "yellow_block", "blue_block"]
plan, meta = pl.plan(task, objs)
print("=== source:", meta["source"], "===")
print("--- raw LLM output (head) ---")
print(meta["raw"][:600])
print("--- parsed plan ---")
print(json.dumps(plan, indent=2, ensure_ascii=False))

# 校验
assert len(plan) == 2, f"期望2步，得到{len(plan)}"
assert plan[0]["name"] == "pick" and plan[0]["args"]["obj"] == "green_block"
assert plan[1]["name"] == "place_on"
assert plan[1]["args"]["obj"] == "green_block" and plan[1]["args"]["dst"] == "yellow_block"
assert "stacked_on(green_block,yellow_block)" in plan[1]["add"]
print("\n*** 规划层断言通过 ***")

"""① LLM 任务规划：动作序列 + ⟨pre, add, del⟩ 符号效应
=========================================================

对应 CABTO 论文的高层符号规划。给 LLM：
  - 任务自然语言目标（如 "stack the green block on the yellow block"）
  - 场景物体列表
  - 可用 skill（带签名）
  - 谓词词汇表（沿用 exp2：on_table/in_gripper/lifted/stacked_on/clear/holding_nothing）
  - （可选）上一轮失败的反馈，用于 self-correction 重规划

让 LLM 输出一个 JSON 动作序列，每步含 name/args 和 ⟨pre, add, del⟩ 效应。

鲁棒性（本地 3B 模型规划能力弱）：
  - 强约束 prompt + few-shot 模板。
  - 严格 JSON 解析 + schema 校验（兼容 markdown code fence / 尾部截断）。
  - 规则 fallback：解析失败或 schema 不合法时，用确定性规则按目标拼出合法计划，
    保证闭环不被"模型抽风"卡死（论文里 LLM 规划本就允许回退/重试）。
"""

from __future__ import annotations

import json
import re


# 可用 skill 及其符号效应模板（args 用占位符 {obj}/{dst}）
SKILLS = {
    "pick": {
        "args": ["obj"],
        "pre":  ["on_table({obj})", "clear({obj})", "holding_nothing()"],
        "add":  ["in_gripper({obj})", "lifted({obj})"],
        "del":  ["on_table({obj})", "holding_nothing()"],
    },
    "place_on": {
        "args": ["obj", "dst"],
        "pre":  ["in_gripper({obj})", "clear({dst})"],
        # 叠放后 obj 既不在桌面(on_table=False)、其底面仍高于桌面(lifted 仍可能 True)，
        # 故效应只断言确实成立的谓词：obj 叠在 dst 上、手已空、dst 顶被占。
        "add":  ["stacked_on({obj},{dst})", "holding_nothing()"],
        "del":  ["in_gripper({obj})", "clear({dst})"],
    },
}

PREDICATES = ["on_table(x)", "in_gripper(x)", "lifted(x)",
              "stacked_on(a,b)", "clear(x)", "holding_nothing()"]


SYSTEM_PROMPT = """You are a symbolic task planner for a Franka robot arm doing tabletop manipulation.
Output ONLY a JSON array of action steps. No prose, no markdown.

Available skills (STRIP-style operators):
- pick(obj): grasp a clear object from the table.
    pre = [on_table(obj), clear(obj), holding_nothing()]
    add = [in_gripper(obj), lifted(obj)]
    del = [on_table(obj), holding_nothing()]
- place_on(obj, dst): place the held object onto another object.
    pre = [in_gripper(obj), clear(dst)]
    add = [stacked_on(obj,dst), holding_nothing()]
    del = [in_gripper(obj), clear(dst)]

Each step is an object:
{"name": "pick", "args": {"obj": "green_block"}, "pre": [...], "add": [...], "del": [...]}

Rules:
- Fill pre/add/del by substituting the operator templates above with the actual args.
- The plan must be a valid sequence: each step's pre must hold after previous steps' effects.
- Output a JSON array only."""

FEWSHOT = """Example.
Task: stack the green block on the yellow block.
Objects: green_block, yellow_block, blue_block.
Output:
[
  {"name": "pick", "args": {"obj": "green_block"},
   "pre": ["on_table(green_block)", "clear(green_block)", "holding_nothing()"],
   "add": ["in_gripper(green_block)", "lifted(green_block)"],
   "del": ["on_table(green_block)", "holding_nothing()"]},
  {"name": "place_on", "args": {"obj": "green_block", "dst": "yellow_block"},
   "pre": ["in_gripper(green_block)", "clear(yellow_block)"],
   "add": ["stacked_on(green_block,yellow_block)", "holding_nothing()"],
   "del": ["in_gripper(green_block)", "clear(yellow_block)"]}
]"""


class Planner:
    def __init__(self, llm_backend):
        self.llm = llm_backend

    def plan(self, task: str, objects: list[str], feedback: str | None = None,
             max_tokens: int = 1024) -> tuple[list[dict], dict]:
        """返回 (plan, meta)。plan 是动作步骤列表；meta 记录来源 source=llm|fallback。"""
        user = self._build_user_prompt(task, objects, feedback)
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": FEWSHOT + "\n\n" + user}]
        raw = ""
        try:
            raw = self.llm.chat(messages, max_tokens=max_tokens, temperature=0.0)
        except Exception as e:
            raw = f"<llm error: {e}>"
        plan = _parse_plan(raw)
        plan = [s for s in plan if _valid_step(s)]
        if plan and self._plan_consistent(plan):
            return self._normalize(plan), {"source": "llm", "raw": raw}
        # fallback：规则规划，保证闭环可继续
        fb = self._rule_fallback(task, objects)
        return fb, {"source": "fallback", "raw": raw}

    def _build_user_prompt(self, task, objects, feedback):
        s = (f"Task: {task}.\n"
             f"Objects: {', '.join(objects)}.\n")
        if feedback:
            s += ("\nThe previous plan FAILED. Feedback from visual effect check:\n"
                  f"{feedback}\n"
                  "Revise the plan to fix this. Output the corrected JSON array.\n")
        s += "Output:"
        return s

    def _plan_consistent(self, plan):
        """轻量一致性检查：模拟 add/del，看每步 pre 是否被满足（初始假设都 on_table/clear）。"""
        objs = set()
        for st in plan:
            objs |= set(st.get("args", {}).values())
        state = set()
        for o in objs:
            state.add(f"on_table({o})")
            state.add(f"clear({o})")
        state.add("holding_nothing()")
        for st in plan:
            for p in st.get("pre", []):
                if p not in state:
                    return False
            for d in st.get("del", []):
                state.discard(d)
            for a in st.get("add", []):
                state.add(a)
        return True

    def _normalize(self, plan):
        out = []
        for st in plan:
            out.append({
                "name": st["name"],
                "args": st["args"],
                "pre": st.get("pre", []),
                "add": st.get("add", []),
                "del": st.get("del", []),
            })
        return out

    def _rule_fallback(self, task, objects):
        """从任务文本里抠出 (obj, dst) 拼 pick+place_on；抠不出则取前两个物体。"""
        obj, dst = self._extract_pair(task, objects)
        return [
            _instantiate("pick", {"obj": obj}),
            _instantiate("place_on", {"obj": obj, "dst": dst}),
        ]

    @staticmethod
    def _extract_pair(task, objects):
        t = task.lower()
        found = [o for o in objects if o.replace("_", " ") in t or o in t]
        # "stack A on B" / "place A on B"
        m = re.search(r"(\w+(?:_\w+)?)\s+on(?:to)?\s+(\w+(?:_\w+)?)", t)
        if m:
            a = _match_obj(m.group(1), objects)
            b = _match_obj(m.group(2), objects)
            if a and b:
                return a, b
        if len(found) >= 2:
            return found[0], found[1]
        if len(objects) >= 2:
            return objects[0], objects[1]
        return objects[0], objects[0]


def _instantiate(skill_name, args):
    sk = SKILLS[skill_name]
    def sub(lst):
        return [s.format(**args) for s in lst]
    return {"name": skill_name, "args": args,
            "pre": sub(sk["pre"]), "add": sub(sk["add"]), "del": sub(sk["del"])}


def _match_obj(word, objects):
    word = word.strip().lower()
    for o in objects:
        if o.lower() == word or o.lower().startswith(word) or word in o.lower():
            return o
    return None


def _valid_step(st):
    if not isinstance(st, dict):
        return False
    if st.get("name") not in SKILLS:
        return False
    args = st.get("args")
    if not isinstance(args, dict):
        return False
    need = SKILLS[st["name"]]["args"]
    return all(k in args for k in need)


def _parse_plan(text: str):
    """从 LLM 输出抠出 JSON 动作数组，兼容 code fence / 尾部截断。"""
    if not text:
        return []
    cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
    # 1) 直接 parse
    for cand in (cleaned, _first_array(cleaned)):
        if not cand:
            continue
        try:
            obj = json.loads(cand)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict) and "plan" in obj:
                return obj["plan"]
        except Exception:
            continue
    return []


def _first_array(s):
    i = s.find("[")
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "[":
            depth += 1
        elif s[j] == "]":
            depth -= 1
            if depth == 0:
                return s[i:j + 1]
    return None

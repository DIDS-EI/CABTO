"""① 双臂 LLM 任务规划：动作序列 + ⟨pre, add, del⟩ 符号效应
=============================================================
stage3b_dual_cabto 的高层符号规划，覆盖三个双臂协作任务：
  - handover : 左臂抓盒 → 空中交接给右臂 → 右臂放到 -y 目标位
  - pour     : 左臂抓罐 → 移到右区杯上方 → 倾倒使球入杯 → 放回罐
  - storage  : 双臂把 item 放入 carton → 右臂抓 carton 提手 → 搬到 shelf

每个 skill 带 ⟨pre, add, del⟩，动作显式标注使用的机械臂（arm=_L/_R），
使规划出的序列能被 dual_codegen 直接翻译成对两条 ArmSkills 的调用。

与单臂 planner 同构：Planner.plan(task, objects, feedback) -> (plan, meta)。
本地小模型规划能力弱，故规则 fallback 是主力（尤其 oracle 后端只需正确的动作
序列即可跑通），LLM 输出仅在通过一致性校验时采用。
"""
from __future__ import annotations

import json
import re


# 双臂 skill 及其符号效应模板。args 占位符：{obj}/{dst}/{arm}/{cup}/{carton}
SKILLS = {
    # ---- handover ----
    "pick_arm": {
        "args": ["obj", "arm"],
        "pre":  ["holding_nothing({arm})"],
        "add":  ["held({obj},{arm})"],
        "del":  ["holding_nothing({arm})"],
    },
    "handover": {
        # 左臂把手中盒举到交接点，右臂接管（确定性空中焊接）。
        "args": ["obj"],
        "pre":  ["held({obj},_L)"],
        "add":  ["handed_over({obj})", "held({obj},_R)"],
        "del":  ["held({obj},_L)"],
    },
    "place_to": {
        # 右臂把手中盒放到 -y 目标位。
        "args": ["obj", "arm"],
        "pre":  ["held({obj},{arm})"],
        "add":  ["at_place({obj})", "holding_nothing({arm})"],
        "del":  ["held({obj},{arm})"],
    },
    # ---- pour ----
    "pick_can": {
        "args": ["obj", "arm"],
        "pre":  ["holding_nothing({arm})"],
        "add":  ["held({obj},{arm})"],
        "del":  ["holding_nothing({arm})"],
    },
    "pour_into": {
        # 持罐臂移到杯上方并翻罐倾倒，使 ball 落入 cup。
        "args": ["can", "ball", "cup", "arm"],
        "pre":  ["held({can},{arm})"],
        "add":  ["poured({can})", "in_cup({ball},{cup})"],
        "del":  [],
    },
    "put_back_can": {
        "args": ["can", "arm"],
        "pre":  ["held({can},{arm})"],
        "add":  ["holding_nothing({arm})"],
        "del":  ["held({can},{arm})"],
    },
    # ---- storage ----
    "pack_item": {
        # 用指定臂抓 item 放入 carton。
        "args": ["item", "arm"],
        "pre":  ["holding_nothing({arm})"],
        "add":  ["in_carton({item})", "holding_nothing({arm})"],
        "del":  [],
    },
    "carry_to_shelf": {
        # 右臂抓 carton 提手搬到 shelf。
        "args": ["carton", "arm"],
        "pre":  ["in_carton(item_g1)", "holding_nothing({arm})"],
        "add":  ["on_shelf({carton})", "holding_nothing({arm})"],
        "del":  [],
    },
}

PREDICATES = [
    "holding_nothing(arm)", "held(obj,arm)",
    "handed_over(box)", "at_place(box)",
    "poured(can)", "in_cup(ball,cup)",
    "in_carton(item)", "on_shelf(carton)",
]


SYSTEM_PROMPT = """You are a symbolic task planner for a DUAL-arm Franka robot (left arm _L, right arm _R) doing tabletop manipulation.
Output ONLY a JSON array of action steps. No prose, no markdown.

Available dual-arm skills (STRIP-style operators). {arm} is "_L" or "_R":
- pick_arm(obj, arm): the given arm grasps obj.
    pre=[holding_nothing(arm)] add=[held(obj,arm)] del=[holding_nothing(arm)]
- handover(obj): left arm hands obj over to right arm in mid-air.
    pre=[held(obj,_L)] add=[handed_over(obj), held(obj,_R)] del=[held(obj,_L)]
- place_to(obj, arm): the arm places held obj onto its target spot.
    pre=[held(obj,arm)] add=[at_place(obj), holding_nothing(arm)] del=[held(obj,arm)]
- pick_can(obj, arm): the arm grasps a can.
    pre=[holding_nothing(arm)] add=[held(obj,arm)] del=[holding_nothing(arm)]
- pour_into(can, ball, cup, arm): the arm tips the can over the cup so ball falls in.
    pre=[held(can,arm)] add=[poured(can), in_cup(ball,cup)] del=[]
- put_back_can(can, arm): the arm puts the can back on the table.
    pre=[held(can,arm)] add=[holding_nothing(arm)] del=[held(can,arm)]
- pack_item(item, arm): the arm places item into the carton.
    pre=[holding_nothing(arm)] add=[in_carton(item), holding_nothing(arm)] del=[]
- carry_to_shelf(carton, arm): right arm carries the carton by its handle to the shelf.
    pre=[in_carton(item_g1), holding_nothing(arm)] add=[on_shelf(carton), holding_nothing(arm)] del=[]

Each step: {"name": "...", "args": {...}, "pre": [...], "add": [...], "del": [...]}
Fill pre/add/del by substituting the operator template with the actual args.
Output a JSON array only."""

FEWSHOT = """Example 1.
Task: hand the box from the left arm over to the right arm and place it on the right side.
Objects: box0.
Output:
[
  {"name": "pick_arm", "args": {"obj": "box0", "arm": "_L"},
   "pre": ["holding_nothing(_L)"], "add": ["held(box0,_L)"], "del": ["holding_nothing(_L)"]},
  {"name": "handover", "args": {"obj": "box0"},
   "pre": ["held(box0,_L)"], "add": ["handed_over(box0)", "held(box0,_R)"], "del": ["held(box0,_L)"]},
  {"name": "place_to", "args": {"obj": "box0", "arm": "_R"},
   "pre": ["held(box0,_R)"], "add": ["at_place(box0)", "holding_nothing(_R)"], "del": ["held(box0,_R)"]}
]

Example 2.
Task: pour the ball from the left can into the right cup.
Objects: canL, ballL, cupR.
Output:
[
  {"name": "pick_can", "args": {"obj": "canL", "arm": "_L"},
   "pre": ["holding_nothing(_L)"], "add": ["held(canL,_L)"], "del": ["holding_nothing(_L)"]},
  {"name": "pour_into", "args": {"can": "canL", "ball": "ballL", "cup": "cupR", "arm": "_L"},
   "pre": ["held(canL,_L)"], "add": ["poured(canL)", "in_cup(ballL,cupR)"], "del": []},
  {"name": "put_back_can", "args": {"can": "canL", "arm": "_L"},
   "pre": ["held(canL,_L)"], "add": ["holding_nothing(_L)"], "del": ["held(canL,_L)"]}
]

Example 3.
Task: put the item into the carton and store the carton on the shelf.
Objects: item_g1, carton.
Output:
[
  {"name": "pack_item", "args": {"item": "item_g1", "arm": "_L"},
   "pre": ["holding_nothing(_L)"], "add": ["in_carton(item_g1)", "holding_nothing(_L)"], "del": []},
  {"name": "carry_to_shelf", "args": {"carton": "carton", "arm": "_R"},
   "pre": ["in_carton(item_g1)", "holding_nothing(_R)"], "add": ["on_shelf(carton)", "holding_nothing(_R)"], "del": []}
]"""


class Planner:
    def __init__(self, llm_backend):
        self.llm = llm_backend

    def plan(self, task: str, objects: list[str], feedback: str | None = None,
             max_tokens: int = 1024) -> tuple[list[dict], dict]:
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
        if plan:
            return self._normalize(plan), {"source": "llm", "raw": raw}
        fb = self._rule_fallback(task, objects)
        return fb, {"source": "fallback", "raw": raw}

    def _build_user_prompt(self, task, objects, feedback):
        s = f"Task: {task}.\nObjects: {', '.join(objects)}.\n"
        if feedback:
            s += ("\nThe previous plan FAILED. Feedback from ground-truth effect check:\n"
                  f"{feedback}\n"
                  "Revise the plan to fix this. Output the corrected JSON array.\n")
        s += "Output:"
        return s

    def _normalize(self, plan):
        out = []
        for st in plan:
            out.append({"name": st["name"], "args": st["args"],
                        "pre": st.get("pre", []), "add": st.get("add", []),
                        "del": st.get("del", [])})
        return out

    def _rule_fallback(self, task, objects):
        """规则规划兜底：按任务文本识别 handover / pour / storage 并拼合法序列。"""
        t = task.lower()

        # storage：入箱 + 上架
        if any(k in t for k in ("carton", "shelf", "store", "box into", "pack")):
            steps = [_instantiate("pack_item", {"item": "item_g1", "arm": "_L"})]
            steps.append(_instantiate("carry_to_shelf", {"carton": "carton", "arm": "_R"}))
            return steps

        # pour：倒球入杯
        if any(k in t for k in ("pour", "cup", "ball into", "tip")):
            steps = [
                _instantiate("pick_can", {"obj": "canL", "arm": "_L"}),
                _instantiate("pour_into", {"can": "canL", "ball": "ballL",
                                            "cup": "cupR", "arm": "_L"}),
                _instantiate("put_back_can", {"can": "canL", "arm": "_L"}),
            ]
            return steps

        # handover（默认）：交接 + 放置
        box = next((o for o in objects if o.startswith("box")), "box0")
        return [
            _instantiate("pick_arm", {"obj": box, "arm": "_L"}),
            _instantiate("handover", {"obj": box}),
            _instantiate("place_to", {"obj": box, "arm": "_R"}),
        ]


def _instantiate(skill_name, args):
    sk = SKILLS[skill_name]

    def sub(lst):
        return [s.format(**args) for s in lst]

    return {"name": skill_name, "args": args,
            "pre": sub(sk["pre"]), "add": sub(sk["add"]), "del": sub(sk["del"])}


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
    if not text:
        return []
    cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
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


if __name__ == "__main__":
    p = Planner(llm_backend=None)
    for task, objs in [
        ("hand the box over from left to right and place it on the right side", ["box0"]),
        ("pour the ball from the left can into the right cup", ["canL", "ballL", "cupR"]),
        ("put the item into the carton and store it on the shelf",
         ["item_g1", "item_g2", "carton"]),
    ]:
        plan = p._rule_fallback(task, objs)
        print(f"\nTask: {task}")
        for st in plan:
            print(f"  {st['name']}({st['args']})")
            print(f"     pre={st['pre']} add={st['add']} del={st['del']}")

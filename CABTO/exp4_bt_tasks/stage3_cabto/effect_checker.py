"""D. ④ VLM 视觉校验效应
==========================
EffectChecker(vlm_backend).check(before_img, after_img, step) -> dict
把 before/after 两张图喂 VLM，针对 step 的 add/del 谓词逐条问「该效应是否发生」，
为控制 VLM 调用次数，把多个谓词合并成 **一个问题**，让 VLM 一次性回 JSON。
同时调 world_state.check_effect 拿 ground-truth，便于对比 VLM 判断与真值是否一致。

返回 {"vlm_ok", "vlm_violations", "vlm_raw", "gt_ok", "gt_violations",
       "agree"(vlm_ok==gt_ok)}。
"""

from __future__ import annotations

import json
import re

import numpy as np

import world_state as ws


def _pred_to_question(pred):
    """把谓词翻成自然语言判断句。"""
    m = re.match(r'stacked_on\((\w+),\s*(\w+)\)', pred)
    if m:
        a = m.group(1).split("_")[0]
        b = m.group(2).split("_")[0]
        return f"the {a} block is now stacked on top of the {b} block"
    m = re.match(r'on_table\((\w+)\)', pred)
    if m:
        return f"the {m.group(1).split('_')[0]} block is resting on the table"
    m = re.match(r'in_gripper\((\w+)\)', pred)
    if m:
        return f"the {m.group(1).split('_')[0]} block is held by the gripper"
    m = re.match(r'lifted\((\w+)\)', pred)
    if m:
        return f"the {m.group(1).split('_')[0]} block is lifted off the table"
    m = re.match(r'clear\((\w+)\)', pred)
    if m:
        return f"the top of the {m.group(1).split('_')[0]} block is clear (nothing on it)"
    if pred.startswith("holding_nothing"):
        return "the gripper is empty (holding nothing)"
    return pred


def _parse_yesno_json(text, keys):
    """解析 VLM 返回的 {key: "yes"/"no"}。兼容 code fence / 散落 yes-no。
    返回 dict[key->bool]，无法判定的 key 缺省 None。"""
    res = {k: None for k in keys}
    if not text:
        return res
    cleaned = re.sub(r'```(?:json)?|```', '', text).strip()
    # 1) 尝试整体 JSON
    obj = None
    try:
        obj = json.loads(cleaned)
    except Exception:
        m = re.search(r'\{.*\}', cleaned, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if isinstance(obj, dict):
        # 按位置/编号匹配
        for i, k in enumerate(keys):
            for cand in (k, f"q{i+1}", f"Q{i+1}", str(i + 1), f"pred{i+1}"):
                if cand in obj:
                    res[k] = _truthy(obj[cand])
                    break
    # 2) 兜底：若仍有 None，按出现顺序抓 yes/no
    if any(v is None for v in res.values()):
        tokens = re.findall(r'\b(yes|no|true|false)\b', cleaned.lower())
        ti = 0
        for k in keys:
            if res[k] is None and ti < len(tokens):
                res[k] = tokens[ti] in ("yes", "true")
                ti += 1
    return res


def _truthy(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    s = str(v).strip().lower()
    return s in ("yes", "true", "1", "y")


class EffectChecker:
    def __init__(self, vlm_backend):
        self.vlm = vlm_backend

    def check(self, before_img, after_img, step, env=None, state_before=None,
              use_vlm=True):
        add = step.get("add", [])
        del_ = step.get("del", [])

        # ---------- ground-truth 路 ----------
        gt_ok, gt_violations = (None, [])
        if env is not None:
            gt_ok, gt_violations = ws.check_effect(env, add, del_, state_before)

        # ---------- VLM 路 ----------
        vlm_ok, vlm_violations, vlm_raw = None, [], ""
        if use_vlm:
            # 只校验 add 谓词中「视觉可判」的核心项（stacked_on / lifted / on_table），
            # 合并成一个问题一次性问，控制调用次数（一步 1 次）。
            checkable = [p for p in add
                         if p.split("(")[0] in ("stacked_on", "lifted", "on_table")]
            if not checkable:
                checkable = add[:2]
            qs = [_pred_to_question(p) for p in checkable]
            qlist = "\n".join(f'  "q{i+1}": is it true that {q}?'
                              for i, q in enumerate(qs))
            prompt = (
                "You are given TWO images: the FIRST is BEFORE an action, the "
                "SECOND is AFTER the action. Compare them and answer whether each "
                "statement about the AFTER image is true.\n"
                f"{qlist}\n"
                'Respond ONLY with a JSON object like {"q1":"yes","q2":"no"}. '
                "Use lowercase yes or no."
            )
            try:
                vlm_raw = self.vlm.chat_vision(prompt, [before_img, after_img],
                                               max_tokens=128, temperature=0.0)
            except Exception as e:
                vlm_raw = f"<vlm error: {e}>"
            ans = _parse_yesno_json(vlm_raw, checkable)
            for p in checkable:
                if ans.get(p) is False:
                    vlm_violations.append(f"VLM 判定未发生: {p}")
                elif ans.get(p) is None:
                    vlm_violations.append(f"VLM 无法判定: {p}")
            vlm_ok = len(vlm_violations) == 0

        agree = None
        if vlm_ok is not None and gt_ok is not None:
            agree = (vlm_ok == gt_ok)

        return {
            "vlm_ok": vlm_ok,
            "vlm_violations": vlm_violations,
            "vlm_raw": vlm_raw,
            "gt_ok": gt_ok,
            "gt_violations": gt_violations,
            "agree": agree,
        }

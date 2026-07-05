"""④ 双臂效应校验（ground-truth 权威 + 可选 VLM 记录）
=======================================================
stage3b 用 oracle 后端：失败判定权威 = ground-truth（dual_world_state.check_effect）。
VLM 视觉校验作为对比记录（本阶段默认关闭，接口保留，便于后续对齐 stage3 的 VLM 路）。

check(step, env, state_before, before_img=None, after_img=None, use_vlm=False) -> dict
返回 {gt_ok, gt_violations, vlm_ok, vlm_violations, vlm_raw, agree}。
"""
from __future__ import annotations

import dual_world_state as WS


class DualEffectChecker:
    def __init__(self, vlm_backend=None):
        self.vlm = vlm_backend

    def check(self, step, env, state_before=None, before_img=None,
              after_img=None, use_vlm=False):
        add = step.get("add", [])
        del_ = step.get("del", [])

        gt_ok, gt_violations = WS.check_effect(env, add, del_, state_before)

        vlm_ok, vlm_violations, vlm_raw = None, [], ""
        if use_vlm and self.vlm is not None and before_img is not None:
            # 记录用途：把 add 谓词合并成一句问 VLM（不参与失败判定）。
            qs = "; ".join(add[:3])
            prompt = ("Two images: BEFORE then AFTER a dual-arm robot action. "
                      f"Did these become true: {qs}? Answer yes or no.")
            try:
                vlm_raw = self.vlm.chat_vision(prompt, [before_img, after_img],
                                               max_tokens=64, temperature=0.0)
                vlm_ok = "yes" in vlm_raw.lower()
            except Exception as e:
                vlm_raw = f"<vlm error: {e}>"
            if vlm_ok is False:
                vlm_violations = [f"VLM 记录: 未确认 {qs}"]

        agree = (vlm_ok == gt_ok) if (vlm_ok is not None and gt_ok is not None) else None
        return {"gt_ok": gt_ok, "gt_violations": gt_violations,
                "vlm_ok": vlm_ok, "vlm_violations": vlm_violations,
                "vlm_raw": vlm_raw, "agree": agree}

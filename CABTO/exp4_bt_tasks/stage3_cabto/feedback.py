"""只整理观测和待检验假设；有限失败不构成病因证明。仅依赖标准库。"""

from copy import deepcopy
import hashlib
import json


def build_failure_feedback(step, state_before, state_after, exec_log=None,
                           exception=None, gt_violations=None, images=None,
                           program_text=None, pointer_provenance=None):
    """输入状态是谓词真值字典；images 是图像路径，不从图像臆造事实。

    exception 可为异常对象或已序列化的 type/message 字典。
    不依据成功/失败比例归因，不自动修订符号模型或执行策略。
    """
    if isinstance(exception, BaseException):
        exception = {"type": type(exception).__name__,
                     "module": type(exception).__module__, "message": str(exception)}
    before = deepcopy(state_before or {})
    after = deepcopy(state_after or {})
    preconditions = {p: before.get(p) for p in step.get("pre", [])}
    effects = [{"predicate": p, "expected": expected, "actual": after.get(p)}
               for kind, expected in (("add", True), ("del", False))
               for p in step.get(kind, [])]
    facts = {
        "step": deepcopy(step), "state_before": before, "state_after": after,
        "preconditions": preconditions, "effects": effects,
        "exec_log": deepcopy(exec_log), "exception": deepcopy(exception),
        "gt_violations": list(gt_violations or []),
        "images": deepcopy(images), "images_interpreted": False,
        "program_text": program_text,
        "program_sha256": (hashlib.sha256(program_text.encode("utf-8")).hexdigest()
                           if program_text is not None else None),
        "pointer_provenance": deepcopy(pointer_provenance),
    }
    hypotheses = [
        {"id": "policy_implementation", "status": "unresolved",
         "description": "实现、参数或原语串接可能有误；尚未定位。"},
        {"id": "perception_or_localization", "status": "unresolved",
         "description": "需核对定位来源、坐标转换及 oracle 回退，不能单凭失败排除或确认。"},
        {"id": "execution_or_control", "status": "unresolved",
         "description": "控制、接触或执行状态可能影响效应；不等同于原语表达能力不足。"},
        {"id": "state_or_preconditions", "status": "unresolved",
         "description": "核对已有前提的实际真值；混合成败不能证明缺少前提。"},
        {"id": "symbolic_model", "status": "unresolved",
         "description": "符号效应或前提模型可能不符，需独立证据后才提出修订。"},
    ]
    routes = [
        {"target": "planner", "reason": "依据已观测状态和目标重新规划，不把假设写成事实。"},
        {"target": "policy", "reason": "审查实际执行源码与调用参数；当前仅输出提示。"},
        {"target": "perception", "reason": "核对定位与回退来源，保留其他病因。"},
        {"target": "model_review", "reason": "补足区分性证据前不改动作模型。"},
    ]
    next_tests = [
        {"target": "policy", "test": "对照实际源码、签名和 exec_log，定位最后确认完成的调用。"},
        {"target": "perception", "test": "对照 before/after 图像、定位坐标及来源，检查目标与坐标系。"},
        {"target": "runtime", "test": "在相同状态和参数下复核执行日志，区分定位、实现与控制失败。"},
        {"target": "model_review", "test": "保持其他因素不变，检验候选前提或效应；禁止按全败/混合成败直接归因。"},
    ]
    if exception is not None:
        kind = exception.get("type", "Exception")
        syntax = kind in ("SyntaxError", "IndentationError", "TabError")
        hypotheses.insert(0, {
            "id": "observed_syntax_error" if syntax else "observed_runtime_exception",
            "status": "observed", "exception_type": kind,
            "description": "异常本身已观测；不是其他病因不存在的证明。",
        })
        routes.insert(0, {"target": "runtime", "reason": f"已观测 {kind}，优先核对异常位置与 API 调用。"})
        next_tests.insert(0, {"target": "policy" if syntax else "runtime",
                              "test": "先检查异常堆栈及源码的语法/API 契约，再检验其余并存假设。"})

    policy_evidence = {k: facts[k] for k in
                       ("step", "exception", "exec_log", "gt_violations", "preconditions",
                        "effects", "images", "program_sha256", "pointer_provenance")}
    policy_feedback = (
        "这是待审查的 policy_feedback，不是已执行的策略修复。\n"
        "当前执行器使用 handwritten_template，没有自动消费此提示；策略生成/修复闭环尚未实现。\n"
        "仅据观测审查实际模板源码及 API 调用，提出可区分病因的下一步测试；"
        "不要把有限全失败判为原语不足，也不要把混合成败判为缺前提。"
        "异常可优先走 policy/runtime，但其他病因仍可共存。"
        "在独立证据不足时保持 unresolved，candidate_model_patch=null。\n"
        + json.dumps(policy_evidence, ensure_ascii=False, separators=(",", ":"))
        + "\n实际执行源码（若可用）\n" + (program_text or "未提供源码；禁止用展示伪代码替代。")
    )
    return {
        "diagnosis_status": "unresolved", "observed_facts": facts,
        "hypotheses": hypotheses, "next_tests": next_tests, "routes": routes,
        "candidate_model_patch": None, "policy_feedback": policy_feedback,
    }

"""正式 BT Expansion 适配与保守模型修订；不修改内核，不执行机器人代码。

入口：ModelLibrary(models, bt_root=None).build(start_true_set, goal_set)。
树的 tick 每次只消费外部观测，返回 (status, 原模型字典的独立副本或 None)。
propose_patch 只支持 pre 严格收紧；validation 的 heldout 身份由采集方声明，
本模块校验角色、代码哈希和样本不复用，不能证明采集独立性或代码物理正确性。
所有回归均为有限符号检验，不能据此声称真机验证或全状态空间保证。
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from urllib.parse import quote


DEFAULT_BT_ROOT = Path(__file__).resolve().parents[3] / "BTExpansion-demo"
SCOPE = "finite_empirical_only"
_IMPORT_LOCK = RLock()
_KERNEL_CACHE = {}
_KERNEL_FILES = {
    "bt_expansion": "src/bt_expansion/__init__.py",
    "bt_expansion.planning": "src/bt_expansion/planning.py",
    "behavior_tree": "src/behavior_tree/__init__.py",
    "behavior_tree.BehaviorTree": "src/behavior_tree/BehaviorTree.py",
    "bt_expansion.algorithm": "src/bt_expansion/algorithm.py",
}


class ModelValidationError(ValueError):
    """模型或观测记录不满足输入约束。"""


class PlanningError(RuntimeError):
    """正式内核无法构造可运行的树。"""


def _symbols(value, field, *, list_only=False):
    kinds = (list,) if list_only else (list, tuple, set, frozenset)
    if not isinstance(value, kinds):
        raise ModelValidationError(f"{field} 必须是{'list' if list_only else '符号集合或列表'}")
    if any(not isinstance(atom, str) or not atom.strip() for atom in value):
        raise ModelValidationError(f"{field} 必须只包含非空符号字符串")
    return set(value)


def action_id(model):
    """按参数键排序的唯一键；转义分隔符，避免不同 grounded 动作碰撞。"""
    if not isinstance(model, dict):
        raise ModelValidationError("动作必须为 dict")
    name, args = model.get("name"), model.get("args")
    if not isinstance(name, str) or not name.strip() or not isinstance(args, dict):
        raise ModelValidationError("动作需要非空 name 和 dict 类型 args")
    if any(not isinstance(k, str) or not k.strip()
           or not isinstance(v, str) or not v.strip() for k, v in args.items()):
        raise ModelValidationError("grounded args 的键和值必须为非空字符串")
    pairs = ",".join(f"{quote(k, safe='_.-')}={quote(v, safe='_.-')}"
                     for k, v in sorted(args.items()))
    return f"{quote(name, safe='_.-')}({pairs})"


def validate_model(model):
    key = action_id(model)
    pre, add, delete = (_symbols(model.get(f), f, list_only=True)
                        for f in ("pre", "add", "del"))
    if pre & add:
        raise ModelValidationError(f"{key}: pre ∩ add 必须为空")
    if add & delete:
        raise ModelValidationError(f"{key}: add ∩ del 必须为空")
    return key


def _model_map(models):
    result = {}
    for model in models:
        key = validate_model(model)
        if key in result:
            raise ModelValidationError(f"重复 action_id: {key}")
        result[key] = copy.deepcopy(model)
    return result


def _load_kernel(bt_root):
    root = Path(bt_root).expanduser().resolve()
    with _IMPORT_LOCK:
        provenance = {}
        for module_name, relative in _KERNEL_FILES.items():
            path = root / relative
            provenance[module_name] = {
                "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            loaded = sys.modules.get(module_name)
            if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() != path:
                raise ImportError(f"{module_name} 已从其他内核路径导入；请使用独立进程切换 bt_root")
        if root in _KERNEL_CACHE:
            algorithm, action, previous = _KERNEL_CACHE[root]
            if previous != provenance:
                raise ImportError("正式内核文件在导入后发生变化；请在独立进程重新加载")
            return algorithm, action, copy.deepcopy(previous)
        old_path = sys.path[:]
        try:
            sys.path.insert(0, str(root / "src"))
            algorithm = importlib.import_module("bt_expansion.algorithm").BTExpAlgorithm
            action = importlib.import_module("bt_expansion.planning").Action
            for module_name, relative in _KERNEL_FILES.items():
                if Path(sys.modules[module_name].__file__).resolve() != root / relative:
                    raise ImportError(f"{module_name} 未从指定 bt_root 加载")
                if hashlib.sha256((root / relative).read_bytes()).hexdigest() != provenance[module_name]["sha256"]:
                    raise ImportError("正式内核在导入期间变化，拒绝记录不一致的哈希")
        finally:
            sys.path[:] = old_path
        _KERNEL_CACHE[root] = (algorithm, action, provenance)
        return algorithm, action, copy.deepcopy(provenance)


class FormalBTBuilder:
    """仅通过正式 Action 与 clear/run_algorithm_selTree 建树。"""

    def __init__(self, models, bt_root=None, *, model_version=0):
        self._models = _model_map(models)
        self.bt_root = Path(bt_root) if bt_root is not None else DEFAULT_BT_ROOT
        self.model_version = model_version

    def build(self, start_true_set, goal_set):
        start = _symbols(start_true_set, "start")
        goal = _symbols(goal_set, "goal")
        algorithm_class, action_class, kernel = _load_kernel(self.bt_root)
        actions = [action_class(name=key, pre=set(model["pre"]),
                                add=set(model["add"]), del_set=set(model["del"]))
                   for key, model in self._models.items()]
        algorithm = algorithm_class()
        algorithm.clear()
        root = algorithm.run_algorithm_selTree(set(start), set(goal), actions)
        if root is False or root is None:
            raise PlanningError("正式 BT 内核无法从该 start 构建目标树")
        return FormalBT(root, self._models, start, goal, kernel, self.model_version)


class FormalBT:
    """无符号预测状态缓存的正式树句柄；运行时 tick 不替代真实观测。"""

    def __init__(self, root, models, start, goal, kernel, model_version):
        self._root = root
        self._models = copy.deepcopy(models)
        self.start = frozenset(start)
        self.goal = frozenset(goal)
        self._kernel = copy.deepcopy(kernel)
        self.model_version = model_version

    @property
    def kernel(self):
        return copy.deepcopy(self._kernel)

    def tick(self, true_state):
        status, selected = self._root.tick(_symbols(true_state, "true_state"))
        if status == "running":
            if selected.name not in self._models:
                raise PlanningError("内核返回未知动作")
            return status, copy.deepcopy(self._models[selected.name])
        if status not in ("success", "failure"):
            raise PlanningError(f"未知内核状态: {status}")
        return status, None

    def export(self, format="json"):
        """自写只读遍历；返回字符串，不调用可能修改动作名的内核导出器。"""
        nodes, edges = [], []
        pending = [(self._root, None)]
        while pending:
            node, parent = pending.pop()
            index = len(nodes)
            item = {"id": index, "type": node.type}
            if hasattr(node, "children"):
                item["kind"] = "control"
                pending.extend((child, index) for child in reversed(node.children))
            elif node.type == "act":
                item.update(kind="action", action_id=node.content.name,
                            model=copy.deepcopy(self._models[node.content.name]))
            else:
                item.update(kind="condition", symbols=sorted(node.content))
            nodes.append(item)
            if parent is not None:
                edges.append([parent, index])
        metadata = {"kernel": self.kernel, "model_version": self.model_version,
                    "start": sorted(self.start), "goal": sorted(self.goal)}
        if format == "json":
            return json.dumps({**metadata, "nodes": nodes, "edges": edges},
                              ensure_ascii=False, indent=2)
        if format == "dot":
            lines = ["digraph FormalBT {", "  // " + json.dumps(metadata, ensure_ascii=True)]
            for node in nodes:
                label = node.get("action_id", " & ".join(node.get("symbols", []))
                                 or ("TRUE" if node["kind"] == "condition" else node["type"]))
                lines.append(f'  n{node["id"]} [label={json.dumps(label, ensure_ascii=True)}];')
            lines.extend(f"  n{a} -> n{b};" for a, b in edges)
            return "\n".join(lines + ["}"])
        raise ValueError("export 仅支持 json 或 dot")


def symbolic_dry_run(bt, start, goal, max_steps=256):
    """只在复制的符号状态上执行 STRIPS 重放，绝不是物理执行证据。"""
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps 必须为正整数")
    state, target = _symbols(start, "start"), _symbols(goal, "goal")
    seen, trace = set(), []
    reason = "超过有限符号步数上限"
    for step in range(max_steps + 1):
        status, model = bt.tick(state)
        if status == "success":
            reason = "符号目标达成" if target <= state else "树报告成功但符号目标未达成"
            return {"reached_goal": target <= state, "reason": reason,
                    "steps": len(trace), "action_ids": trace,
                    "final_state": sorted(state), "scope": "symbolic_only"}
        if status != "running":
            reason = "符号重放中树失败"
            break
        if step == max_steps:
            break
        key = frozenset(state)
        if key in seen:
            reason = "符号重放检测到状态循环"
            break
        seen.add(key)
        if not set(model["pre"]) <= state:
            reason = "内核选择的动作前提不满足"
            break
        trace.append(action_id(model))
        state = (state | set(model["add"])) - set(model["del"])
    return {"reached_goal": False, "reason": reason, "steps": len(trace),
            "action_ids": trace, "final_state": sorted(state), "scope": "symbolic_only"}


def _violations(model, after):
    pre, add, delete = (set(model[f]) for f in ("pre", "add", "del"))
    return {"missing_add": sorted(add - after),
            "remaining_del": sorted(delete & after),
            "missing_preserved_pre": sorted((pre - delete) - after)}


def _check_evidence(key, old, candidate, evidence):
    if not isinstance(evidence, list) or not evidence:
        raise ModelValidationError("证据不足：需要具体 observed transition records")
    required = {"before", "after", "action_id", "code_sha256", "valid_trial",
                "exception", "code_executed", "role"}
    records, hashes, counterexamples, validations = [], set(), set(), set()
    for index, record in enumerate(evidence):
        if not isinstance(record, dict) or not required <= record.keys():
            raise ModelValidationError(f"证据 {index} 缺失必要字段")
        if not isinstance(record["action_id"], str) or not record["action_id"]:
            raise ModelValidationError(f"证据 {index} 的 action_id 无效")
        if type(record["valid_trial"]) is not bool or type(record["code_executed"]) is not bool:
            raise ModelValidationError("valid_trial/code_executed 必须为 bool")
        if record["role"] not in ("counterexample", "validation"):
            raise ModelValidationError("证据 role 必须为 counterexample 或 validation")
        before = _symbols(record["before"], "before")
        after = _symbols(record["after"], "after")
        if record["action_id"] != key:
            continue
        if (not record["valid_trial"] or not record["code_executed"]
                or record["exception"] not in (None, "")):
            # 错误代码、未执行和无效试次不能成为模型修订依据。
            continue
        code_hash = record["code_sha256"]
        if not isinstance(code_hash, str) or re.fullmatch(r"[0-9a-fA-F]{64}", code_hash) is None:
            raise ModelValidationError("有效目标轨迹缺少合法非空 code_sha256")
        code_hash = code_hash.lower()
        hashes.add(code_hash)
        signature = (frozenset(before), frozenset(after), code_hash)
        records.append((index, record["role"], before, after))
        if record["role"] == "counterexample":
            if not set(old["pre"]) <= before or not any(_violations(old, after).values()):
                raise ModelValidationError("标记的 counterexample 并未反驳旧模型")
            counterexamples.add(signature)
        else:
            validations.add(signature)
    if len(hashes) != 1 or not counterexamples or not validations:
        raise ModelValidationError("证据不足：需同一非空代码哈希的有效反例和 heldout validation；代码异常不能归因模型")
    if counterexamples & validations:
        raise ModelValidationError("heldout validation 不得复用反例记录")
    covered, heldout = 0, 0
    for index, role, before, after in records:
        if not set(candidate["pre"]) <= before:
            continue
        covered += 1
        violations = _violations(candidate, after)
        if any(violations.values()):
            raise ModelValidationError(f"候选覆盖的有效轨迹 {index} 违约: {violations}")
        heldout += role == "validation"
    if not heldout:
        raise ModelValidationError("证据不足：候选 pre 必须覆盖至少一个 heldout validation")
    return {"code_sha256": next(iter(hashes)), "valid_target_records": len(records),
            "covered_records": covered, "covered_heldout": heldout,
            "counterexamples": len(counterexamples), "heldout_role_source": "caller_declared"}


@dataclass(frozen=True)
class _LibrarySnapshot:
    version: int
    models: dict
    trees: tuple


class ModelLibrary:
    """模型、版本和树以单个快照原子提交；失败保留原快照与旧树句柄。"""

    def __init__(self, models, bt_root=None, *, max_steps=256):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps 必须为正整数")
        self.bt_root = Path(bt_root) if bt_root is not None else DEFAULT_BT_ROOT
        self.max_steps = max_steps
        self._lock = RLock()
        self._snapshot = _LibrarySnapshot(0, _model_map(models), ())

    @property
    def version(self):
        with self._lock:
            return self._snapshot.version

    @property
    def models(self):
        with self._lock:
            return copy.deepcopy(self._snapshot.models)

    @property
    def trees(self):
        with self._lock:
            return self._snapshot.trees

    def build(self, start_true_set, goal_set):
        with self._lock:
            snapshot = self._snapshot
            tree = FormalBTBuilder(snapshot.models.values(), self.bt_root,
                                   model_version=snapshot.version).build(start_true_set, goal_set)
            self._snapshot = _LibrarySnapshot(snapshot.version, snapshot.models,
                                              snapshot.trees + (tree,))
            return tree

    def propose_patch(self, action_id, replacement, evidence, task_set):
        """评估独立候选，验收所有指定任务及已登记树的任务；不接受更广修订。

        旧树引用保持旧版本；成功后调用方应重新获取 trees 或调用 build。
        task_set 必须非空，每项为 {start, goal}；返回理由和有限经验范围。
        """
        with self._lock:
            snapshot = self._snapshot
            reports = []
            result = {"accepted": False, "scope": SCOPE, "regression": "symbolic_only",
                      "version_before": snapshot.version, "version_after": snapshot.version,
                      "tasks": reports}
            try:
                if action_id not in snapshot.models:
                    raise ModelValidationError("未知 action_id")
                candidate = copy.deepcopy(replacement)
                if validate_model(candidate) != action_id:
                    raise ModelValidationError("replacement 不得改变动作身份")
                old = snapshot.models[action_id]
                if (not set(old["pre"]) <= set(candidate["pre"])
                        or any(set(old[f]) != set(candidate[f]) for f in ("add", "del"))):
                    raise ModelValidationError("仅允许 pre 收紧且 add/del 不变；更广修订需要专项验证")
                if set(old["pre"]) == set(candidate["pre"]):
                    raise ModelValidationError("候选未严格收紧 pre")
                # 元数据可能决定执行代码；不能借修订前提暗改其他字段。
                if ({k: v for k, v in old.items() if k not in ("pre", "add", "del")}
                        != {k: v for k, v in candidate.items() if k not in ("pre", "add", "del")}):
                    raise ModelValidationError("非符号字段变化需要专项验证")
                evidence_report = _check_evidence(action_id, old, candidate, copy.deepcopy(evidence))
                result["evidence"] = evidence_report
                if not isinstance(task_set, (list, tuple)) or not task_set:
                    raise ModelValidationError("task_set 必须为非空任务列表，不能跳过回归")
                tasks = []
                for task in copy.deepcopy(task_set):
                    if not isinstance(task, dict) or not {"start", "goal"} <= task.keys():
                        raise ModelValidationError("每个 task 必须包含 start 和 goal")
                    tasks.append((frozenset(_symbols(task["start"], "task.start")),
                                  frozenset(_symbols(task["goal"], "task.goal"))))
                for tree in snapshot.trees:
                    task = (tree.start, tree.goal)
                    if task not in tasks:
                        tasks.append(task)
                candidate_models = copy.deepcopy(snapshot.models)
                candidate_models[action_id] = candidate
                builder = FormalBTBuilder(candidate_models.values(), self.bt_root,
                                          model_version=snapshot.version + 1)
                for index, (start, goal) in enumerate(tasks):
                    tree = builder.build(start, goal)
                    report = symbolic_dry_run(tree, start, goal, self.max_steps)
                    reports.append({"task_index": index, "start": sorted(start),
                                    "goal": sorted(goal), **report})
                    result["kernel"] = tree.kernel
                    if not report["reached_goal"]:
                        raise PlanningError(f"任务 {index} 未通过有限符号回归: {report['reason']}")
                # 全部验收之后重新建树；任何一次重建失败都不提交。
                fresh_trees = tuple(builder.build(start, goal) for start, goal in tasks)
                next_snapshot = _LibrarySnapshot(snapshot.version + 1, candidate_models, fresh_trees)
                result.update(accepted=True, version_after=next_snapshot.version,
                              reason="同代码有效反例与覆盖的 heldout 通过；pre 保守收紧，全部任务通过有限符号回归并已重建树")
                self._snapshot = next_snapshot
                return result
            except (ValueError, TypeError, RuntimeError, ImportError, OSError) as exc:
                result["reason"] = str(exc)
                return result

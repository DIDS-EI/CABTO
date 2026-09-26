"""把 lifted PDDL 算子 grounding 成 BT Expansion 的 Action 列表。

产出 (goal, start, actions)，与 src/bt_expansion/examples.py 的返回约定完全一致，
因此可以直接喂给 BTExpAlgorithm。

两项非平凡处理：
  1. typing：按 :types 的继承链判断对象是否可代入某个参数位
  2. 负前提/负目标的补谓词编码：为谓词 P 引入 not-P，
     并在所有增删 P 的动作里同步维护，使模型回到"全正文字"，
     这样 BT Expansion 基于集合包含的条件判定才是正确的
"""

from __future__ import annotations

import itertools
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bt_expansion.planning import Action
from .parser import Atom, Formula
from .compat import collect_atoms
from .cond_effects import compile_conditional_effects, split_effect


NEG_PREFIX = 'not-'


# ------------------------------------------------------------------ typing

def build_type_closure(types: dict) -> dict:
    """child -> 其所有祖先类型集合（含自身）。"""
    closure = {}
    for child in types:
        chain, cur, guard = set(), child, 0
        while cur and guard < 64:
            chain.add(cur)
            cur = types.get(cur)
            guard += 1
        chain.add('object')
        closure[child] = chain
    return closure


def objects_of_type(all_objects, type_name, closure) -> list:
    """挑出可代入 type_name 参数位的对象。无类型或 object 则全部可用。"""
    if type_name is None or type_name == 'object':
        return [name for name, _ in all_objects]
    out = []
    for name, obj_type in all_objects:
        if obj_type is None:
            out.append(name)                      # 未标类型视作 object
        elif obj_type == type_name:
            out.append(name)
        elif type_name in closure.get(obj_type, {obj_type}):
            out.append(name)                      # 子类型可代入父类型位
    return out


# ------------------------------------------------------------------ negation

def find_negated_predicates(dom, prob) -> set:
    """找出真正需要补谓词编码的谓词。

    只有出现在**前提**或**目标**负位置的谓词才需要 not-P：
    因为那里要做"某文字为假"的判定，而状态是正文字集合，无法直接判定。
    仅出现在效果负位置（即 del_set）的谓词不需要编码——删除操作本身
    在 (state|add)-del 里已经正确表达。

    特别地，条件效果 (when C E) 的 C 在编译后会成为动作前提，
    且 else 分支需要表达 ¬C，因此 C 中出现的谓词**一律**需要编码。
    """
    negated = set()
    for schema in dom.actions:
        for atom, neg in collect_atoms(schema.precondition):
            if neg:
                negated.add(atom.predicate)
        # 条件效果的条件会被编译进 pre，且 else 分支要表达其否定
        _, conds = split_effect(schema.effect)
        for ce in conds:
            for atom, _neg in ce.cond:
                negated.add(atom.predicate)
    if prob.goal is not None:
        for atom, neg in collect_atoms(prob.goal):
            if neg:
                negated.add(atom.predicate)
    return negated


def _neg_atom(atom: Atom) -> Atom:
    return Atom(NEG_PREFIX + atom.predicate, atom.args)


# ------------------------------------------------------------------ grounding

class GroundingResult:
    def __init__(self, goal, start, actions, stats):
        self.goal = goal
        self.start = start
        self.actions = actions
        self.stats = stats

    def as_tuple(self):
        """与 examples.py 一致的 (goal, start, actions)。"""
        return self.goal, self.start, self.actions


def find_static_predicates(dom) -> set:
    """找出静态谓词：从不出现在任何动作效果中的谓词。

    这类谓词的真值在整个规划过程中恒定，完全由 :init 决定，
    因此可以在 grounding 期直接求值，用来剪掉大量永不可执行的动作实例。
    典型用途是把"类型系统表达不了的约束"编码成静态谓词，例如
    (connected ?from ?to)、(can-grasp ?robot ?item)。

    注意必须扫描条件效果内部——(when C (p ...)) 里的 p 也是被修改的，不算静态。
    """
    mutable = set()
    for schema in dom.actions:
        uncond, conds = split_effect(schema.effect)
        for atom, _neg in uncond:
            mutable.add(atom.predicate)
        for ce in conds:
            for atom, _neg in ce.eff:
                mutable.add(atom.predicate)
    return set(dom.predicates) - mutable


def ground(dom, prob, encode_negation=True, max_actions=200000,
           prune_static=True) -> GroundingResult:
    """把 (domain, problem) 实例化为 BT Expansion 输入。

    encode_negation=True 时自动做补谓词编码，处理负前提与负目标。
    prune_static=True 时用静态谓词在 grounding 期剪掉不可执行的动作实例。
    """
    closure = build_type_closure(dom.types)
    all_objects = list(prob.objects) + list(dom.constants)

    # 需要补谓词编码的谓词集合
    neg_preds = find_negated_predicates(dom, prob) if encode_negation else set()

    # 静态谓词：真值恒定，可在 grounding 期直接判定
    static_preds = find_static_predicates(dom) if prune_static else set()

    # ---- 初始状态
    start = set()
    init_positive = set()
    for formula in prob.init:
        for atom, neg in collect_atoms(formula):
            if not neg:
                init_positive.add(atom)
                start.add(atom.to_literal())

    # 闭世界：对需要编码的谓词，所有未在 :init 中出现的实例，其 not-P 为真
    # 注意必须尊重谓词的类型签名，否则会生成大量类型非法的无意义文字
    if neg_preds:
        for pred in sorted(neg_preds):
            arity = dom.predicates.get(pred)
            if arity is None:
                continue
            sig = dom.predicate_params.get(pred)
            if sig:
                domains = [objects_of_type(all_objects, t, closure)
                           for _, t in sig]
            else:
                obj_names = [name for name, _ in all_objects]
                domains = [obj_names] * arity
            if arity == 0:
                atom = Atom(pred, ())
                if atom not in init_positive:
                    start.add(_neg_atom(atom).to_literal())
                continue
            if any(len(d) == 0 for d in domains):
                continue
            for combo in itertools.product(*domains):
                atom = Atom(pred, combo)
                if atom not in init_positive:
                    start.add(_neg_atom(atom).to_literal())

    # ---- 目标
    goal = set()
    if prob.goal is not None:
        for atom, neg in collect_atoms(prob.goal):
            goal.add(_neg_atom(atom).to_literal() if neg else atom.to_literal())

    # ---- 动作 grounding
    actions = []
    skipped_static_eq = 0
    skipped_static_pred = 0
    cond_effect_schemas = 0
    cond_effect_instances = 0
    for schema in dom.actions:
        param_names = [p for p, _ in schema.parameters]
        param_types = [t for _, t in schema.parameters]
        domains = [objects_of_type(all_objects, t, closure) for t in param_types]

        if any(len(d) == 0 for d in domains):
            continue                               # 某参数位无可用对象

        # 条件效果编译：一个 schema 可能产出多个互斥分支
        cases = compile_conditional_effects(schema.effect)
        if len(cases) > 1 or split_effect(schema.effect)[1]:
            cond_effect_schemas += 1

        for combo in itertools.product(*domains) if param_names else [()]:
            binding = dict(zip(param_names, combo))

            base_pre_atoms = [(a.substitute(binding), neg)
                              for a, neg in collect_atoms(schema.precondition)]

            for case in cases:
                pre_atoms = list(base_pre_atoms)
                pre_atoms.extend((a.substitute(binding), neg)
                                 for a, neg in case.extra_pre)
                eff_atoms = [(a.substitute(binding), neg)
                             for a, neg in case.effect_literals]

                # :equality —— (= ?x ?y) 在 grounding 期静态求值
                # 静态谓词 —— 真值恒定，同样在 grounding 期求值
                skip = False
                filtered_pre = []
                for atom, neg in pre_atoms:
                    if atom.predicate == '=':
                        if len(atom.args) == 2:
                            equal = atom.args[0] == atom.args[1]
                            if equal == neg:       # 与前提要求矛盾
                                skip = True
                                skipped_static_eq += 1
                                break
                        continue                   # 静态满足，不进入 pre

                    if atom.predicate in static_preds:
                        # 静态谓词真值完全由 :init 决定（闭世界：未列出即为假）
                        holds = atom in init_positive
                        if holds == neg:           # 该实例永远不可执行
                            skip = True
                            skipped_static_pred += 1
                            break
                        continue                   # 恒真，无需作为运行时守卫

                    filtered_pre.append((atom, neg))
                if skip:
                    continue

                # 前提自相矛盾（同一原子既要求真又要求假）的实例直接丢弃
                seen = {}
                contradictory = False
                for atom, neg in filtered_pre:
                    if atom in seen and seen[atom] != neg:
                        contradictory = True
                        break
                    seen[atom] = neg
                if contradictory:
                    continue

                pre = set()
                for atom, neg in filtered_pre:
                    pre.add(_neg_atom(atom).to_literal() if neg
                            else atom.to_literal())

                add, dele = set(), set()
                for atom, neg in eff_atoms:
                    if neg:
                        dele.add(atom.to_literal())
                        if atom.predicate in neg_preds:
                            add.add(_neg_atom(atom).to_literal())
                    else:
                        add.add(atom.to_literal())
                        if atom.predicate in neg_preds:
                            dele.add(_neg_atom(atom).to_literal())

                inst_name = schema.name
                if combo:
                    inst_name = f"{schema.name}({','.join(combo)})"
                inst_name += case.suffix
                if case.suffix:
                    cond_effect_instances += 1

                actions.append(Action(name=inst_name, pre=pre, add=add,
                                      del_set=dele, cost=schema.cost))

                if len(actions) > max_actions:
                    raise MemoryError(
                        f'grounding 产生超过 {max_actions} 条动作，疑似组合爆炸。'
                        f'请减少对象数或加强 :types 约束')

    stats = {
        'objects': len(all_objects),
        'schemas': len(dom.actions),
        'grounded_actions': len(actions),
        'negated_predicates': sorted(neg_preds),
        'start_literals': len(start),
        'goal_literals': len(goal),
        'skipped_by_equality': skipped_static_eq,
        'static_predicates': sorted(static_preds),
        'skipped_by_static_pred': skipped_static_pred,
        'schemas_with_cond_effects': cond_effect_schemas,
        'cond_effect_instances': cond_effect_instances,
    }
    return GroundingResult(goal, start, actions, stats)

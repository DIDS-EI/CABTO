"""条件效果 (when C E) 编译器：把带分支的 add/del 编译成多个互斥的普通动作。

背景
----
BT Expansion 的 Action 只有扁平的 (pre, add, del_set)：

    new_state = (state | add) - del_set

add 是无条件并入的固定集合，没有"在 C 成立时 add E"的表达能力。
这就是 PDDL 的 :conditional-effects 无法直接表达的原因。

编译原理
--------
对动作 a，设其效果为「无条件部分 U」+「条件部分 (when C1 E1), (when C2 E2), ...」。
执行时每个 Ci 要么成立要么不成立，共 2^n 种组合。为每种组合生成一个动作实例：

    a_case_k:  pre    = pre(a) ∪ {选中分支的 Ci} ∪ {未选中分支的 ¬Ci}
               effect = U ∪ {选中分支的 Ei}

这些实例的前提两两互斥、且穷尽所有情况，因此与原动作语义等价。
BT Expansion 在扩展时会自动挑选能达成目标的那个分支，
并在必要时先把状态引导到该分支的前提上（实测会主动规避有害分支）。

关键难点：¬C 是析取
-------------------
C 通常是合取 (and c1 c2)，其否定 ¬C = (or ¬c1 ¬c2) 是析取，
而 pre 必须是合取文字集合。处理方式：对 C 中每个文字各生成一个 else 实例，
即 ¬C 的每个析取子句对应一个动作。这会让实例数进一步增多，
但保持了语义完整性（穷尽性由"每个子句一个实例"共同保证）。

代价
----
实例数随条件效果个数指数增长（2^n），再乘以 ¬C 的析取展开。
compile_conditional_effects 提供 max_cases 上限，超限时抛错而非静默截断——
静默截断会破坏穷尽性，从而破坏 Completeness。
"""

from __future__ import annotations

import itertools

from .parser import Atom, Formula


class CondEffect:
    """一条条件效果 (when C E)。cond / eff 均为 (atom, negated) 列表。"""

    __slots__ = ('cond', 'eff')

    def __init__(self, cond, eff):
        self.cond = cond
        self.eff = eff

    def __repr__(self):
        def fmt(items):
            return [('not ' if n else '') + a.to_literal() for a, n in items]
        return f'when({fmt(self.cond)} -> {fmt(self.eff)})'


def _flat_literals(formula: Formula, negated=False) -> list:
    """收集 and/not/atom 结构下的 (atom, negated)，遇到 when 则跳过。"""
    out = []
    if formula is None:
        return out
    if formula.op == 'atom':
        out.append((formula.atom, negated))
    elif formula.op == 'and':
        for c in formula.children:
            out.extend(_flat_literals(c, negated))
    elif formula.op == 'not':
        for c in formula.children:
            out.extend(_flat_literals(c, not negated))
    return out


def split_effect(formula: Formula):
    """把效果公式拆成 (无条件文字列表, [CondEffect, ...])。

    支持嵌套在 and 下的多个 when；when 内部再嵌 when 不支持（PDDL 也不允许）。
    """
    uncond, conds = [], []

    def walk(node, negated=False):
        if node is None:
            return
        if node.op == 'when':
            if len(node.children) < 2:
                return
            cond = _flat_literals(node.children[0])
            eff = []
            for child in node.children[1:]:
                eff.extend(_flat_literals(child))
            conds.append(CondEffect(cond, eff))
        elif node.op == 'and':
            for c in node.children:
                walk(c, negated)
        elif node.op == 'not':
            for c in node.children:
                walk(c, not negated)
        elif node.op == 'atom':
            uncond.append((node.atom, negated))

    walk(formula)
    return uncond, conds


def has_conditional_effect(formula: Formula) -> bool:
    _, conds = split_effect(formula)
    return bool(conds)


class CompiledCase:
    """编译产出的一个分支实例。"""

    __slots__ = ('suffix', 'extra_pre', 'effect_literals', 'taken')

    def __init__(self, suffix, extra_pre, effect_literals, taken):
        self.suffix = suffix                 # 动作名后缀，便于追溯
        self.extra_pre = extra_pre           # 需追加到 pre 的 (atom, negated)
        self.effect_literals = effect_literals   # 该分支的完整效果 (atom, negated)
        self.taken = taken                   # 哪些条件分支被选中（供诊断）


def compile_conditional_effects(effect: Formula, max_cases=64) -> list:
    """把带条件效果的动作效果编译成多个互斥分支。

    返回 [CompiledCase, ...]。若无条件效果，返回单个 case（原效果不变）。
    """
    uncond, conds = split_effect(effect)

    if not conds:
        return [CompiledCase('', [], list(uncond), ())]

    cases = []
    # 每个条件效果有两种走向：True（条件成立，效果生效）/ False（不成立）
    # False 分支需要展开 ¬C 的每个析取子句
    branch_options = []
    for idx, ce in enumerate(conds):
        options = [('T', idx, None)]
        # ¬C 的析取展开：C 中每个文字取反各成一个子句
        for lit_i in range(len(ce.cond)):
            options.append(('F', idx, lit_i))
        if not ce.cond:
            # 空条件等价于无条件效果，只保留 T 分支
            options = [('T', idx, None)]
        branch_options.append(options)

    for combo in itertools.product(*branch_options):
        extra_pre, eff_lits, taken, tags = [], list(uncond), [], []
        for (mode, idx, lit_i) in combo:
            ce = conds[idx]
            if mode == 'T':
                extra_pre.extend(ce.cond)            # 要求 C 成立
                eff_lits.extend(ce.eff)              # 效果生效
                taken.append(idx)
                tags.append('+' + _cond_tag(ce))
            else:
                atom, neg = ce.cond[lit_i]
                extra_pre.append((atom, not neg))    # 要求该文字取反
                tags.append('-' + atom.predicate)
        suffix = '[' + ','.join(tags) + ']' if tags else ''
        cases.append(CompiledCase(suffix, extra_pre, eff_lits, tuple(taken)))
        if len(cases) > max_cases:
            raise MemoryError(
                f'条件效果编译产生超过 {max_cases} 个分支实例。'
                f'该动作含 {len(conds)} 条 (when ...)，分支数随其指数增长。'
                f'请减少条件效果数量，或提高 max_cases（注意 grounding 规模）')

    return _drop_contradictory(cases)


def _cond_tag(ce: CondEffect) -> str:
    """给 then 分支起一个可读标签，用条件谓词名拼接。"""
    return '&'.join(
        ('!' if neg else '') + atom.predicate for atom, neg in ce.cond) or 'true'


def _drop_contradictory(cases: list) -> list:
    """剔除前提自相矛盾的分支（同一原子被同时要求为真和为假）。"""
    kept = []
    for case in cases:
        seen = {}
        bad = False
        for atom, neg in case.extra_pre:
            if atom in seen and seen[atom] != neg:
                bad = True
                break
            seen[atom] = neg
        if not bad:
            kept.append(case)
    return kept

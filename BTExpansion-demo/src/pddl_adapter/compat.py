"""PDDL → BT Expansion 兼容性判定。

这个模块是对"什么样的 PDDL 能拿到 Sound/Complete 的响应式策略树"这一问题的可执行回答。

判定结论分三档：
  SUPPORTED    直接可转，保持论文的 Sound + Complete 保证
  REWRITABLE   需等价改写后可转（改写后仍 Sound/Complete，但状态空间被放大）
  UNSUPPORTED  超出 BT Expansion 的命题 STRIPS 模型，无法保证 Sound/Complete

判定依据来自算法内核的实证分析（src/bt_expansion/algorithm.py）：
  1. 状态是"正文字字符串集合"，条件判定是 c <= state（集合包含），
     是闭世界 + 无负文字的一等表示     -> 负前提/负目标需改写
  2. 动作是 (pre, add, del_set) 三元组，效果无条件
                                        -> 条件效果(when)无法表达
  3. 回归公式 c_attr = (pre | c) - add  -> 要求 pre ∩ add = ∅，否则丢解
  4. 动作筛选要求 (c - del_set) == c    -> 目标文字不可被该动作删除（算法自身处理）
  5. tick 基于集合包含，确定性单一后继 -> 概率/非确定效果无法表达
  6. 无数值/时间维度                    -> fluents / durative-action 无法表达
"""

from __future__ import annotations

from .parser import Atom, Formula
from .cond_effects import split_effect

SUPPORTED = 'SUPPORTED'
REWRITABLE = 'REWRITABLE'
UNSUPPORTED = 'UNSUPPORTED'

_RANK = {SUPPORTED: 0, REWRITABLE: 1, UNSUPPORTED: 2}


class Finding:
    """一条诊断。level 为三档之一，where 指出位置，hint 给出可操作建议。"""

    __slots__ = ('level', 'code', 'where', 'message', 'hint')

    def __init__(self, level, code, where, message, hint=''):
        self.level = level
        self.code = code
        self.where = where
        self.message = message
        self.hint = hint

    def __repr__(self):
        return f'[{self.level}] {self.where}: {self.message}'


# ------------------------------------------------------------------ helpers

def _walk(formula: Formula):
    """深度遍历公式树。"""
    if formula is None:
        return
    yield formula
    for child in formula.children:
        yield from _walk(child)


def collect_atoms(formula: Formula, negated=False) -> list:
    """收集 (atom, is_negated) 列表。只处理 and/not/atom，其他结构由调用方先行判定。"""
    out = []
    if formula is None:
        return out
    if formula.op == 'atom':
        out.append((formula.atom, negated))
    elif formula.op == 'and':
        for c in formula.children:
            out.extend(collect_atoms(c, negated))
    elif formula.op == 'not':
        for c in formula.children:
            out.extend(collect_atoms(c, not negated))
    return out


# ------------------------------------------------------------------ checks

# PDDL requirement 标签的支持程度
_REQ_TABLE = {
    ':strips': SUPPORTED,
    ':typing': SUPPORTED,
    ':equality': REWRITABLE,
    ':negative-preconditions': REWRITABLE,
    ':disjunctive-preconditions': UNSUPPORTED,
    ':existential-preconditions': UNSUPPORTED,
    ':universal-preconditions': UNSUPPORTED,
    ':quantified-preconditions': UNSUPPORTED,
    ':conditional-effects': REWRITABLE,
    ':adl': UNSUPPORTED,
    ':fluents': UNSUPPORTED,
    ':numeric-fluents': UNSUPPORTED,
    ':object-fluents': UNSUPPORTED,
    ':durative-actions': UNSUPPORTED,
    ':duration-inequalities': UNSUPPORTED,
    ':continuous-effects': UNSUPPORTED,
    ':derived-predicates': UNSUPPORTED,
    ':timed-initial-literals': UNSUPPORTED,
    ':preferences': UNSUPPORTED,
    ':constraints': UNSUPPORTED,
    ':action-costs': SUPPORTED,
}

_REQ_REASON = {
    ':negative-preconditions': ('状态是正文字集合，无负文字一等表示',
                                '为每个出现在负位置的谓词 P 引入显式补谓词 not-P，'
                                '并在所有修改 P 的动作里同步维护 not-P'),
    ':equality': ('= 不是状态里的可变命题',
                  'grounding 阶段静态求值：参数相等/不等直接决定该实例是否保留'),
    ':disjunctive-preconditions': ('前提必须是合取的文字集合（一个 set）',
                                   '把析取前提拆成多个动作实例，每个分支一个动作'),
    ':conditional-effects': ('动作效果是固定的 add/del 集合，不能依赖执行时状态',
                             '适配器自动编译：对每条 (when C E) 生成 then/else 互斥'
                             '动作分支（then 的 pre 加 C 且效果含 E，else 的 pre 加 ¬C）。'
                             '语义等价，但实例数随条件效果个数指数增长'),
    ':fluents': ('状态无数值维度，集合包含无法表达数值比较',
                 '无法改写；需改用数值规划器，或把数值离散化成命题'),
    ':numeric-fluents': ('同上，状态无数值维度', '离散化或改用数值规划器'),
    ':durative-actions': ('模型无时间维度，tick 是离散瞬时的',
                          '无法改写；需时序规划器'),
    ':derived-predicates': ('无公理/推导规则求闭包机制',
                            '若推导层次有限，可手工展开进基础谓词'),
    ':universal-preconditions': ('前提无全称量词展开机制',
                                 '对象集有限且已知时可在 grounding 期展开成合取'),
    ':existential-preconditions': ('前提无存在量词，且展开后是析取',
                                   '展开成析取后再按析取前提拆分动作'),
}


def _check_requirements(dom, prob, findings):
    # :negative-preconditions 只在"前提或目标里真的用了负文字"时才需要改写。
    # 仅在效果里用 (not ...) 是标准的 del_set，不需要任何改写。
    actually_needs_neg_encoding = False
    for schema in dom.actions:
        if any(neg for _, neg in collect_atoms(schema.precondition)):
            actually_needs_neg_encoding = True
            break
    if prob is not None and prob.goal is not None:
        if any(neg for _, neg in collect_atoms(prob.goal)):
            actually_needs_neg_encoding = True

    for req in dom.requirements:
        if req == ':negative-preconditions' and not actually_needs_neg_encoding:
            continue                               # 声明了但没实际用在前提/目标，无需改写
        level = _REQ_TABLE.get(req)
        if level is None:
            findings.append(Finding(
                UNSUPPORTED, 'REQ_UNKNOWN', f'domain :requirements {req}',
                f'未知的 requirement {req}，无法判定其语义',
                '请确认是否为 PDDL 标准标签；非标准扩展一律视为不支持'))
        elif level != SUPPORTED:
            reason, hint = _REQ_REASON.get(req, ('超出命题 STRIPS 模型', ''))
            findings.append(Finding(
                level, 'REQ_' + req.strip(':').upper().replace('-', '_'),
                f'domain :requirements {req}',
                f'{req} —— {reason}', hint))


def _check_formula_shape(formula, where, findings, is_effect):
    """检查公式是否落在"合取的（可带 not 的）文字"这一形状内。"""
    for node in _walk(formula):
        op = node.op
        if op.startswith('unsupported:'):
            findings.append(Finding(
                UNSUPPORTED, 'CONSTRUCT', where,
                f'使用了不支持的 PDDL 构造 {op.split(":", 1)[1]}',
                '该构造超出命题 STRIPS 模型'))
        elif op.startswith('numeric:'):
            findings.append(Finding(
                UNSUPPORTED, 'NUMERIC', where,
                f'出现数值/比较算子 {op.split(":", 1)[1]}',
                '状态是命题集合，无数值维度；需离散化或改用数值规划器'))
        elif op == 'or':
            findings.append(Finding(
                UNSUPPORTED, 'DISJUNCTION', where,
                '出现析取 (or ...)',
                '前提需为合取文字集合；可把每个析取分支拆成独立动作实例。'
                '若出现在目标中，可对每个分支各生成一棵树后用顶层 Selector 合并'))
        elif op == 'imply':
            findings.append(Finding(
                UNSUPPORTED, 'IMPLY', where,
                '出现蕴含 (imply ...)',
                '先化为 (or (not a) b)，再按析取处理'))
        elif op == 'when':
            findings.append(Finding(
                REWRITABLE, 'CONDITIONAL_EFFECT', where,
                '出现条件效果 (when ...)',
                '适配器自动编译为 then/else 互斥动作分支：then 的前提追加 C 且效果含 E，'
                'else 的前提追加 ¬C 的一个析取子句。语义等价，Sound/Complete 保持，'
                '代价是动作实例数增长（2^n 再乘 ¬C 的析取展开）'))
        elif op in ('forall', 'exists'):
            findings.append(Finding(
                UNSUPPORTED, 'QUANTIFIER', where,
                f'出现量词 ({op} ...)',
                'forall 在有限对象集上可 grounding 期展开为合取；'
                'exists 展开为析取后需再拆动作'))


def _check_action(schema, index, findings):
    where = f'action "{schema.name}"'

    _check_formula_shape(schema.precondition, where + ' :precondition', findings, False)
    _check_formula_shape(schema.effect, where + ' :effect', findings, True)

    pre_atoms = collect_atoms(schema.precondition)
    eff_atoms = collect_atoms(schema.effect)

    # 条件效果的分支数预估（collect_atoms 不进入 when，故需单独处理）
    _, cond_effects = split_effect(schema.effect)
    if cond_effects:
        branches = 1
        for ce in cond_effects:
            branches *= (1 + max(len(ce.cond), 1))   # T 分支 + ¬C 的析取子句数
        findings.append(Finding(
            REWRITABLE, 'COND_EFFECT_BLOWUP', where + ' :effect',
            f'含 {len(cond_effects)} 条条件效果，编译后约 {branches} 个动作分支',
            '分支数 = ∏(1 + |Ci|)，随条件效果个数指数增长。'
            '再乘以参数 grounding 的组合数即为最终动作数，注意规模'
            if branches > 8 else
            '规模可控，适配器会自动编译为互斥分支'))

        # 条件效果内部也要检查 pre∩add 冲突（分支合并后才能看出）
        uncond, _ = split_effect(schema.effect)
        uncond_add = {a for a, neg in uncond if not neg}
        pre_positive = {a for a, neg in collect_atoms(schema.precondition)
                        if not neg}
        for ce in cond_effects:
            ce_add = {a for a, neg in ce.eff if not neg}
            ce_cond_pos = {a for a, neg in ce.cond if not neg}
            # then 分支的 pre 是 pre ∪ C，其 add 含 E
            overlap = (pre_positive | ce_cond_pos) & (ce_add | uncond_add)
            if overlap:
                findings.append(Finding(
                    UNSUPPORTED, 'PRE_ADD_OVERLAP', where + ' :effect (when)',
                    f'条件效果的 then 分支出现 pre∩add 非空: '
                    f'{sorted(a.to_literal() for a in overlap)}',
                    '编译后该分支的前提含条件 C，若 C 中原子同时出现在效果的 add 中，'
                    '回归公式 (pre|c)-add 会把它减掉而丢解。'
                    '请把该原子从 add 中移除（它已由前提保证成立）'))

    # 负前提 -> 需要补谓词编码
    for atom, neg in pre_atoms:
        if neg:
            findings.append(Finding(
                REWRITABLE, 'NEG_PRECOND', where + ' :precondition',
                f'负前提 (not ({atom.predicate} ...))',
                f'引入补谓词 not-{atom.predicate}，并在所有增删 {atom.predicate} '
                f'的动作中同步维护它（本适配器可自动完成）'))

    # 条件效果的条件 C 会被编译进前提，其中的谓词一律需要补谓词编码
    for ce in cond_effects:
        for atom, _neg in ce.cond:
            findings.append(Finding(
                REWRITABLE, 'COND_NEEDS_NEG_ENCODING', where + ' :effect (when)',
                f'条件效果的条件含谓词 {atom.predicate}，else 分支需表达其否定',
                f'适配器自动引入补谓词 not-{atom.predicate} 并同步维护'))

    add_set = {a for a, neg in eff_atoms if not neg}
    del_set = {a for a, neg in eff_atoms if neg}
    pre_set = {a for a, neg in pre_atoms if not neg}

    # add ∩ del ≠ ∅：同一动作同时增删同一原子，语义歧义
    both = add_set & del_set
    if both:
        findings.append(Finding(
            UNSUPPORTED, 'ADD_DEL_CONFLICT', where + ' :effect',
            f'同一原子同时出现在正/负效果中: '
            f'{sorted(a.to_literal() for a in both)}',
            'state_transition 为 (state|add)-del，del 会覆盖 add，'
            '语义取决于实现细节，应在建模层消除歧义'))

    # 关键：pre ∩ add ≠ ∅ 会让回归公式 (pre|c)-add 丢掉真实前提 -> 丢解
    overlap = pre_set & add_set
    if overlap:
        findings.append(Finding(
            UNSUPPORTED, 'PRE_ADD_OVERLAP', where,
            f'前提与正效果交集非空: '
            f'{sorted(a.to_literal() for a in overlap)}',
            '回归公式 c_attr=(pre|c)-add 会把这些既是前提又是效果的原子减掉，'
            '导致生成的树缺少必要守卫而丢解（完备性被破坏）。'
            '建议改写：把"保持不变的前提"从 add 中移除（它本来就成立，无需再 add）'))

    # 空效果动作：不改变状态，扩展时无贡献
    if not add_set and not del_set:
        findings.append(Finding(
            REWRITABLE, 'EMPTY_EFFECT', where + ' :effect',
            '动作没有任何效果',
            '该动作永远不会被算法选中（不贡献任何目标文字），可安全删除'))


def _check_goal(goal, findings):
    if goal is None:
        findings.append(Finding(
            UNSUPPORTED, 'NO_GOAL', 'problem :goal',
            '缺少目标', '必须提供 :goal'))
        return
    _check_formula_shape(goal, 'problem :goal', findings, False)
    for atom, neg in collect_atoms(goal):
        if neg:
            findings.append(Finding(
                REWRITABLE, 'NEG_GOAL', 'problem :goal',
                f'负目标 (not ({atom.predicate} ...))',
                f'引入补谓词 not-{atom.predicate} 后目标变为正文字'))


def _check_init(prob, findings):
    for formula in prob.init:
        for atom, neg in collect_atoms(formula):
            if neg:
                findings.append(Finding(
                    REWRITABLE, 'NEG_INIT', 'problem :init',
                    f'初始状态中出现负文字 {atom.predicate}',
                    'PDDL 是闭世界，:init 中未列出即为假，负文字通常可直接删除'))
        if formula.op.startswith('numeric:'):
            findings.append(Finding(
                UNSUPPORTED, 'NUMERIC_INIT', 'problem :init',
                '初始状态含数值赋值', '状态无数值维度'))


# ------------------------------------------------------------------ report

class CompatReport:
    def __init__(self, findings):
        self.findings = findings

    @property
    def verdict(self) -> str:
        if not self.findings:
            return SUPPORTED
        return max((f.level for f in self.findings), key=lambda lv: _RANK[lv])

    @property
    def can_convert(self) -> bool:
        """是否能得到带 Sound/Complete 保证的策略树（可能需先自动改写）。"""
        return self.verdict != UNSUPPORTED

    def by_level(self, level) -> list:
        return [f for f in self.findings if f.level == level]

    def format(self) -> str:
        lines = []
        verdict = self.verdict
        head = {
            SUPPORTED: '可直接转换，保持 Sound + Complete 保证',
            REWRITABLE: '需等价改写后可转换，改写后仍保持 Sound + Complete',
            UNSUPPORTED: '无法转换：超出 BT Expansion 的命题 STRIPS 模型',
        }[verdict]
        lines.append(f'判定: {verdict} —— {head}')
        for level in (UNSUPPORTED, REWRITABLE):
            items = self.by_level(level)
            if not items:
                continue
            label = '阻断项' if level == UNSUPPORTED else '需改写项'
            lines.append('')
            lines.append(f'{label} ({len(items)}):')
            for f in items:
                lines.append(f'  - {f.where}')
                lines.append(f'    问题: {f.message}')
                if f.hint:
                    lines.append(f'    建议: {f.hint}')
        return '\n'.join(lines)


def check(dom, prob) -> CompatReport:
    """对 (domain, problem) 做完整兼容性判定。"""
    findings = []
    _check_requirements(dom, prob, findings)
    for i, schema in enumerate(dom.actions):
        _check_action(schema, i, findings)
    _check_goal(prob.goal, findings)
    _check_init(prob, findings)
    return CompatReport(findings)

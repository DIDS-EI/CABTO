"""零依赖的 PDDL S-表达式解析器。

只解析结构，不做语义判定；能力判定交给 compat.py，grounding 交给 grounding.py。

解析范围：
  domain  : :requirements :types :constants :predicates :action(:parameters/:precondition/:effect)
  problem : :objects :init :goal :metric

设计原则：宽进严出——解析阶段尽量把 PDDL 原样读进来（包括不支持的构造），
把"能不能转成 BT Expansion 的输入"这一判断完全留给兼容性检查器，
这样才能给出精确的诊断而不是笼统的解析失败。
"""

from __future__ import annotations


# ---------------------------------------------------------------- tokenizer

def tokenize(text: str) -> list:
    """把 PDDL 文本切成 token 列表，去掉 ; 注释。"""
    out = []
    for raw_line in text.splitlines():
        line = raw_line.split(';', 1)[0]          # PDDL 用 ; 作行注释
        line = line.replace('(', ' ( ').replace(')', ' ) ')
        out.extend(tok for tok in line.split() if tok)
    return out


def parse_sexpr(tokens: list) -> list:
    """token 列表 -> 嵌套 list（S-表达式）。原子统一转小写（PDDL 大小写不敏感）。"""
    pos = 0

    def walk():
        nonlocal pos
        if pos >= len(tokens):
            raise SyntaxError('PDDL 意外结束：括号不匹配')
        tok = tokens[pos]
        pos += 1
        if tok == '(':
            node = []
            while pos < len(tokens) and tokens[pos] != ')':
                node.append(walk())
            if pos >= len(tokens):
                raise SyntaxError('PDDL 缺少右括号 )')
            pos += 1                               # 吃掉 ')'
            return node
        if tok == ')':
            raise SyntaxError('PDDL 出现多余的右括号 )')
        return tok.lower()

    root = walk()
    if pos != len(tokens):
        raise SyntaxError('PDDL 顶层存在多余内容（可能是括号不匹配）')
    return root


def parse_file(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as fh:
        return parse_sexpr(tokenize(fh.read()))


# ---------------------------------------------------------------- typed lists

def parse_typed_list(items: list) -> list:
    """解析 PDDL 的 typed list: (?a ?b - block ?c - table) 或无类型 (?a ?b)。

    返回 [(name, type_or_None), ...]，保持声明顺序（参数顺序对 grounding 很关键）。
    """
    result, pending = [], []
    i = 0
    while i < len(items):
        tok = items[i]
        if tok == '-':
            if i + 1 >= len(items):
                raise SyntaxError('typed list 中 - 之后缺少类型名')
            type_name = items[i + 1]
            if isinstance(type_name, list):
                # (either a b) 这类联合类型，原样保留供 compat 判定
                type_name = ['either-unsupported'] + type_name
            result.extend((nm, type_name) for nm in pending)
            pending = []
            i += 2
            continue
        pending.append(tok)
        i += 1
    result.extend((nm, None) for nm in pending)    # 尾部未标类型的
    return result


# ---------------------------------------------------------------- structures

class Atom:
    """一阶原子： predicate(arg1, arg2, ...)，args 可含 ?var 变量。"""

    __slots__ = ('predicate', 'args')

    def __init__(self, predicate: str, args: tuple):
        self.predicate = predicate
        self.args = tuple(args)

    @property
    def is_ground(self) -> bool:
        return not any(a.startswith('?') for a in self.args)

    def variables(self) -> set:
        return {a for a in self.args if a.startswith('?')}

    def substitute(self, binding: dict) -> 'Atom':
        return Atom(self.predicate, tuple(binding.get(a, a) for a in self.args))

    def to_literal(self) -> str:
        """转成 BT Expansion 用的字符串命题，与 examples.py 的风格一致。"""
        if not self.args:
            return self.predicate
        return f"{self.predicate}({','.join(self.args)})"

    def __eq__(self, other):
        return (isinstance(other, Atom)
                and self.predicate == other.predicate
                and self.args == other.args)

    def __hash__(self):
        return hash((self.predicate, self.args))

    def __repr__(self):
        return f'Atom({self.to_literal()})'


def parse_atom(node) -> Atom:
    if isinstance(node, str):
        return Atom(node, ())
    if not node:
        raise SyntaxError('遇到空原子 ()')
    return Atom(node[0], tuple(node[1:]))


class Formula:
    """公式树。op ∈ {atom, and, or, not, imply, forall, exists, when, 以及未知算子}。

    刻意保留 or / forall / when 等不支持的结构，让 compat.py 能精确报告
    "第几个动作的效果里有条件效果"，而不是解析时就崩掉。
    """

    __slots__ = ('op', 'children', 'atom', 'params')

    def __init__(self, op, children=None, atom=None, params=None):
        self.op = op
        self.children = children or []
        self.atom = atom
        self.params = params or []

    def __repr__(self):
        if self.op == 'atom':
            return self.atom.to_literal()
        if self.params:
            return f'({self.op} {self.params} {self.children})'
        return f'({self.op} {self.children})'


QUANTIFIERS = ('forall', 'exists')
CONNECTIVES = ('and', 'or', 'not', 'imply', 'when')


def parse_formula(node) -> Formula:
    if isinstance(node, str):
        return Formula('atom', atom=Atom(node, ()))
    if not node:
        return Formula('and')                      # () 视作空合取 = True
    head = node[0]
    if isinstance(head, str) and head in CONNECTIVES:
        return Formula(head, [parse_formula(c) for c in node[1:]])
    if isinstance(head, str) and head in QUANTIFIERS:
        params = parse_typed_list(node[1]) if len(node) > 1 and isinstance(node[1], list) else []
        body = [parse_formula(c) for c in node[2:]]
        return Formula(head, body, params=params)
    # 数值/比较算子（PDDL2.1 fluents）与其他未知算子，原样记录供 compat 报错
    if isinstance(head, str) and head in ('=', '<', '>', '<=', '>=',
                                          'increase', 'decrease', 'assign',
                                          'scale-up', 'scale-down'):
        return Formula('numeric:' + head, [])
    return Formula('atom', atom=parse_atom(node))


class ActionSchema:
    """PDDL 的 :action —— 即 lifted 一阶提升算子。"""

    def __init__(self, name, parameters, precondition, effect, cost=1):
        self.name = name
        self.parameters = parameters               # [(?x, type|None), ...]
        self.precondition = precondition           # Formula
        self.effect = effect                       # Formula
        self.cost = cost

    def __repr__(self):
        return f'ActionSchema({self.name}, params={self.parameters})'


class Domain:
    def __init__(self):
        self.name = ''
        self.requirements = []
        self.types = {}                            # child -> parent
        self.constants = []                        # [(name, type)]
        self.predicates = {}                       # name -> arity
        self.predicate_params = {}                 # name -> [(param, type|None)]
        self.actions = []                          # [ActionSchema]


class Problem:
    def __init__(self):
        self.name = ''
        self.domain_name = ''
        self.objects = []                          # [(name, type)]
        self.init = []                             # [Formula]
        self.goal = None                           # Formula
        self.metric = None


# ---------------------------------------------------------------- domain

def build_domain(ast: list) -> Domain:
    if not ast or ast[0] != 'define':
        raise SyntaxError('domain 文件必须以 (define ...) 开头')
    dom = Domain()
    for section in ast[1:]:
        if not isinstance(section, list) or not section:
            continue
        key = section[0]
        if key == 'domain':
            dom.name = section[1] if len(section) > 1 else ''
        elif key == ':requirements':
            dom.requirements = list(section[1:])
        elif key == ':types':
            for name, parent in parse_typed_list(section[1:]):
                dom.types[name] = parent or 'object'
        elif key == ':constants':
            dom.constants = parse_typed_list(section[1:])
        elif key == ':predicates':
            for pred in section[1:]:
                if isinstance(pred, list) and pred:
                    # 参数是 typed list：(item-at ?i - item ?l - location) 的
                    # arity 是 2，不能用 len(pred)-1（那会把 '-' 和类型名也算进去）
                    dom.predicates[pred[0]] = len(parse_typed_list(pred[1:]))
                    dom.predicate_params[pred[0]] = parse_typed_list(pred[1:])
        elif key == ':action':
            dom.actions.append(_build_action(section))
        elif key in (':durative-action', ':derived', ':functions'):
            # 保留标记，交给 compat 明确报不支持
            dom.actions.append(ActionSchema(
                name=f'{key}:{section[1] if len(section) > 1 else "?"}',
                parameters=[],
                precondition=Formula('unsupported:' + key),
                effect=Formula('unsupported:' + key),
            ))
    return dom


def _build_action(section: list) -> ActionSchema:
    name = section[1]
    params, pre, eff, cost = [], Formula('and'), Formula('and'), 1
    i = 2
    while i < len(section):
        key = section[i]
        val = section[i + 1] if i + 1 < len(section) else None
        if key == ':parameters':
            params = parse_typed_list(val) if isinstance(val, list) else []
        elif key in (':precondition', ':condition'):
            pre = parse_formula(val)
        elif key == ':effect':
            eff = parse_formula(val)
        elif key == ':cost':
            try:
                cost = float(val)
            except (TypeError, ValueError):
                cost = 1
        i += 2
    return ActionSchema(name, params, pre, eff, cost)


# ---------------------------------------------------------------- problem

def build_problem(ast: list) -> Problem:
    if not ast or ast[0] != 'define':
        raise SyntaxError('problem 文件必须以 (define ...) 开头')
    prob = Problem()
    for section in ast[1:]:
        if not isinstance(section, list) or not section:
            continue
        key = section[0]
        if key == 'problem':
            prob.name = section[1] if len(section) > 1 else ''
        elif key == ':domain':
            prob.domain_name = section[1] if len(section) > 1 else ''
        elif key == ':objects':
            prob.objects = parse_typed_list(section[1:])
        elif key == ':init':
            prob.init = [parse_formula(f) for f in section[1:]]
        elif key == ':goal':
            prob.goal = parse_formula(section[1]) if len(section) > 1 else None
        elif key == ':metric':
            prob.metric = list(section[1:])
    return prob


def load_domain(path: str) -> Domain:
    return build_domain(parse_file(path))


def load_problem(path: str) -> Problem:
    return build_problem(parse_file(path))

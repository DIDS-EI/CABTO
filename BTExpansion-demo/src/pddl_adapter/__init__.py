"""PDDL -> BT Expansion 适配器顶层入口。

用法:
    from pddl_adapter import solve_pddl
    res = solve_pddl('domain.pddl', 'problem.pddl')
    print(res.report.format())
    print(res.ptml)
"""

from .parser import (Atom, Formula, Domain, Problem, ActionSchema,
                     load_domain, load_problem, parse_sexpr, tokenize,
                     build_domain, build_problem)
from .compat import (check, CompatReport, Finding,
                     SUPPORTED, REWRITABLE, UNSUPPORTED, collect_atoms)
from .grounding import ground, GroundingResult
from .runner import solve_pddl, SolveResult

__all__ = [
    'Atom', 'Formula', 'Domain', 'Problem', 'ActionSchema',
    'load_domain', 'load_problem', 'parse_sexpr', 'tokenize',
    'build_domain', 'build_problem',
    'check', 'CompatReport', 'Finding', 'collect_atoms',
    'SUPPORTED', 'REWRITABLE', 'UNSUPPORTED',
    'ground', 'GroundingResult',
    'solve_pddl', 'SolveResult',
]

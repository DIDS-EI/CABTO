"""端到端驱动：PDDL 文件 -> 兼容性判定 -> grounding -> BT Expansion -> 策略树。

关于合取目标的重要说明
----------------------
bt_expansion.algorithm.BTExpAlgorithm.run_algorithm 在 len(goal)>1 时，
会对 goal 集合里的每个文字单独扩展一棵子树，再用顶层 Selector 合并。
这个分支有两个问题：

  1. 它把 `for g in goal` 得到的单个字符串直接传给 run_algorithm_selTree，
     而内部要做 `c <= start` 的集合运算，str 与 set 比较会抛 TypeError。
  2. 即使类型可用，"每个子目标各自一棵树 + Selector 合并"在语义上是析取而非合取，
     对合取目标是错的。

因此本适配器直接调用 run_algorithm_selTree(start, goal_set, actions)，
把整个目标集合作为一个合取条件交给算法——这与论文里"目标是一个条件集合 c"
的设定一致，实测可正确处理多文字目标。
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bt_expansion.algorithm import BTExpAlgorithm
from bt_expansion.planning import state_transition

from .parser import load_domain, load_problem
from .compat import check, UNSUPPORTED
from .grounding import ground


class SolveResult:
    def __init__(self):
        self.report = None          # CompatReport
        self.grounding = None       # GroundingResult
        self.algo = None            # BTExpAlgorithm
        self.bt = None              # 策略树
        self.ptml = ''
        self.plan = []              # 实际执行的动作名序列
        self.reached_goal = False
        self.tree_size = 0
        self.cost = 0
        self.ticks = 0
        self.planning_time = 0.0
        self.aborted_reason = ''

    @property
    def ok(self) -> bool:
        return self.bt is not None and self.reached_goal


def solve_pddl(domain_path, problem_path, verbose=False,
               execute=True, strict=True, ptml_name=None,
               output_dir=None, max_steps=500) -> SolveResult:
    """完整流程。strict=True 时遇到 UNSUPPORTED 直接停止并给出诊断。"""
    res = SolveResult()

    dom = load_domain(domain_path)
    prob = load_problem(problem_path)

    res.report = check(dom, prob)
    if strict and res.report.verdict == UNSUPPORTED:
        res.aborted_reason = (
            '存在阻断项，无法保证生成的策略树具有 Sound/Complete 性质；'
            '已停止转换（如需强行尝试请传 strict=False）')
        return res

    res.grounding = ground(dom, prob)
    goal, start, actions = res.grounding.as_tuple()

    if not goal:
        res.aborted_reason = '目标为空，无需规划'
        return res

    algo = BTExpAlgorithm(verbose=verbose)
    algo.clear()

    t0 = time.time()
    # 见模块 docstring：绕开 run_algorithm 的多目标分支，直接传整个目标集合
    bt = algo.run_algorithm_selTree(start, goal, actions)
    res.planning_time = time.time() - t0

    if bt is False or bt is None:
        res.aborted_reason = (
            '算法返回 Failure：在给定动作集下该目标不可达（完备性保证下即为真无解）')
        return res

    algo.bt = bt
    res.algo = algo
    res.bt = bt
    res.tree_size = bt.count_size() - 1

    # 注意：不要用 algo.get_ptml_many_act()——它内部的 dfs_ptml_many_act 会
    # 原地执行 child.content.name = re.sub(r'\d+', '', name)，把 Action 的名字
    # 改掉（r1 -> r、cup1 -> cup），污染后续执行输出。save_ptml_file 走的
    # dfs_ptml 没有这个副作用，故统一使用它。
    try:
        out_dir = output_dir
        name = ptml_name or os.path.splitext(os.path.basename(problem_path))[0]
        if out_dir is None:
            # 未指定输出目录时也只生成字符串，不落盘
            algo.ptml_string = "selector{\n"
            algo.dfs_ptml(algo.bt.children[0])
            algo.ptml_string += '}\n'
            res.ptml = algo.ptml_string
        else:
            res.ptml = algo.save_ptml_file(name, out_dir)
    except Exception as exc:                       # PTML 导出失败不影响主流程
        res.ptml = f'(PTML 导出失败: {exc})'

    if execute:
        state = set(start)
        steps = 0
        val, obj, cost, ticks = bt.cost_tick(state, 0, 0)
        total_cost = cost
        while val not in ('success', 'failure') and steps < max_steps:
            state = state_transition(state, obj)
            res.plan.append(obj.name)
            val, obj, cost, ticks = bt.cost_tick(state, 0, ticks)
            total_cost += cost
            steps += 1
        res.cost = total_cost
        res.ticks = ticks
        res.reached_goal = goal <= state
        if val == 'failure':
            res.aborted_reason = f'执行在第 {steps} 步返回 failure'

    return res

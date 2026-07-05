"""
实验1 (High-level model proposal) 离线重放复现
=============================================
对论文实验1 的 7 个任务集 x 10 次 try x 3 个难度 goal 做离线复现：

  对每个 (task, try):
    1) 从 a_exp1_llm_bt_results/<task>_try_<i>_<ts>/llm_output.txt 取出当时真实 LLM 输出
       (ReplayLLM，零 API 调用)
    2) 用与原始脚本完全相同的正则逻辑抽取 OGAction/OGCondition 代码块，写出 exec_lib
    3) 对 easy/medium/hard 三个 goal 调 obtea 规划 + execute_bt 做 sound&complete 验证
    4) 记录 success / expanded_num / act_num

  汇总成功率，与论文表1 (GPT-4o) 对照。

用法:
  PYTHONPATH=<repo> python reproduce/scripts/run_exp1_replay.py
依赖: numpy sympy pandas py_trees antlr4-python3-runtime shortuuid
完全离线，不需要任何 LLM key / GPU / 仿真器。
"""
import os
import re
import sys
import shutil
import pathlib
import traceback
from datetime import datetime

import pandas as pd

# ---- 路径 ----
REPO = pathlib.Path(__file__).resolve().parents[2]          # .../CABTO
EXP_DIR = REPO / "exps_bt_learning"
RESULTS_SRC = EXP_DIR / "exp1_results"                     # 缓存的 llm_output 来源（原 a_exp1_llm_bt_results）
TASKS_DIR = EXP_DIR / "a_exp1_llm_bt_tasks"                 # exec_lib 写入位置（被动态 import 的 python 包，保持原名）
REPRO_DIR = REPO / "reproduce"
OUT_DIR = REPRO_DIR / "results"
WORK_LIB_ROOT = REPRO_DIR / "_work_libs"                    # 重放时临时行为库根目录

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from replay_llm import ReplayLLM                            # noqa: E402
from exps_bt_learning.validate_bt_fun import validate_bt_fun  # noqa: E402

# ---- 任务定义 (与原始 1_main_cal_SR_models_feedback.py 完全一致) ----
task2name = {
    "task1": "Cover", "task2": "Blocks", "task3": "Clean", "task4": "Handover",
    "task5": "Pour", "task6": "HomeRearrangement", "task7": "MealPreparation",
}
task2objects = {
    "task1": ['a', 'place_a', 'b', 'place_b', 'c', 'place_c'],
    "task2": ['a', 'b', 'c', 'd'],
    "task3": ['left_franka', 'right_franka', 'left_lego', 'right_lego', "center_big_box", "box_board", "left_table", "right_table", "center_table"],
    "task4": ['left_franka', 'right_franka', 'left_box1', 'left_box2', 'right_box', "left_table", "right_table"],
    "task5": ['left_franka', 'right_franka', 'left_cup', 'right_cup', 'right_milk', 'right_milk', 'left_table', 'right_table'],
    "task6": ['fridge', 'pen', 'cabinet', "book", "banana", "table", "breakfast_table"],
    "task7": ['oven', 'chickenleg', 'soup', 'microwave', 'light', "radio", 'table', "pie", "breakfast_table"],
}
task2initial_state = {
    "task1": {'IsHandEmpty()', 'IsEmpty(place_a)', 'IsEmpty(place_b)', 'IsEmpty(place_c)', 'On(a,table)', 'On(b,table)', 'On(c,table)'},
    "task2": {'IsHandEmpty()', 'On(a,table)', 'On(b,table)', 'On(c,table)', 'On(d,table)'},
    "task3": {'IsHandEmpty(left_franka)', 'IsHandEmpty(right_franka)', 'On(left_franka,left_table)', 'On(right_franka,right_table)', 'On(left_lego,left_table)', 'On(right_lego,right_table)', 'On(center_big_box,center_table)', 'On(box_board,center_table)', "BigBoxNeedTwoFrankaHoldTogether()"},
    "task4": {'IsHandEmpty(left_franka)', 'IsHandEmpty(right_franka)', 'On(left_franka,left_table)', 'On(right_franka,right_table)', 'On(left_box1,left_table)', 'On(left_box2,left_table)', 'On(right_box,right_table)'},
    "task5": {'IsHandEmpty(left_franka)', 'IsHandEmpty(right_franka)', 'On(left_cup,left_table)', 'On(right_cup,right_table)', 'On(left_milk,left_table)', 'On(right_milk,right_table)', 'IsFull(left_milk)', 'IsFull(right_milk)', 'IsEmpty(left_cup)', 'IsEmpty(right_cup)', 'CupNeedGraspAndSupport()', 'CanGrasp(left_franka,left_milk)', 'CanGrasp(right_franka,right_milk)'},
    "task6": {'IsHandEmpty()', 'On(pen,breakfast_table)', 'On(book,table)', 'On(banana,table)', 'IsClosed(fridge)', 'IsClosed(cabinet)'},
    "task7": {'IsHandEmpty()', 'On(chickenleg,table)', 'On(soup,table)', 'IsSwitchedOff(oven)', 'IsSwitchedOff(microwave)', 'IsSwitchedOff(light)', "IsSwitchedOff(radio)", 'IsClosed(oven)', 'IsClosed(microwave)', 'On(pie,table)'},
}
task2goal_str = {
    "task1": ['On(a,place_a)', 'On(b,place_b) & On(c,place_c)', 'On(b,place_a) & On(a,place_c) & On(c,place_b)'],
    "task2": ['On(a,b)', 'On(c,b) & On(b,a)', 'On(a,c) & On(c,b) & On(b,d)'],
    "task3": ['In(left_lego,center_big_box)', 'In(right_lego,center_big_box)', 'On(center_big_box,box_board)'],
    "task4": ['On(left_box1,right_table)', 'On(left_box2,right_table)&On(right_box,left_table)', 'IsHolding(left_franka,left_box1)&IsHolding(right_franka,right_box)'],
    "task5": ['IsHalfFull(right_cup)', 'IsHalfFull(right_cup) & IsHalfFull(left_cup)', 'IsHalfFull(right_cup) & IsHalfFull(left_cup) & IsHandEmpty(left_franka) & IsHandEmpty(right_franka) & On(left_milk,left_table) & On(right_milk,right_table)'],
    "task6": ['In(book,cabinet) & IsOpened(cabinet)', 'On(book,breakfast_table)', 'In(banana,fridge)  & IsClosed(fridge)'],
    "task7": ['In(chickenleg,oven) & IsSwitchedOn(oven) & In(pie,oven)', 'IsSwitchedOn(microwave) & In(soup,microwave) ', 'IsOpened(microwave) & IsOpened(oven) & IsSwitchedOn(light) & IsSwitchedOn(radio) & On(pie,breakfast_table)'],
}

DIFFS = ["easy", "medium", "hard"]
TOTAL_TRY = 10
TASK_NAMES = ["task1", "task2", "task3", "task4", "task5", "task6", "task7"]


def extract_and_write_lib(answer, behavior_lib_path):
    """复刻原始 llm_generate_lib_func 的代码抽取逻辑，把 answer 中的
    OGAction/OGCondition 类写到 behavior_lib_path/{Action,Condition}/。"""
    action_dir = os.path.join(behavior_lib_path, "Action")
    condition_dir = os.path.join(behavior_lib_path, "Condition")
    os.makedirs(action_dir, exist_ok=True)
    os.makedirs(condition_dir, exist_ok=True)
    # 清空
    for d in (action_dir, condition_dir):
        for fn in os.listdir(d):
            fp = os.path.join(d, fn)
            if os.path.isfile(fp) and fn.endswith(".py"):
                os.remove(fp)

    code_blocks = re.findall(r"```python\n(.*?)```", answer, re.DOTALL)
    class_pattern = r'class\s+(\w+)\s*\((\w+)\):'

    # 计算 _base 的导入路径前缀 (基于 behavior_lib_path 相对 EXP_DIR)
    rel_path = os.path.relpath(behavior_lib_path, str(EXP_DIR))
    path_parts = rel_path.split(os.path.sep)
    if 'a_exp1_llm_bt_tasks' in path_parts:
        idx = path_parts.index('a_exp1_llm_bt_tasks')
        _lib_path = '.'.join(path_parts[:idx + 1])
    else:
        _lib_path = 'a_exp1_llm_bt_tasks'

    n_action = n_condition = 0
    for code in code_blocks:
        for match in re.finditer(class_pattern, code):
            class_name = match.group(1)
            base_class = match.group(2)
            if base_class == 'OGAction':
                class_type = "Action"
                import_statement = f"from exps_bt_learning.{_lib_path}._base.OGAction import OGAction\nimport itertools\n\n"
            elif base_class == 'OGCondition':
                class_type = "Condition"
                import_statement = f"from exps_bt_learning.{_lib_path}._base.OGCondition import OGCondition\nimport itertools\n\n"
            else:
                continue
            start_index = code.find(f"class {class_name}")
            end_index = code.find("class ", start_index + 1)
            if end_index == -1:
                end_index = len(code)
            single = code[start_index:end_index].strip()
            single = re.sub(r'from\s+\w+\s+import\s+\w+', '', single)
            class_dir = os.path.join(behavior_lib_path, class_type)
            file_path = os.path.join(class_dir, f"{class_name}.py")
            counter = 1
            while os.path.exists(file_path):
                file_path = os.path.join(class_dir, f"{class_name}{counter}.py")
                if class_type == "Action":
                    single = re.sub(r'class\s+(\w+)\s*\(\s*OGAction\s*\):',
                                    f'class {class_name}{counter}(OGAction):', single)
                counter += 1
            with open(file_path, "w") as f:
                f.write(import_statement + single)
            if class_type == "Action":
                n_action += 1
            else:
                n_condition += 1
    return n_action, n_condition


def ensure_base_symlink(task_lib_root):
    """exec_lib 用到的 _base (OGAction/OGCondition) 必须在 a_exp1_llm_bt_tasks 下。
    我们把生成的 exec_lib 直接写在真实 a_exp1_llm_bt_tasks/<task>/exec_lib 里，
    复用现成的 _base，因此无需 symlink。此函数保留占位。"""
    return


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    llm = ReplayLLM(results_dir=str(RESULTS_SRC), request_model="gpt-4o", verbose=False)
    ts = datetime.now().strftime("%Y%m%d%H%M")

    rows = []
    for task in TASK_NAMES:
        objects = task2objects[task]
        initial_state = set(task2initial_state[task])
        goals = task2goal_str[task]
        # 行为库写到真实 tasks 目录(复用其中的 _base)
        behavior_lib_path = os.path.join(str(TASKS_DIR), task, "exec_lib")
        for try_idx in range(1, TOTAL_TRY + 1):
            llm.set_context(task, try_idx)
            try:
                answer = llm.request("")  # 重放：返回缓存的 llm_output
            except FileNotFoundError as e:
                print(f"[skip] {task} try{try_idx}: {e}")
                continue
            try:
                n_act, n_cond = extract_and_write_lib(answer, behavior_lib_path)
            except Exception as e:
                print(f"[extract-fail] {task} try{try_idx}: {e}")
                for di, g in enumerate(goals):
                    rows.append(dict(task=task, task_name=task2name[task], try_idx=try_idx,
                                     difficulty=DIFFS[di], goal=g, success=0,
                                     expanded_num=-1, act_num=-1,
                                     n_action=0, n_condition=0, note="extract_fail"))
                continue
            for di, goal in enumerate(goals):
                rec = dict(task=task, task_name=task2name[task], try_idx=try_idx,
                           difficulty=DIFFS[di], goal=goal, n_action=n_act,
                           n_condition=n_cond, note="")
                try:
                    r = validate_bt_fun(behavior_lib_path=behavior_lib_path,
                                        goal_str=goal, cur_cond_set=set(initial_state),
                                        output_dir=None)
                    error = r[0]
                    rec.update(success=0 if error else 1,
                               expanded_num=r[2], act_num=r[3])
                except Exception as e:
                    rec.update(success=0, expanded_num=-1, act_num=-1,
                               note=f"validate_err:{type(e).__name__}")
                rows.append(rec)
            print(f"[done] {task} ({task2name[task]}) try{try_idx}: "
                  f"acts={n_act} conds={n_cond} "
                  f"results={[rows[-3]['success'], rows[-2]['success'], rows[-1]['success']]}")

    df = pd.DataFrame(rows)
    detail_csv = os.path.join(OUT_DIR, f"exp1_replay_detail_{ts}.csv")
    df.to_csv(detail_csv, index=False)
    print(f"\nSaved detail -> {detail_csv}  ({len(df)} rows)")

    # ---- 汇总成功率 ----
    summ = []
    for task in TASK_NAMES:
        sub = df[df.task == task]
        row = {"task": task, "name": task2name[task]}
        for d in DIFFS:
            s = sub[sub.difficulty == d]
            row[f"{d}_SR"] = round(100.0 * s.success.mean(), 1) if len(s) else 0.0
        row["all_SR"] = round(100.0 * sub.success.mean(), 1) if len(sub) else 0.0
        row["n_try"] = sub.try_idx.nunique()
        summ.append(row)
    sdf = pd.DataFrame(summ)
    # overall
    overall = {"task": "ALL", "name": "Overall"}
    for d in DIFFS:
        s = df[df.difficulty == d]
        overall[f"{d}_SR"] = round(100.0 * s.success.mean(), 1) if len(s) else 0.0
    overall["all_SR"] = round(100.0 * df.success.mean(), 1) if len(df) else 0.0
    overall["n_try"] = TOTAL_TRY
    sdf = pd.concat([sdf, pd.DataFrame([overall])], ignore_index=True)
    summ_csv = os.path.join(OUT_DIR, f"exp1_replay_summary_{ts}.csv")
    sdf.to_csv(summ_csv, index=False)
    print(f"Saved summary -> {summ_csv}")
    print("\n=== 实验1 离线重放成功率 (%) ===")
    print(sdf.to_string(index=False))
    return detail_csv, summ_csv


if __name__ == "__main__":
    run()

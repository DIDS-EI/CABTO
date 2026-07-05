"""
实验3 (Cross-level refinement) 离线复现 / 校验
=============================================
实验3 在论文里需要：视觉 LLM(gpt-4o) + failure 图片 + 仿真执行上下文，
这些在本机无法离线获得。但论文作者已保存了完整产物：
  - a_exp3_cross_feedback_results/results.csv : 每个 try 的 initial/goal/feedback_times/success
  - behavior_lib_case1/try_N/ : 经过【跨层精化修正后】的动作库
  - behavior_lib_case2/try_N/ : 另一个 case 的修正库

本脚本做"离线复算 + 一致性校验"：
  1) 读取 results.csv 的真实记录
  2) 对每个 try，加载对应的【修正后动作库】，用 obtea 重跑 sound&complete 验证
  3) 把"我们重跑出的 success" 与 "作者记录的 success" 对照，验证可复现性
  4) 复算"反馈前(feedback=0) 成功率" vs "总成功率"，对应论文表3 的结论：
     跨层精化反馈能把缺陷动作模型修正为可成功规划。

完全离线，不需要 LLM key / 图片 / GPU / 仿真。
"""
import os
import csv
import sys
import ast
import pathlib
from datetime import datetime

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]          # .../CABTO（可移植）
EXP_DIR = REPO / "exps_bt_learning"
EXP3_DIR = EXP_DIR / "exp3_results"                          # 原 a_exp3_cross_feedback_results
OUT_DIR = REPO / "reproduce" / "results"

sys.path.insert(0, str(REPO))
from exps_bt_learning.validate_bt_fun import validate_bt_fun  # noqa: E402


def load_results_csv():
    path = EXP3_DIR / "results.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    return rows


def find_fixed_lib(case_idx, try_idx):
    """返回该 case/try 对应的修正库目录(若存在)。
    behavior_lib_case{c}/try_{n}/ 。case1 有 try_1..10, case2 只有 try_1,2。"""
    d = EXP3_DIR / f"behavior_lib_case{case_idx}" / f"try_{try_idx}"
    return d if d.is_dir() else None


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_results_csv()
    ts = datetime.now().strftime("%Y%m%d%H%M")

    out = []
    for r in rows:
        case_name = r["case_name"]          # e.g. case1_try1
        try_idx = int(r["try_idx"])
        case_idx = 1 if case_name.startswith("case1") else 2
        goal = r["goal"]
        initial = ast.literal_eval(r["initial_state"])
        rec_feedback = int(r["feedback_times"])
        rec_success = (r["success"] == "True")

        lib = find_fixed_lib(case_idx, try_idx)
        rerun_success = None
        expanded = act = -1
        note = ""
        if lib is None:
            note = "no_fixed_lib"
        else:
            try:
                res = validate_bt_fun(behavior_lib_path=str(lib), goal_str=goal,
                                      cur_cond_set=set(initial), output_dir=None)
                rerun_success = (not res[0])
                expanded, act = res[2], res[3]
            except Exception as e:
                note = f"rerun_err:{type(e).__name__}"

        match = (rerun_success == rec_success) if rerun_success is not None else None
        out.append(dict(
            case=f"case{case_idx}", try_idx=try_idx, goal=goal,
            recorded_feedback_times=rec_feedback,
            recorded_success=int(rec_success),
            rerun_success=("" if rerun_success is None else int(rerun_success)),
            match=("" if match is None else int(match)),
            expanded_num=expanded, act_num=act, note=note,
        ))
        print(f"[exp3] case{case_idx} try{try_idx}: recorded={int(rec_success)} "
              f"feedback={rec_feedback} -> rerun={rerun_success} match={match} {note}")

    df = pd.DataFrame(out)
    detail_csv = os.path.join(str(OUT_DIR), f"exp3_replay_detail_{ts}.csv")
    df.to_csv(detail_csv, index=False)

    # ---- 汇总：反馈前后成功率 (基于作者记录) + 重跑一致率 ----
    n = len(df)
    rec_succ = df.recorded_success.sum()
    # 反馈前(feedback==0)子集成功率
    no_fb = df[df.recorded_feedback_times == 0]
    with_fb = df[df.recorded_feedback_times > 0]
    valid_match = df[df["match"] != ""]
    match_rate = 100.0 * (valid_match["match"].astype(int).mean()) if len(valid_match) else 0.0

    summary = {
        "案例": "Lift(big_box,board) [case1]",
        "总try数": n,
        "记录_总成功率(%)": round(100.0 * rec_succ / n, 1),
        "记录_无反馈成功率(%)": round(100.0 * no_fb.recorded_success.mean(), 1) if len(no_fb) else 0.0,
        "记录_含反馈条目数": len(with_fb),
        "含反馈条目成功率(%)": round(100.0 * with_fb.recorded_success.mean(), 1) if len(with_fb) else 0.0,
        "离线重跑_与记录一致率(%)": round(match_rate, 1),
    }
    sdf = pd.DataFrame([summary])
    summ_csv = os.path.join(str(OUT_DIR), f"exp3_replay_summary_{ts}.csv")
    sdf.to_csv(summ_csv, index=False)

    print("\n=== 实验3 离线复算/校验 ===")
    print(df.to_string(index=False))
    print("\n--- 汇总 ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nSaved -> {detail_csv}\nSaved -> {summ_csv}")
    return str(detail_csv), str(summ_csv)


if __name__ == "__main__":
    run()

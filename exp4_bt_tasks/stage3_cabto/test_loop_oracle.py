"""F. 闭环测试入口
====================
1) oracle 闭环（定位走真值）—— 硬指标：success=True 且 stacked_on 真值为真。
2) qwen 闭环（VLM 真打点 + VLM 真校验）—— 记录 VLM 表现，定位不准导致失败可接受。

用法：
  PYTHONPATH=. <PY> test_loop_oracle.py oracle
  PYTHONPATH=. <PY> test_loop_oracle.py qwen
  PYTHONPATH=. <PY> test_loop_oracle.py oracle_inject   # 人为失败验证 self-correction
"""

import sys

from loop import run_closed_loop

TASK = "stack the green block on the yellow block"
OBJECTS = ["green_block", "yellow_block", "blue_block"]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "oracle"
    if mode == "oracle":
        r = run_closed_loop(TASK, OBJECTS, max_rounds=3, backend="oracle",
                            render=True)
        assert r["success"], "oracle 闭环未成功"
        assert r["final_stacked_on_green_yellow"], "stacked_on 真值为假"
        print("\n*** ORACLE 闭环硬指标通过 ***")
    elif mode == "oracle_inject":
        r = run_closed_loop(TASK, OBJECTS, max_rounds=3, backend="oracle",
                            render=True, inject_failure=True)
        print(f"\n[inject] rounds_used={r['rounds_used']} success={r['success']}")
    elif mode == "qwen":
        r = run_closed_loop(TASK, OBJECTS, max_rounds=2, backend="qwen",
                            render=True)
        print(f"\n[qwen] success={r['success']} "
              f"stacked={r['final_stacked_on_green_yellow']}")
    else:
        raise SystemExit(f"未知模式 {mode}")


if __name__ == "__main__":
    main()

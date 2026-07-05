"""handover 可达性：左臂抓 box、抬到交接点；右臂到交接点接物。"""
import os, sys
import numpy as np
import mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp4_env import Exp4Env, TABLE_TOP_Z, HOME_QPOS
import scene_handover as sh

OUT = "/tmp/exp4_tasks/out_handover"
os.makedirs(OUT, exist_ok=True)


def main():
    env = Exp4Env(sh.build_scene_xml(), arms=[("_L", -np.pi/2), ("_R", np.pi/2)],
                  obj_names=["green_box", "dec_box", "dec_btl"], render=True, img_size=560)
    env.reset()
    aL, aR = env.arms[0], env.arms[1]
    box = env.get_object_pose("green_box")[0]
    print(f"[init] green_box={np.round(box,3)}")
    env.save_png(os.path.join(OUT, "handover_init.png"), "overview")

    def reach(arm, name, pos):
        q = arm.solve_ik(np.asarray(pos, float), arm.grasp_quat)
        s = env.ik_data; s.qpos[:] = env.d.qpos
        for k, adr in enumerate(arm.arm_qadr):
            s.qpos[adr] = q[k]
        mujoco.mj_kinematics(env.m, s); mujoco.mj_comPos(env.m, s)
        got = s.site_xpos[arm.tcp_sid].copy()
        pe = float(np.linalg.norm(np.asarray(pos, float) - got))
        print(f"  [{arm.s}] {name} {np.round(pos,3)} -> pe={pe:.4f}")
        return pe

    hx, hy = sh.HANDOVER_XY
    hz = TABLE_TOP_Z + 0.22
    print("[reach]")
    reach(aL, "grasp_box", [box[0], box[1], box[2] + 0.01])
    reach(aL, "above_box", [box[0], box[1], TABLE_TOP_Z + 0.30])
    reach(aL, "handover_hold", [hx, hy + 0.01, hz])     # 左臂在交接点偏 +y 持物
    reach(aR, "handover_recv", [hx, hy - 0.04, hz])     # 右臂从 -y 侧接近
    reach(aR, "carry_away", [hx, hy - 0.16, TABLE_TOP_Z + 0.26])  # 右臂带走
    print("[saved]", OUT)


if __name__ == "__main__":
    main()

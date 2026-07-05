"""扫描左/右臂在桌面抓取高度(z=0.46)的可达 xy 区域，确定双臂物体安全摆位。"""
import os, sys
import numpy as np
import mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp4_env import Exp4Env, HOME_QPOS
import scene_pour as sp


def main():
    env = Exp4Env(sp.build_scene_xml(), arms=[("_L", -np.pi/2), ("_R", np.pi/2)],
                  obj_names=["can", "content"], render=False)
    env.reset()
    aL, aR = env.arms[0], env.arms[1]

    def reach_ok(arm, x, y, z):
        tgt = np.array([x, y, z])
        q = arm.solve_ik(tgt, arm.grasp_quat)
        s = env.ik_data; s.qpos[:] = env.d.qpos
        for k, adr in enumerate(arm.arm_qadr):
            s.qpos[adr] = q[k]
        mujoco.mj_kinematics(env.m, s); mujoco.mj_comPos(env.m, s)
        got = s.site_xpos[arm.tcp_sid]
        return float(np.linalg.norm(tgt - got))

    z = 0.47
    xs = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    ys = [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30]
    print(f"=== 抓取高度 z={z} 可达性 pe (✓<0.02) ===")
    for arm, tag in [(aL, "L(base y=+0.55)"), (aR, "R(base y=-0.55)")]:
        print(f"\n[{tag}]   x\\y " + " ".join(f"{y:+.2f}" for y in ys))
        for x in xs:
            row = []
            for y in ys:
                pe = reach_ok(arm, x, y, z)
                row.append(" ✓  " if pe < 0.02 else (" .  " if pe < 0.06 else " x  "))
            print(f"        x={x:.2f} " + "".join(row))


if __name__ == "__main__":
    main()

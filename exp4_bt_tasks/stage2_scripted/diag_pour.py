"""pour 可达性诊断：渲染初始场景 + 测左臂够 can/cup、右臂够 cup 旁。"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp4_env import Exp4Env, TABLE_TOP_Z, GRASP_QUAT
import scene_pour as sp

OUT = "/tmp/exp4_tasks/out_pour"
os.makedirs(OUT, exist_ok=True)


def main():
    env = Exp4Env(sp.build_scene_xml(), arms=[("_L", -np.pi/2), ("_R", np.pi/2)],
                  obj_names=["can", "content"], render=True, img_size=560)
    env.reset()
    aL, aR = env.arms[0], env.arms[1]
    print(f"[arms] L.s={aL.s} R.s={aR.s}")
    print(f"[init] can={np.round(env.get_object_pose('can')[0],3)} "
          f"content={np.round(env.get_object_pose('content')[0],3)} "
          f"cup={np.round(env.get_body_pos('cup'),3)}")
    print(f"[ee0]  L={np.round(aL.get_ee_pose()[0],3)} R={np.round(aR.get_ee_pose()[0],3)}")
    env.save_png(os.path.join(OUT, "pour_init_overview.png"), "overview")
    env.save_png(os.path.join(OUT, "pour_init_front.png"), "front")

    # 可达性：用 _lm_solve 直接测各臂能否到达关键目标（solve_ik 只回 best_q）
    def reach(arm, name, pos, quat=None):
        quat = arm.grasp_quat if quat is None else quat
        s = env.ik_data
        s.qpos[:] = env.d.qpos
        from exp4_env import HOME_QPOS
        _, pe, re = arm._lm_solve(s, HOME_QPOS.copy(), np.asarray(pos, float), quat, 150, 0.5)
        print(f"  [{arm.s}] reach {name} {np.round(pos,3)} -> pe={pe:.4f} re={re:.4f}")
        return pe

    can = env.get_object_pose("can")[0]
    cup = env.get_body_pos("cup")
    print("[reach-test]")
    # 左臂：can 上方 / can 抓取高度 / cup 上方（倒水位）
    reach(aL, "above_can", [can[0], can[1], TABLE_TOP_Z + 0.30])
    reach(aL, "grasp_can", [can[0], can[1], can[2] + 0.03])
    reach(aL, "above_cup", [cup[0], cup[1], TABLE_TOP_Z + 0.34])
    # 右臂：cup 旁（扶位） / cup 上方
    reach(aR, "beside_cup", [cup[0], cup[1] - (sp.CUP_R_IN + 0.08), TABLE_TOP_Z + 0.10])
    reach(aR, "above_cup", [cup[0], cup[1], TABLE_TOP_Z + 0.30])
    # 用完整 solve_ik（多 restart）复测 grasp_can（关键抓取位）
    print("[solve_ik full-restart re-test]")
    for zoff in (0.00, 0.02, 0.04):
        tgt = np.array([can[0], can[1], can[2] + zoff])
        q = aL.solve_ik(tgt, aL.grasp_quat)
        s = env.ik_data; s.qpos[:] = env.d.qpos
        for k, adr in enumerate(aL.arm_qadr):
            s.qpos[adr] = q[k]
        import mujoco as _mj
        _mj.mj_kinematics(env.m, s); _mj.mj_comPos(env.m, s)
        got = s.site_xpos[aL.tcp_sid].copy()
        print(f"  grasp_can z+{zoff:.2f} tgt={np.round(tgt,3)} got={np.round(got,3)} "
              f"pe={np.linalg.norm(tgt-got):.4f}")
    print("[saved]", OUT)


if __name__ == "__main__":
    main()

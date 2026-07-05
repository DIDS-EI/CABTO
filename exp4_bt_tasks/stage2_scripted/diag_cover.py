"""诊断 cover 抓取：逐阶段打印 tcp 高度，确认 IK 是否能下到把手。"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp4_env import Exp4Env, TABLE_TOP_Z
from arm_skills import ArmSkills
import scene_cover as sc


def main():
    env = Exp4Env(sc.build_scene_xml(), arms=[("", 0.0)], obj_names=["lid"],
                  render=False, img_size=256)
    env.reset()
    arm = env.arms[0]
    arm.grasp_z_tol = 0.10          # 抓高把手：放宽吸附垂直容差
    sk = ArmSkills(env, arm, recorder=None, block_half=sc.LID_HH)

    lid0 = env.get_object_pose("lid")[0]
    ee0 = arm.get_ee_pose()[0]
    print(f"[init] lid={np.round(lid0,3)} ee={np.round(ee0,3)}")

    handle_top = lid0[2] + sc.LID_HH + 2 * sc.HANDLE_HH
    grasp_z = handle_top - 0.30 * (2 * sc.HANDLE_HH)
    print(f"[plan] handle_top={handle_top:.3f} grasp_z={grasp_z:.3f}")

    sk.approach(lid0)
    print(f"[approach] ee={np.round(arm.get_ee_pose()[0],3)}")
    sk.align_xy(lid0)
    print(f"[align]    ee={np.round(arm.get_ee_pose()[0],3)}")
    sk.descend_to(grasp_z, lock_xy=lid0[:2])
    ee_d = arm.get_ee_pose()[0]
    print(f"[descend]  ee={np.round(ee_d,3)}  (target_z={grasp_z:.3f}, "
          f"dz_err={ee_d[2]-grasp_z:+.3f})")

    # descend 后看 finger_mid 与 lid 中心的几何关系（吸附判定依据）
    mid = arm._finger_mid()
    b = env.obj_qadr["lid"]
    lid_c = env.d.qpos[b:b + 3]
    horiz = float(np.linalg.norm(mid[:2] - lid_c[:2]))
    vert = abs(float(mid[2] - lid_c[2]))
    print(f"[geom]     finger_mid={np.round(mid,3)} lid_c={np.round(lid_c,3)} "
          f"horiz={horiz:.4f} vert={vert:.4f} (need horiz<0.035 vert<0.06)")
    print(f"[grip_w]   width_before_grasp={arm.get_gripper_width():.4f}")

    sk.grasp(lid0[:2], n=28)
    print(f"[grasp]    width_after={arm.get_gripper_width():.4f} "
          f"active_weld={arm._active_weld_name()} grip_cmd={arm._grip_cmd}")
    sk.lift(top=TABLE_TOP_Z + 0.30)
    sk.record_grab_offset("lid")
    print(f"[lift]     ee={np.round(arm.get_ee_pose()[0],3)} "
          f"lid={np.round(env.get_object_pose('lid')[0],3)} "
          f"holding={arm.is_holding('lid')} grab_dz={sk.grab_dz:.3f}")
    print(f"[grip_w]   width={arm.get_gripper_width():.4f}")


if __name__ == "__main__":
    main()

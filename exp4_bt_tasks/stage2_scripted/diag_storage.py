"""检查 storage：item1/item2 抓取位、carton 两半放置位对各臂可达性。"""
import os, sys
import numpy as np
import mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp4_env import Exp4Env
import scene_storage as ss


def main():
    env = Exp4Env(ss.build_scene_xml(), arms=[("_L", -np.pi/2), ("_R", np.pi/2)],
                  obj_names=["item_g1", "item_g2"], render=False)
    env.reset()
    aL, aR = env.arms[0], env.arms[1]

    def reach(arm, x, y, z):
        tgt = np.array([x, y, z])
        q = arm.solve_ik(tgt, arm.grasp_quat)
        s = env.ik_data; s.qpos[:] = env.d.qpos
        for k, adr in enumerate(arm.arm_qadr):
            s.qpos[adr] = q[k]
        mujoco.mj_kinematics(env.m, s); mujoco.mj_comPos(env.m, s)
        return float(np.linalg.norm(tgt - s.site_xpos[arm.tcp_sid]))

    carton = env.get_body_pos("carton")
    print(f"carton body pos = {np.round(carton,3)}  CARTON_TOP_Z={ss.CARTON_TOP_Z}")
    i1 = env.get_object_pose("item_g1")[0]
    i2 = env.get_object_pose("item_g2")[0]
    print(f"item_g1={np.round(i1,3)} item_g2={np.round(i2,3)}")

    # 抓取高度 ~ item 中心 z
    grab_z = i1[2]
    # 放置悬停高度（箱口上方）
    hover_z = ss.CARTON_TOP_Z + 0.12
    drop_L = (ss.CARTON_XY[0], ss.CARTON_XY[1] + 0.035)
    drop_R = (ss.CARTON_XY[0], ss.CARTON_XY[1] - 0.035)

    print("\n--- 抓取位可达 (z=%.3f) ---" % grab_z)
    print(f"  L grab item1@{ss.ITEM1_XY}: pe={reach(aL,*ss.ITEM1_XY,grab_z):.4f}")
    print(f"  R grab item2@{ss.ITEM2_XY}: pe={reach(aR,*ss.ITEM2_XY,grab_z):.4f}")
    print("\n--- 放置悬停位可达 (z=%.3f) ---" % hover_z)
    print(f"  L hover dropL@{drop_L}: pe={reach(aL,*drop_L,hover_z):.4f}")
    print(f"  R hover dropR@{drop_R}: pe={reach(aR,*drop_R,hover_z):.4f}")
    print("\n--- 放置下沉位可达 (z=%.3f, 箱内) ---" % (ss.CARTON_TOP_Z - 0.01))
    zin = ss.CARTON_TOP_Z - 0.01
    print(f"  L dropL@{drop_L}: pe={reach(aL,*drop_L,zin):.4f}")
    print(f"  R dropR@{drop_R}: pe={reach(aR,*drop_R,zin):.4f}")


if __name__ == "__main__":
    main()

"""隔离诊断：分别让左右臂执行抓取段，打印 descend 后 tcp 实际 xy 与 item 偏差。
目的：定位右臂没抓到的原因（伺服稳态偏置方向不对称）。"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp4_env import Exp4Env
from arm_skills import ArmSkills
import scene_storage as ss


def try_grab(env, sk, arm, item):
    p0 = env.get_object_pose(item)[0]
    sk.approach(p0)
    sk.align_xy(p0)
    # align 后悬停 tcp
    ee_hover = sk.ee()
    sk.descend_to(p0[2] + 0.004, lock_xy=p0[:2])
    ee_desc = sk.ee()
    op_before = env.get_object_pose(item)[0]
    sk.grasp(p0[:2], n=30)
    ee_grasp = sk.ee()
    holding = arm.is_holding(item)
    gw = arm.get_gripper_width()
    op_after = env.get_object_pose(item)[0]
    print(f"  item0     = {np.round(p0,4)}")
    print(f"  ee hover  = {np.round(ee_hover,4)}  (xy off {np.round(ee_hover[:2]-p0[:2],4)})")
    print(f"  ee desc   = {np.round(ee_desc,4)}  (xy off {np.round(ee_desc[:2]-p0[:2],4)})")
    print(f"  ee grasp  = {np.round(ee_grasp,4)}  (xy off {np.round(ee_grasp[:2]-op_before[:2],4)})")
    print(f"  item after= {np.round(op_after,4)}")
    print(f"  holding={holding} grip_w={gw:.4f}")
    print(f"  bias_world={np.round(arm.bias_world,4)}")


def main():
    env = Exp4Env(ss.build_scene_xml(), arms=[("_L", -np.pi/2), ("_R", np.pi/2)],
                  obj_names=["item_g1", "item_g2"], render=False)
    env.reset()
    aL, aR = env.arms[0], env.arms[1]
    aL.grasp_z_tol = 0.12
    aR.grasp_z_tol = 0.12
    skL = ArmSkills(env, aL, block_half=ss.ITEM_HALF[2])
    skR = ArmSkills(env, aR, block_half=ss.ITEM_HALF[2])
    print("[左臂抓 item_g1]")
    try_grab(env, skL, aL, "item_g1")
    print("[右臂抓 item_g2]")
    try_grab(env, skR, aR, "item_g2")


if __name__ == "__main__":
    main()

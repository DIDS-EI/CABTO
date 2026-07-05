"""渲染 5 个任务场景的静态预览图（手臂置于自然预备姿态）。"""
import os, sys
import mujoco
import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_builder as sb

# 自然预备姿态：肘抬起，末端悬在前方桌面上空（比 home 更舒展、不戳桌）
READY = [0.0, -0.30, 0.0, -2.0, 0.0, 1.78, 0.785]


def set_ready(m, d):
    for suf in ("", "_L", "_R"):
        for i in range(1, 8):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}{suf}")
            if jid >= 0:
                d.qpos[m.jnt_qposadr[jid]] = READY[i - 1]
        # 夹爪张开
        for fj in (f"finger_joint1{suf}", f"finger_joint2{suf}"):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, fj)
            if jid >= 0:
                d.qpos[m.jnt_qposadr[jid]] = 0.04


def render(task, scene_dir, out_dir, w=1100, h=520):
    m = mujoco.MjModel.from_xml_path(os.path.join(scene_dir, f"scene_{task}.xml"))
    d = mujoco.MjData(m)
    set_ready(m, d)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, height=h, width=w)
    r.update_scene(d, camera="overview")
    img = r.render()
    p = os.path.join(out_dir, f"static_{task}.png")
    imageio.imwrite(p, img)
    r.close()
    return p


if __name__ == "__main__":
    scene_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/exp4_scenes"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/exp4_scenes"
    os.makedirs(out_dir, exist_ok=True)
    for t in sb.BUILDERS:
        print(render(t, scene_dir, out_dir))

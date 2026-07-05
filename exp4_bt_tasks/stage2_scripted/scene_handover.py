"""handover 双臂场景（连续 3 物品交接放桌版）：
- 3 个绿色长方盒 box0/box1/box2，初始一字排在左臂可达区（+y 侧）。
- 每个盒有两个 weld：grasp_weld_L_box{i}（左臂）、grasp_weld_R_box{i}（右臂）。
- 任务：左臂逐个抓盒→放到桌中交接点→右臂接住→右臂放到右区(-y)桌面三个目标位。
  连做 3 次，最终 3 个盒都从 +y 区被搬运到 -y 区桌面（双臂接力交接）。
- 成功：3 个盒都最终落在各自 -y 目标位附近、贴桌面（z≈桌面+半高），且都已离开初始 +y 区。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_dual_common as dc

TABLE_H = dc.TABLE_H

BOX_HALF = (0.028, 0.024, 0.058)        # 竖长方盒；y半宽0.024(外宽0.048)<间距0.06留余量防漂移
N_BOX = 3
# 三盒初始一字排在 +y（左臂可达区，实测左臂舒适抓取 y≤0.14）。
# 盒外宽0.048，间距0.06>外宽，初始化抖动不会相互推挤漂移。y 全落 [0.03,0.15] 可达区。
BOX_XYS = [(0.29, 0.03), (0.29, 0.09), (0.29, 0.15)]
HANDOVER_XY = (0.30, 0.0)               # 两臂重叠交接点
# 右区(-y)三个放置目标位，一字排（实测右臂搬运 y∈[-0.08,-0.20] 全可达）
PLACE_XYS = [(0.30, -0.08), (0.30, -0.14), (0.30, -0.20)]


def build_scene_xml():
    objs = []
    welds = []
    for i, (bx, by) in enumerate(BOX_XYS):
        box_z = TABLE_H + BOX_HALF[2] + 0.002
        hx, hy, hz = BOX_HALF
        name = f"box{i}"
        objs.append(
            f"""    <body name="{name}" pos="{bx} {by} {box_z}">
      <freejoint name="{name}_free"/>
      <geom type="box" size="{hx} {hy} {hz}" material="mat_emerald" mass="0.05" friction="1.0 0.02 0.001"/>
    </body>""")
        welds.append(
            f'    <weld name="grasp_weld_L_{name}" body1="hand_L" body2="{name}" '
            'active="false" solimp="0.95 0.99 0.001" solref="0.005 1"/>')
        welds.append(
            f'    <weld name="grasp_weld_R_{name}" body1="hand_R" body2="{name}" '
            'active="false" solimp="0.95 0.99 0.001" solref="0.005 1"/>')
    return dc.make_dual_xml("exp4_handover", "\n".join(objs), "\n".join(welds))


if __name__ == "__main__":
    import mujoco
    xml = build_scene_xml()
    m = mujoco.MjModel.from_xml_string(xml)
    print(f"[ok] handover 编译成功 nq={m.nq} nv={m.nv} neq={m.neq}")
    print(f"  boxes={BOX_XYS} -> places={PLACE_XYS}")

"""
Panda Franka body XML 生成器（可加后缀、可固定基座放任意位置）
=================================================================
从 franka_mobile_scene.xml 抽取的 Panda 7-DoF + 夹爪 body 链，封装成函数，
便于在一个场景里放置 1 个或 2 个 Franka（双臂任务）。

关键改动（相对原 mobile_scene）：
  - 基座固定：直接用 <body pos=...> 把 link0 钉在桌边，不再用 freejoint+weld+mocap
    （桌面操作任务双臂底座不需要移动，固定更稳更简单）。
  - 命名加后缀 suffix（"" / "_L" / "_R"），让两个 Franka 的 joint/actuator/site 不冲突。
  - tcp site 命名为 f"tcp{suffix}"，actuator 命名 f"actuator{i}{suffix}"。

资产（mesh）只需在 <asset> 里声明一次（mesh 名不带后缀，两臂共享同一批 mesh）。
"""


def panda_asset_block():
    """返回 <asset> 内的 mesh 声明（两臂共享，只声明一次）。"""
    lines = [
        '    <mesh name="link0_c" file="link0.stl"/>',
        '    <mesh name="link1_c" file="link1.stl"/>',
        '    <mesh name="link2_c" file="link2.stl"/>',
        '    <mesh name="link3_c" file="link3.stl"/>',
        '    <mesh name="link4_c" file="link4.stl"/>',
        '    <mesh name="link5_c0" file="link5_collision_0.obj"/>',
        '    <mesh name="link5_c1" file="link5_collision_1.obj"/>',
        '    <mesh name="link5_c2" file="link5_collision_2.obj"/>',
        '    <mesh name="link6_c" file="link6.stl"/>',
        '    <mesh name="link7_c" file="link7.stl"/>',
        '    <mesh name="hand_c" file="hand.stl"/>',
        '    <mesh file="link0_0.obj"/><mesh file="link0_1.obj"/><mesh file="link0_2.obj"/>',
        '    <mesh file="link0_3.obj"/><mesh file="link0_4.obj"/><mesh file="link0_5.obj"/>',
        '    <mesh file="link0_7.obj"/><mesh file="link0_8.obj"/><mesh file="link0_9.obj"/>',
        '    <mesh file="link0_10.obj"/><mesh file="link0_11.obj"/>',
        '    <mesh file="link1.obj"/><mesh file="link2.obj"/>',
        '    <mesh file="link3_0.obj"/><mesh file="link3_1.obj"/><mesh file="link3_2.obj"/><mesh file="link3_3.obj"/>',
        '    <mesh file="link4_0.obj"/><mesh file="link4_1.obj"/><mesh file="link4_2.obj"/><mesh file="link4_3.obj"/>',
        '    <mesh file="link5_0.obj"/><mesh file="link5_1.obj"/><mesh file="link5_2.obj"/>',
        '    <mesh file="link6_0.obj"/><mesh file="link6_1.obj"/><mesh file="link6_2.obj"/><mesh file="link6_3.obj"/>',
        '    <mesh file="link6_4.obj"/><mesh file="link6_5.obj"/><mesh file="link6_6.obj"/><mesh file="link6_7.obj"/>',
        '    <mesh file="link6_8.obj"/><mesh file="link6_9.obj"/><mesh file="link6_10.obj"/><mesh file="link6_11.obj"/>',
        '    <mesh file="link6_12.obj"/><mesh file="link6_13.obj"/><mesh file="link6_14.obj"/><mesh file="link6_15.obj"/>',
        '    <mesh file="link6_16.obj"/>',
        '    <mesh file="link7_0.obj"/><mesh file="link7_1.obj"/><mesh file="link7_2.obj"/><mesh file="link7_3.obj"/>',
        '    <mesh file="link7_4.obj"/><mesh file="link7_5.obj"/><mesh file="link7_6.obj"/><mesh file="link7_7.obj"/>',
        '    <mesh file="hand_0.obj"/><mesh file="hand_1.obj"/><mesh file="hand_2.obj"/>',
        '    <mesh file="hand_3.obj"/><mesh file="hand_4.obj"/>',
        '    <mesh file="finger_0.obj"/><mesh file="finger_1.obj"/>',
    ]
    return "\n".join(lines)


def panda_default_block():
    """返回 <default class="panda"> 块（两臂共享 class，只声明一次）。"""
    return """    <default class="panda">
      <material specular="0.5" shininess="0.25"/>
      <joint armature="0.1" damping="1" axis="0 0 1" range="-2.8973 2.8973"/>
      <general dyntype="none" biastype="affine" ctrlrange="-2.8973 2.8973" forcerange="-87 87"/>
      <default class="finger">
        <joint axis="0 1 0" type="slide" range="0 0.04"/>
      </default>
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom type="mesh" group="3"/>
        <default class="fingertip_pad_collision_1">
          <geom type="box" size="0.0085 0.004 0.0085" pos="0 0.0055 0.0445"/>
        </default>
        <default class="fingertip_pad_collision_2">
          <geom type="box" size="0.003 0.002 0.003" pos="0.0055 0.002 0.05"/>
        </default>
        <default class="fingertip_pad_collision_3">
          <geom type="box" size="0.003 0.002 0.003" pos="-0.0055 0.002 0.05"/>
        </default>
        <default class="fingertip_pad_collision_4">
          <geom type="box" size="0.003 0.002 0.0035" pos="0.0055 0.002 0.0395"/>
        </default>
        <default class="fingertip_pad_collision_5">
          <geom type="box" size="0.003 0.002 0.0035" pos="-0.0055 0.002 0.0395"/>
        </default>
      </default>
    </default>"""


def panda_body(suffix="", base_pos=(0.0, 0.0, 0.0), base_yaw_quat="1 0 0 0",
               pedestal=True):
    """生成一个固定基座的 Panda body 链。

    suffix: "" / "_L" / "_R" —— 给 joint/site/finger 命名加后缀。
    base_pos: link0 基座世界坐标。
    base_yaw_quat: 基座朝向四元数 "w x y z"（让臂面向桌子中心）。
    pedestal: 是否在基座下画一个支撑垫块。
    """
    s = suffix
    ped = (f'      <geom type="box" size="0.06 0.06 {base_pos[2]/2 if base_pos[2]>0 else 0.04}" '
           f'pos="0 0 {-base_pos[2]/2 if base_pos[2]>0 else -0.04}" material="pedestal"/>') if pedestal else ""
    return f"""    <body name="link0{s}" childclass="panda" pos="{base_pos[0]} {base_pos[1]} {base_pos[2]}" quat="{base_yaw_quat}">
      <inertial mass="0.629769" pos="-0.041018 -0.00014 0.049974"
        fullinertia="0.00315 0.00388 0.004285 8.2904e-7 0.00015 8.2299e-6"/>
{ped}
      <geom mesh="link0_0" material="off_white" class="visual"/>
      <geom mesh="link0_1" material="black" class="visual"/>
      <geom mesh="link0_2" material="off_white" class="visual"/>
      <geom mesh="link0_3" material="black" class="visual"/>
      <geom mesh="link0_4" material="off_white" class="visual"/>
      <geom mesh="link0_5" material="black" class="visual"/>
      <geom mesh="link0_7" material="white" class="visual"/>
      <geom mesh="link0_8" material="white" class="visual"/>
      <geom mesh="link0_9" material="black" class="visual"/>
      <geom mesh="link0_10" material="off_white" class="visual"/>
      <geom mesh="link0_11" material="white" class="visual"/>
      <geom mesh="link0_c" class="collision"/>
      <body name="link1{s}" pos="0 0 0.333">
        <inertial mass="4.970684" pos="0.003875 0.002081 -0.04762"
          fullinertia="0.70337 0.70661 0.0091170 -0.00013900 0.0067720 0.019169"/>
        <joint name="joint1{s}"/>
        <geom material="white" mesh="link1" class="visual"/>
        <geom mesh="link1_c" class="collision"/>
        <body name="link2{s}" quat="1 -1 0 0">
          <inertial mass="0.646926" pos="-0.003141 -0.02872 0.003495"
            fullinertia="0.0079620 2.8110e-2 2.5995e-2 -3.925e-3 1.0254e-2 7.04e-4"/>
          <joint name="joint2{s}" range="-1.7628 1.7628"/>
          <geom material="white" mesh="link2" class="visual"/>
          <geom mesh="link2_c" class="collision"/>
          <body name="link3{s}" pos="0 -0.316 0" quat="1 1 0 0">
            <joint name="joint3{s}"/>
            <inertial mass="3.228604" pos="2.7518e-2 3.9252e-2 -6.6502e-2"
              fullinertia="3.7242e-2 3.6155e-2 1.083e-2 -4.761e-3 -1.1396e-2 -1.2805e-2"/>
            <geom mesh="link3_0" material="white" class="visual"/>
            <geom mesh="link3_1" material="white" class="visual"/>
            <geom mesh="link3_2" material="white" class="visual"/>
            <geom mesh="link3_3" material="black" class="visual"/>
            <geom mesh="link3_c" class="collision"/>
            <body name="link4{s}" pos="0.0825 0 0" quat="1 1 0 0">
              <inertial mass="3.587895" pos="-5.317e-2 1.04419e-1 2.7454e-2"
                fullinertia="2.5853e-2 1.9552e-2 2.8323e-2 7.796e-3 -1.332e-3 8.641e-3"/>
              <joint name="joint4{s}" range="-3.0718 -0.0698"/>
              <geom mesh="link4_0" material="white" class="visual"/>
              <geom mesh="link4_1" material="white" class="visual"/>
              <geom mesh="link4_2" material="black" class="visual"/>
              <geom mesh="link4_3" material="white" class="visual"/>
              <geom mesh="link4_c" class="collision"/>
              <body name="link5{s}" pos="-0.0825 0.384 0" quat="1 -1 0 0">
                <inertial mass="1.225946" pos="-1.1953e-2 4.1065e-2 -3.8437e-2"
                  fullinertia="3.5549e-2 2.9474e-2 8.627e-3 -2.117e-3 -4.037e-3 2.29e-4"/>
                <joint name="joint5{s}"/>
                <geom mesh="link5_0" material="black" class="visual"/>
                <geom mesh="link5_1" material="white" class="visual"/>
                <geom mesh="link5_2" material="white" class="visual"/>
                <geom mesh="link5_c0" class="collision"/>
                <geom mesh="link5_c1" class="collision"/>
                <geom mesh="link5_c2" class="collision"/>
                <body name="link6{s}" quat="1 1 0 0">
                  <inertial mass="1.666555" pos="6.0149e-2 -1.4117e-2 -1.0517e-2"
                    fullinertia="1.964e-3 4.354e-3 5.433e-3 1.09e-4 -1.158e-3 3.41e-4"/>
                  <joint name="joint6{s}" range="-0.0175 3.7525"/>
                  <geom mesh="link6_0" material="off_white" class="visual"/>
                  <geom mesh="link6_1" material="white" class="visual"/>
                  <geom mesh="link6_2" material="black" class="visual"/>
                  <geom mesh="link6_3" material="white" class="visual"/>
                  <geom mesh="link6_4" material="white" class="visual"/>
                  <geom mesh="link6_5" material="white" class="visual"/>
                  <geom mesh="link6_6" material="white" class="visual"/>
                  <geom mesh="link6_7" material="light_blue" class="visual"/>
                  <geom mesh="link6_8" material="light_blue" class="visual"/>
                  <geom mesh="link6_9" material="black" class="visual"/>
                  <geom mesh="link6_10" material="black" class="visual"/>
                  <geom mesh="link6_11" material="white" class="visual"/>
                  <geom mesh="link6_12" material="green" class="visual"/>
                  <geom mesh="link6_13" material="white" class="visual"/>
                  <geom mesh="link6_14" material="black" class="visual"/>
                  <geom mesh="link6_15" material="black" class="visual"/>
                  <geom mesh="link6_16" material="white" class="visual"/>
                  <geom mesh="link6_c" class="collision"/>
                  <body name="link7{s}" pos="0.088 0 0" quat="1 1 0 0">
                    <inertial mass="7.35522e-01" pos="1.0517e-2 -4.252e-3 6.1597e-2"
                      fullinertia="1.2516e-2 1.0027e-2 4.815e-3 -4.28e-4 -1.196e-3 -7.41e-4"/>
                    <joint name="joint7{s}"/>
                    <geom mesh="link7_0" material="white" class="visual"/>
                    <geom mesh="link7_1" material="black" class="visual"/>
                    <geom mesh="link7_2" material="black" class="visual"/>
                    <geom mesh="link7_3" material="black" class="visual"/>
                    <geom mesh="link7_4" material="black" class="visual"/>
                    <geom mesh="link7_5" material="black" class="visual"/>
                    <geom mesh="link7_6" material="black" class="visual"/>
                    <geom mesh="link7_7" material="white" class="visual"/>
                    <geom mesh="link7_c" class="collision"/>
                    <body name="hand{s}" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">
                      <inertial mass="0.73" pos="-0.01 0 0.03" diaginertia="0.001 0.0025 0.0017"/>
                      <geom mesh="hand_0" material="off_white" class="visual"/>
                      <geom mesh="hand_1" material="black" class="visual"/>
                      <geom mesh="hand_2" material="black" class="visual"/>
                      <geom mesh="hand_3" material="white" class="visual"/>
                      <geom mesh="hand_4" material="off_white" class="visual"/>
                      <geom mesh="hand_c" class="collision"/>
                      <site name="tcp{s}" pos="0 0 0.103" size="0.01" rgba="1 1 0 0.6"/>
                      <body name="left_finger{s}" pos="0 0 0.0584">
                        <inertial mass="0.015" pos="0 0 0" diaginertia="2.375e-6 2.375e-6 7.5e-7"/>
                        <joint name="finger_joint1{s}" class="finger"/>
                        <geom mesh="finger_0" material="off_white" class="visual"/>
                        <geom mesh="finger_1" material="black" class="visual"/>
                        <geom mesh="finger_0" class="collision"/>
                        <geom class="fingertip_pad_collision_1"/>
                        <geom class="fingertip_pad_collision_2"/>
                        <geom class="fingertip_pad_collision_3"/>
                        <geom class="fingertip_pad_collision_4"/>
                        <geom class="fingertip_pad_collision_5"/>
                      </body>
                      <body name="right_finger{s}" pos="0 0 0.0584" quat="0 0 0 1">
                        <inertial mass="0.015" pos="0 0 0" diaginertia="2.375e-6 2.375e-6 7.5e-7"/>
                        <joint name="finger_joint2{s}" class="finger"/>
                        <geom mesh="finger_0" material="off_white" class="visual"/>
                        <geom mesh="finger_1" material="black" class="visual"/>
                        <geom mesh="finger_0" class="collision"/>
                        <geom class="fingertip_pad_collision_1"/>
                        <geom class="fingertip_pad_collision_2"/>
                        <geom class="fingertip_pad_collision_3"/>
                        <geom class="fingertip_pad_collision_4"/>
                        <geom class="fingertip_pad_collision_5"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>"""


def panda_tendon_equality_actuator(suffix=""):
    """返回某一个 Panda 的 tendon split + finger equality + 8 个 actuator。"""
    s = suffix
    tendon = f"""    <fixed name="split{s}">
      <joint joint="finger_joint1{s}" coef="0.5"/>
      <joint joint="finger_joint2{s}" coef="0.5"/>
    </fixed>"""
    equality = (f'    <joint joint1="finger_joint1{s}" joint2="finger_joint2{s}" '
                f'solimp="0.95 0.99 0.001" solref="0.005 1"/>')
    actuator = f"""    <general class="panda" name="actuator1{s}" joint="joint1{s}" gainprm="4500" biasprm="0 -4500 -450"/>
    <general class="panda" name="actuator2{s}" joint="joint2{s}" gainprm="4500" biasprm="0 -4500 -450" ctrlrange="-1.7628 1.7628"/>
    <general class="panda" name="actuator3{s}" joint="joint3{s}" gainprm="3500" biasprm="0 -3500 -350"/>
    <general class="panda" name="actuator4{s}" joint="joint4{s}" gainprm="3500" biasprm="0 -3500 -350" ctrlrange="-3.0718 -0.0698"/>
    <general class="panda" name="actuator5{s}" joint="joint5{s}" gainprm="2000" biasprm="0 -2000 -200" forcerange="-12 12"/>
    <general class="panda" name="actuator6{s}" joint="joint6{s}" gainprm="2000" biasprm="0 -2000 -200" forcerange="-12 12" ctrlrange="-0.0175 3.7525"/>
    <general class="panda" name="actuator7{s}" joint="joint7{s}" gainprm="2000" biasprm="0 -2000 -200" forcerange="-12 12"/>
    <general class="panda" name="actuator8{s}" tendon="split{s}" forcerange="-100 100" ctrlrange="0 255" gainprm="0.01568627451 0 0" biasprm="0 -100 -10"/>"""
    return tendon, equality, actuator

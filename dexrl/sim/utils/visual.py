import numpy as np
from omni.isaac.core.objects import VisualCuboid, VisualCylinder, VisualCone
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats
from scipy.spatial.transform import Rotation as R
import time


def create_coordinate_frame(world, position, scale=0.05, name="coordinate_frame"):
    """
    在指定位置创建坐标轴可视化（细长线条）
    
    Args:
        world: 仿真世界对象
        position: 坐标轴原点的位置 [x, y, z]
        scale: 坐标轴的长度，默认0.15
        name: 坐标轴的名称前缀
        
    Returns:
        dict: 包含三个坐标轴VisualCuboid的字典
    """
    # 添加时间戳确保名称唯一性
    timestamp = int(time.time() * 1000) % 10000  # 取后4位数字
    unique_name = f"{name}_{timestamp}"
    
    # 线条粗细
    thickness = scale * 0.1
    
    # X轴 (红色) - 沿X轴方向的细长立方体
    x_axis = VisualCuboid(
        name=f"{unique_name}_x_axis",
        position=np.array([position[0] + scale/2, position[1], position[2]]),
        prim_path=f"/World/{unique_name}_x_axis",
        size=scale,
        scale=np.array([1.0, thickness/scale, thickness/scale]),  # 细长形状
        color=np.array([1, 0, 0]),  # 红色
    )
    
    # Y轴 (绿色) - 沿Y轴方向的细长立方体
    y_axis = VisualCuboid(
        name=f"{unique_name}_y_axis",
        position=np.array([position[0], position[1] + scale/2, position[2]]),
        prim_path=f"/World/{unique_name}_y_axis",
        size=scale,
        scale=np.array([thickness/scale, 1.0, thickness/scale]),  # 细长形状
        color=np.array([0, 1, 0]),  # 绿色
    )
    
    # Z轴 (蓝色) - 沿Z轴方向的细长立方体
    z_axis = VisualCuboid(
        name=f"{unique_name}_z_axis",
        position=np.array([position[0], position[1], position[2] + scale/2]),
        prim_path=f"/World/{unique_name}_z_axis",
        size=scale,
        scale=np.array([thickness/scale, thickness/scale, 1.0]),  # 细长形状
        color=np.array([0, 0, 1]),  # 蓝色
    )
    
    # 将坐标轴添加到场景中
    world.scene.add(x_axis)
    world.scene.add(y_axis)
    world.scene.add(z_axis)
    
    return {
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis
    }


def create_coordinate_frame_with_orientation(world, position, orientation, scale=0.15, name="coordinate_frame"):
    """
    在指定位置和方向创建坐标轴可视化（细长线条）
    
    Args:
        world: 仿真世界对象
        position: 坐标轴原点的位置 [x, y, z]
        orientation: 四元数方向 [w, x, y, z] 或欧拉角 [rx, ry, rz]
        scale: 坐标轴的长度，默认0.15
        name: 坐标轴的名称前缀
        
    Returns:
        dict: 包含三个坐标轴VisualCuboid的字典
    """
    # 添加时间戳确保名称唯一性
    timestamp = int(time.time() * 1000) % 10000  # 取后4位数字
    unique_name = f"{name}_{timestamp}"
    
    # 处理方向输入
    if len(orientation) == 3:
        # 欧拉角转换为四元数
        quat = euler_angles_to_quats(np.array(orientation))
    else:
        # 已经是四元数
        quat = np.array(orientation)
    
    # 计算旋转矩阵
    rot_matrix = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    
    # 计算各轴的方向向量
    x_direction = rot_matrix[:, 0] * scale
    y_direction = rot_matrix[:, 1] * scale
    z_direction = rot_matrix[:, 2] * scale
    
    # 线条粗细
    thickness = scale * 0.1
    
    # X轴 (红色) - 细长立方体
    x_axis = VisualCuboid(
        name=f"{unique_name}_x_axis",
        position=np.array(position) + x_direction/2,
        orientation=quat,
        prim_path=f"/World/{unique_name}_x_axis",
        size=scale,
        scale=np.array([1.0, thickness/scale, thickness/scale]),  # 细长形状
        color=np.array([1, 0, 0]),  # 红色
    )
    
    # Y轴 (绿色) - 细长立方体
    y_axis = VisualCuboid(
        name=f"{unique_name}_y_axis",
        position=np.array(position) + y_direction/2,
        orientation=quat,
        prim_path=f"/World/{unique_name}_y_axis",
        size=scale,
        scale=np.array([thickness/scale, 1.0, thickness/scale]),  # 细长形状
        color=np.array([0, 1, 0]),  # 绿色
    )
    
    # Z轴 (蓝色) - 细长立方体
    z_axis = VisualCuboid(
        name=f"{unique_name}_z_axis",
        position=np.array(position) + z_direction/2,
        orientation=quat,
        prim_path=f"/World/{unique_name}_z_axis",
        size=scale,
        scale=np.array([thickness/scale, thickness/scale, 1.0]),  # 细长形状
        color=np.array([0, 0, 1]),  # 蓝色
    )
    
    # 将坐标轴添加到场景中
    world.scene.add(x_axis)
    world.scene.add(y_axis)
    world.scene.add(z_axis)
    
    return {
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis
    }


def create_visual_point(world, position, size=0.02, color=np.array([1, 1, 0]), name="visual_point"):
    """
    在指定位置创建可视化点
    
    Args:
        world: 仿真世界对象
        position: 点的位置 [x, y, z]
        size: 点的大小，默认0.02
        color: 点的颜色，默认黄色
        name: 点的名称
        
    Returns:
        VisualCuboid: 可视化点对象
    """
    point = VisualCuboid(
        name=name,
        position=np.array(position),
        prim_path=f"/World/{name}",
        size=size,
        color=color,
    )
    
    world.scene.add(point)
    return point


def create_visual_line(world, start_pos, end_pos, thickness=0.01, color=np.array([1, 1, 1]), name="visual_line"):
    """
    在两点之间创建可视化线段
    
    Args:
        world: 仿真世界对象
        start_pos: 起点位置 [x, y, z]
        end_pos: 终点位置 [x, y, z]
        thickness: 线段粗细，默认0.01
        color: 线段颜色，默认白色
        name: 线段的名称
        
    Returns:
        VisualCuboid: 可视化线段对象
    """
    # 计算线段中点
    mid_point = (np.array(start_pos) + np.array(end_pos)) / 2
    
    # 计算线段长度
    length = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
    
    # 计算线段方向
    direction = (np.array(end_pos) - np.array(start_pos)) / length
    
    # 计算旋转（将Z轴对齐到线段方向）
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, direction)
    rotation_angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
    
    if np.linalg.norm(rotation_axis) > 1e-6:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        quat = R.from_rotvec(rotation_angle * rotation_axis).as_quat()
        # 转换为wxyz格式
        quat = np.array([quat[3], quat[0], quat[1], quat[2]])
    else:
        quat = np.array([1, 0, 0, 0])  # 单位四元数
    
    line = VisualCuboid(
        name=name,
        position=mid_point,
        orientation=quat,
        prim_path=f"/World/{name}",
        size=length,
        scale=np.array([thickness, thickness, 1.0]),
        color=color,
    )
    
    world.scene.add(line)
    return line


def update_coordinate_frame_position(world, coordinate_frame, new_position, new_orientation=None):
    """
    更新坐标轴的位置和方向
    
    Args:
        world: 仿真世界对象
        coordinate_frame: 坐标轴字典，包含x_axis, y_axis, z_axis
        new_position: 新的位置 [x, y, z]
        new_orientation: 新的方向（可选），四元数 [w, x, y, z] 或欧拉角 [rx, ry, rz]
    """
    # 使用固定的scale值，因为VisualCuboid可能没有size属性
    scale = 0.05  # 默认scale值，与创建时使用的值一致
    
    if new_orientation is not None:
        # 处理方向输入
        if len(new_orientation) == 3:
            # 欧拉角转换为四元数
            quat = euler_angles_to_quats(np.array(new_orientation))
        else:
            # 已经是四元数
            quat = np.array(new_orientation)
        
        # 计算旋转矩阵
        rot_matrix = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        
        # 计算各轴的方向向量
        x_direction = rot_matrix[:, 0] * scale
        y_direction = rot_matrix[:, 1] * scale
        z_direction = rot_matrix[:, 2] * scale
        
        # 更新X轴位置和方向
        coordinate_frame["x_axis"].set_world_pose(
            position=np.array(new_position) + x_direction/2,
            orientation=quat
        )
        
        # 更新Y轴位置和方向
        coordinate_frame["y_axis"].set_world_pose(
            position=np.array(new_position) + y_direction/2,
            orientation=quat
        )
        
        # 更新Z轴位置和方向
        coordinate_frame["z_axis"].set_world_pose(
            position=np.array(new_position) + z_direction/2,
            orientation=quat
        )
    else:
        # 只更新位置，保持原有方向
        # 获取当前的方向
        x_quat = coordinate_frame["x_axis"].get_world_pose()[1]
        
        # 计算各轴的方向向量（基于当前方向）
        rot_matrix = R.from_quat([x_quat[1], x_quat[2], x_quat[3], x_quat[0]]).as_matrix()
        x_direction = rot_matrix[:, 0] * scale
        y_direction = rot_matrix[:, 1] * scale
        z_direction = rot_matrix[:, 2] * scale
        
        # 更新X轴位置
        coordinate_frame["x_axis"].set_world_pose(
            position=np.array(new_position) + x_direction/2,
            orientation=x_quat
        )
        
        # 更新Y轴位置
        coordinate_frame["y_axis"].set_world_pose(
            position=np.array(new_position) + y_direction/2,
            orientation=x_quat
        )
        
        # 更新Z轴位置
        coordinate_frame["z_axis"].set_world_pose(
            position=np.array(new_position) + z_direction/2,
            orientation=x_quat
        )


def create_arrow(world, start_pos, end_pos, shaft_radius=0.01, head_radius=0.02, head_length=0.05, color=np.array([1, 0, 0]), name="arrow"):
    """
    创建箭头可视化（使用圆柱体作为箭杆，圆锥体作为箭头）
    
    Args:
        world: 仿真世界对象
        start_pos: 箭头起点位置 [x, y, z]
        end_pos: 箭头终点位置 [x, y, z]
        shaft_radius: 箭杆半径，默认0.01
        head_radius: 箭头半径，默认0.02
        head_length: 箭头长度，默认0.05
        color: 箭头颜色，默认红色
        name: 箭头的名称
        
    Returns:
        dict: 包含箭杆和箭头的字典
    """
    # 添加时间戳确保名称唯一性
    timestamp = int(time.time() * 1000) % 10000
    unique_name = f"{name}_{timestamp}"
    
    # 计算箭头总长度和方向
    direction = np.array(end_pos) - np.array(start_pos)
    total_length = np.linalg.norm(direction)
    direction_normalized = direction / total_length
    
    # 计算旋转（将Z轴对齐到箭头方向）
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, direction_normalized)
    rotation_angle = np.arccos(np.clip(np.dot(z_axis, direction_normalized), -1, 1))
    
    if np.linalg.norm(rotation_axis) > 1e-6:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        quat = R.from_rotvec(rotation_angle * rotation_axis).as_quat()
        # 转换为wxyz格式
        quat = np.array([quat[3], quat[0], quat[1], quat[2]])
    else:
        quat = np.array([1, 0, 0, 0])  # 单位四元数
    
    # 创建箭杆（圆柱体）
    shaft_length = total_length - head_length
    shaft_center = np.array(start_pos) + direction_normalized * (shaft_length / 2)
    
    shaft = VisualCylinder(
        name=f"{unique_name}_shaft",
        position=shaft_center,
        orientation=quat,
        prim_path=f"/World/{unique_name}_shaft",
        radius=shaft_radius,
        height=shaft_length,
        color=color,
    )
    
    # 创建箭头（圆锥体）
    head_center = np.array(start_pos) + direction_normalized * (shaft_length + head_length / 2)
    
    head = VisualCone(
        name=f"{unique_name}_head",
        position=head_center,
        orientation=quat,
        prim_path=f"/World/{unique_name}_head",
        radius=head_radius,
        height=head_length,
        color=color,
    )
    
    # 将箭头组件添加到场景中
    world.scene.add(shaft)
    world.scene.add(head)
    
    return {
        "shaft": shaft,
        "head": head
    }


def create_coordinate_frame_arrows(world, position, scale=0.15, name="coordinate_frame"):
    """
    在指定位置创建坐标轴可视化（使用箭头）
    
    Args:
        world: 仿真世界对象
        position: 坐标轴原点的位置 [x, y, z]
        scale: 坐标轴的长度，默认0.15
        name: 坐标轴的名称前缀
        
    Returns:
        dict: 包含三个坐标轴箭头的字典
    """
    # 添加时间戳确保名称唯一性
    timestamp = int(time.time() * 1000) % 10000
    unique_name = f"{name}_{timestamp}"
    
    # 箭头参数
    shaft_radius = scale * 0.02
    head_radius = scale * 0.04
    head_length = scale * 0.1
    
    # X轴 (红色)
    x_arrow = create_arrow(
        world,
        start_pos=position,
        end_pos=np.array([position[0] + scale, position[1], position[2]]),
        shaft_radius=shaft_radius,
        head_radius=head_radius,
        head_length=head_length,
        color=np.array([1, 0, 0]),  # 红色
        name=f"{unique_name}_x_axis"
    )
    
    # Y轴 (绿色)
    y_arrow = create_arrow(
        world,
        start_pos=position,
        end_pos=np.array([position[0], position[1] + scale, position[2]]),
        shaft_radius=shaft_radius,
        head_radius=head_radius,
        head_length=head_length,
        color=np.array([0, 1, 0]),  # 绿色
        name=f"{unique_name}_y_axis"
    )
    
    # Z轴 (蓝色)
    z_arrow = create_arrow(
        world,
        start_pos=position,
        end_pos=np.array([position[0], position[1], position[2] + scale]),
        shaft_radius=shaft_radius,
        head_radius=head_radius,
        head_length=head_length,
        color=np.array([0, 0, 1]),  # 蓝色
        name=f"{unique_name}_z_axis"
    )
    
    return {
        "x_axis": x_arrow,
        "y_axis": y_arrow,
        "z_axis": z_arrow
    }


def create_coordinate_frame_with_orientation_arrows(world, position, orientation, scale=0.15, name="coordinate_frame"):
    """
    在指定位置和方向创建坐标轴可视化（使用箭头）
    
    Args:
        world: 仿真世界对象
        position: 坐标轴原点的位置 [x, y, z]
        orientation: 四元数方向 [w, x, y, z] 或欧拉角 [rx, ry, rz]
        scale: 坐标轴的长度，默认0.15
        name: 坐标轴的名称前缀
        
    Returns:
        dict: 包含三个坐标轴箭头的字典
    """
    # 处理方向输入
    if len(orientation) == 3:
        # 欧拉角转换为四元数
        quat = euler_angles_to_quats(np.array(orientation))
    else:
        # 已经是四元数
        quat = np.array(orientation)
    
    # 计算旋转矩阵
    rot_matrix = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    
    # 计算各轴的方向向量
    x_direction = rot_matrix[:, 0] * scale
    y_direction = rot_matrix[:, 1] * scale
    z_direction = rot_matrix[:, 2] * scale
    
    # 箭头参数
    shaft_radius = scale * 0.02
    head_radius = scale * 0.04
    head_length = scale * 0.1
    
    # 添加时间戳确保名称唯一性
    timestamp = int(time.time() * 1000) % 10000
    unique_name = f"{name}_{timestamp}"
    
    # X轴 (红色)
    x_arrow = create_arrow(
        world,
        start_pos=position,
        end_pos=np.array(position) + x_direction,
        shaft_radius=shaft_radius,
        head_radius=head_radius,
        head_length=head_length,
        color=np.array([1, 0, 0]),  # 红色
        name=f"{unique_name}_x_axis"
    )
    
    # Y轴 (绿色)
    y_arrow = create_arrow(
        world,
        start_pos=position,
        end_pos=np.array(position) + y_direction,
        shaft_radius=shaft_radius,
        head_radius=head_radius,
        head_length=head_length,
        color=np.array([0, 1, 0]),  # 绿色
        name=f"{unique_name}_y_axis"
    )
    
    # Z轴 (蓝色)
    z_arrow = create_arrow(
        world,
        start_pos=position,
        end_pos=np.array(position) + z_direction,
        shaft_radius=shaft_radius,
        head_radius=head_radius,
        head_length=head_length,
        color=np.array([0, 0, 1]),  # 蓝色
        name=f"{unique_name}_z_axis"
    )
    
    return {
        "x_axis": x_arrow,
        "y_axis": y_arrow,
        "z_axis": z_arrow
    }


# 使用示例：
"""
# 导入visual模块
from dexrl.sim.utils import visual

# 1. 创建标准坐标轴（使用箭头）
world_coord_frame = visual.create_coordinate_frame_arrows(
    world,  # 仿真世界对象
    position=np.array([0, 0, 0]),  # 坐标轴原点
    scale=0.15,  # 坐标轴长度
    name="world_frame"  # 名称前缀
)

# 2. 创建带方向的坐标轴（使用箭头）
hand_coord_frame = visual.create_coordinate_frame_with_orientation_arrows(
    world,
    position=np.array([0.1, 0.2, 0.3]),  # 位置
    orientation=np.array([1, 0, 0, 0]),  # 四元数方向
    scale=0.1,
    name="hand_frame"
)

# 3. 创建单个箭头
arrow = visual.create_arrow(
    world,
    start_pos=np.array([0, 0, 0]),
    end_pos=np.array([0.1, 0.1, 0.1]),
    shaft_radius=0.01,
    head_radius=0.02,
    head_length=0.05,
    color=np.array([1, 0, 0]),  # 红色
    name="direction_arrow"
)

# 4. 创建可视化点
point = visual.create_visual_point(
    world,
    position=np.array([0.1, 0.2, 0.3]),
    size=0.02,
    color=np.array([1, 1, 0]),  # 黄色
    name="target_point"
)

# 5. 创建可视化线段
line = visual.create_visual_line(
    world,
    start_pos=np.array([0, 0, 0]),
    end_pos=np.array([0.1, 0.1, 0.1]),
    thickness=0.01,
    color=np.array([1, 1, 1]),  # 白色
    name="connection_line"
)

# 注意：新的箭头函数比原来的细长立方体更直观！
# 原来的函数仍然可用，但建议使用新的箭头函数
""" 
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
def quat_to_6d(quat):
    """
    将四元数转换为6D旋转表示
    
    Args:
        quat: 四元数 [w, x, y, z] 或 [x, y, z, w]
    
    Returns:
        6D旋转表示 [a1, a2, a3, b1, b2, b3]
    """
    quat = np.array(quat, dtype=np.float64)
    
    # 确保四元数格式正确 [x, y, z, w] (scipy格式)
    if len(quat) == 4:
        # 如果输入是 [w, x, y, z] 格式，转换为 [x, y, z, w]
        # 检查哪个分量更大来判断格式
        if abs(quat[0]) > abs(quat[3]):  # 如果第一个分量更大，说明是 [w,x,y,z] 格式
            quat = np.array([quat[1], quat[2], quat[3], quat[0]])  # 转换为 [x,y,z,w]
    
    # 转换为旋转矩阵 (scipy期望[x,y,z,w]格式)
    rot_matrix = R.from_quat(quat).as_matrix()
    
    # 获取前两列
    a = rot_matrix[:, 0]  # 第一列
    b = rot_matrix[:, 1]  # 第二列
    
    # 确保正交性（Gram-Schmidt正交化）
    a = a / np.linalg.norm(a)  # 归一化第一列
    b = b - np.dot(b, a) * a   # 使第二列与第一列正交
    b = b / np.linalg.norm(b)  # 归一化第二列
    
    # 返回6D表示
    return np.concatenate([a, b])

def rot_6d_to_quat(rot_6d):
    """
    将6D旋转表示转换为四元数
    
    Args:
        rot_6d: 6D旋转表示 [a1, a2, a3, b1, b2, b3]
    
    Returns:
        四元数 [w, x, y, z]
    """
    if len(rot_6d) != 6:
        raise ValueError(f"6D旋转表示必须是6维向量，得到{len(rot_6d)}维")
    
    rot_6d = np.array(rot_6d, dtype=np.float64)
    
    # 提取前两列
    a = rot_6d[:3]  # 第一列
    b = rot_6d[3:]  # 第二列
    
    # 检查输入的有效性
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    
    if a_norm < 1e-10:
        raise ValueError("6D表示的第一列范数太小")
    if b_norm < 1e-10:
        raise ValueError("6D表示的第二列范数太小")
    
    # 归一化前两列
    a = a / a_norm
    b = b / b_norm
    
    # 使用Gram-Schmidt正交化确保正交性
    b = b - np.dot(b, a) * a
    b_norm_after_ortho = np.linalg.norm(b)
    
    if b_norm_after_ortho < 1e-10:
        raise ValueError("正交化后第二列范数太小，无法构建有效的旋转矩阵")
    
    b = b / b_norm_after_ortho
    
    # 计算第三列（叉积）
    c = np.cross(a, b)
    
    # 构建旋转矩阵
    rot_matrix = np.column_stack([a, b, c])
    
    # 使用scipy的Rotation类从旋转矩阵创建四元数
    try:
        rotation = R.from_matrix(rot_matrix)
        quat = rotation.as_quat()  # 返回 [x, y, z, w] 格式
        
        # 转换为 [w, x, y, z] 格式
        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
        
        return quat_wxyz
        
    except Exception as e:
        raise ValueError(f"无法从旋转矩阵创建四元数: {e}")

def rot_6d_to_rot_matrix(rot_6d):
    """
    将6D旋转表示转换为3x3旋转矩阵
    
    Args:
        rot_6d: 6D旋转表示 [a1, a2, a3, b1, b2, b3]
    
    Returns:
        3x3旋转矩阵
    """
    if len(rot_6d) != 6:
        raise ValueError(f"6D旋转表示必须是6维向量，得到{len(rot_6d)}维")
    
    rot_6d = np.array(rot_6d, dtype=np.float64)
    
    # 提取前两列
    a = rot_6d[:3]  # 第一列
    b = rot_6d[3:]  # 第二列
    
    # 归一化前两列
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    
    # 使用Gram-Schmidt正交化确保正交性
    b = b - np.dot(b, a) * a
    b = b / np.linalg.norm(b)
    
    # 计算第三列（叉积）
    c = np.cross(a, b)
    
    # 构建旋转矩阵
    rot_matrix = np.column_stack([a, b, c])
    
    return rot_matrix
    
def rot_matrix_to_6d(rot_matrix):
    """
    将3x3旋转矩阵转换为6D旋转表示
    
    Args:
        rot_matrix: 3x3旋转矩阵
    
    Returns:
        6D旋转表示 [a1, a2, a3, b1, b2, b3]
    """
    # return quat_to_6d(rot_matrix_to_quat(rot_matrix))
    # 直接提取旋转矩阵的前两列作为6D表示
    a = rot_matrix[:, 0]  # 第一列
    b = rot_matrix[:, 1]  # 第二列
    
    # 返回6D表示
    return np.concatenate([a, b])

def rot_matrix_to_lookat_direction(rot_matrix, forward_direction=np.array([0, 0, -1])):
    forward_direction = forward_direction
    forward_direction_world =  rot_matrix @ forward_direction
    return forward_direction_world

def rot_matrix_to_lookat(position, rot_matrix, forward_direction=np.array([0, 0, -1]), distance=1.0):
    """
    计算相对相机正对方向指定距离的坐标

    参数:
        camera_position (numpy.array): 相机的坐标 (x, y, z)
        rotation_matrix (numpy.array): 相机的旋转矩阵 (3x3)
        distance (float): 目标点距离相机的距离，默认为1.0米
        forward_direction (numpy.array): 相机正对方向的单位向量，默认为(0,0,-1)

    返回:
        numpy.array: 目标点的坐标 (x, y, z)
    """
    offset = rot_matrix_to_lookat_direction(rot_matrix, forward_direction) * distance
    target_position = position + offset
    return target_position

def quat_to_rot_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert input quaternion to rotation matrix.

    Args:
        quat (np.ndarray): Input quaternion (w, x, y, z).

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    q = np.array(quat, dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < 1e-10:
        return np.identity(3)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array(
        (
            (1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]),
            (q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]),
            (q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]),
        ),
        dtype=np.float64,
    )

def quat_multiply(q1, q2):
    """四元数乘法"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def quat_from_axis_angle(axis, angle):
    """从轴和角度创建四元数"""
    half_angle = np.radians(angle) / 2.0
    sin_half_angle = np.sin(half_angle)
    cos_half_angle = np.cos(half_angle)
    return np.array([
        cos_half_angle,
        axis[0] * sin_half_angle,
        axis[1] * sin_half_angle,
        axis[2] * sin_half_angle
    ])

def rotate_quat(q, axis, angle):
    """绕指定轴旋转四元数"""
    # 创建旋转四元数
    rotation_quaternion = quat_from_axis_angle(axis, angle)
    # 旋转四元数的共轭
    rotation_conjugate = np.array([
        rotation_quaternion[0],
        -rotation_quaternion[1],
        -rotation_quaternion[2],
        -rotation_quaternion[3]
    ])
    # 应用旋转：q' = R * q * R*
    return quat_multiply(quat_multiply(rotation_quaternion, q), rotation_conjugate)


# internal global constants
_POLE_LIMIT = 1.0 - 1e-6


def rot_matrix_to_quat(mat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to Quaternion.

    Args:
        mat (np.ndarray): A 3x3 rotation matrix.

    Returns:
        np.ndarray: quaternion (w, x, y, z).
    """
    if mat.shape == (3, 3):
        tmp = np.eye(4)
        tmp[0:3, 0:3] = mat
        mat = tmp

    q = np.empty((4,), dtype=np.float64)
    t = np.trace(mat)
    if t > mat[3, 3]:
        q[0] = t
        q[3] = mat[1, 0] - mat[0, 1]
        q[2] = mat[0, 2] - mat[2, 0]
        q[1] = mat[2, 1] - mat[1, 2]
    else:
        i, j, k = 0, 1, 2
        if mat[1, 1] > mat[0, 0]:
            i, j, k = 1, 2, 0
        if mat[2, 2] > mat[i, i]:
            i, j, k = 2, 0, 1
        t = mat[i, i] - (mat[j, j] + mat[k, k]) + mat[3, 3]
        q[i + 1] = t
        q[j + 1] = mat[i, j] + mat[j, i]
        q[k + 1] = mat[k, i] + mat[i, k]
        q[0] = mat[k, j] - mat[j, k]
    q *= 0.5 / np.sqrt(t * mat[3, 3])
    return q


def quat_to_rot_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert input quaternion to rotation matrix.

    Args:
        quat (np.ndarray): Input quaternion (w, x, y, z).

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    q = np.array(quat, dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < 1e-10:
        return np.identity(3)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array(
        (
            (1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]),
            (q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]),
            (q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]),
        ),
        dtype=np.float64,
    )


def matrix_to_euler_angles(mat: np.ndarray, degrees: bool = False, extrinsic: bool = True) -> np.ndarray:
    """Convert rotation matrix to Euler XYZ extrinsic or intrinsic angles.

    Args:
        mat (np.ndarray): A 3x3 rotation matrix.
        degrees (bool, optional): Whether returned angles should be in degrees.
        extrinsic (bool, optional): True if the rotation matrix follows the extrinsic matrix
                   convention (equivalent to ZYX ordering but returned in the reverse) and False if it follows
                   the intrinsic matrix conventions (equivalent to XYZ ordering).
                   Defaults to True.

    Returns:
        np.ndarray: Euler XYZ angles (intrinsic form) if extrinsic is False and Euler XYZ angles (extrinsic form) if extrinsic is True.
    """
    if extrinsic:
        if mat[2, 0] > _POLE_LIMIT:
            roll = np.arctan2(mat[0, 1], mat[0, 2])
            pitch = -np.pi / 2
            yaw = 0.0
            return np.array([roll, pitch, yaw])

        if mat[2, 0] < -_POLE_LIMIT:
            roll = np.arctan2(mat[0, 1], mat[0, 2])
            pitch = np.pi / 2
            yaw = 0.0
            return np.array([roll, pitch, yaw])

        roll = np.arctan2(mat[2, 1], mat[2, 2])
        pitch = -np.arcsin(mat[2, 0])
        yaw = np.arctan2(mat[1, 0], mat[0, 0])
        if degrees:
            roll = math.degrees(roll)
            pitch = math.degrees(pitch)
            yaw = math.degrees(yaw)
        return np.array([roll, pitch, yaw])
    else:
        if mat[0, 2] > _POLE_LIMIT:
            roll = np.arctan2(mat[1, 0], mat[1, 1])
            pitch = np.pi / 2
            yaw = 0.0
            return np.array([roll, pitch, yaw])

        if mat[0, 2] < -_POLE_LIMIT:
            roll = np.arctan2(mat[1, 0], mat[1, 1])
            pitch = -np.pi / 2
            yaw = 0.0
            return np.array([roll, pitch, yaw])
        roll = -math.atan2(mat[1, 2], mat[2, 2])
        pitch = math.asin(mat[0, 2])
        yaw = -math.atan2(mat[0, 1], mat[0, 0])

        if degrees:
            roll = math.degrees(roll)
            pitch = math.degrees(pitch)
            yaw = math.degrees(yaw)
        return np.array([roll, pitch, yaw])


def euler_to_rot_matrix(euler_angles: np.ndarray, degrees: bool = False, extrinsic: bool = True) -> np.ndarray:
    """Convert Euler XYZ or ZYX angles to rotation matrix.

    Args:
        euler_angles (np.ndarray): Euler angles.
        degrees (bool, optional): Whether passed angles are in degrees.
        extrinsic (bool, optional): True if the euler angles follows the extrinsic angles
                   convention (equivalent to ZYX ordering but returned in the reverse) and False if it follows
                   the intrinsic angles conventions (equivalent to XYZ ordering).
                   Defaults to True.

    Returns:
        np.ndarray:  A 3x3 rotation matrix in its extrinsic or intrinsic form depends on the extrinsic argument.
    """
    if extrinsic:
        yaw, pitch, roll = euler_angles
    else:
        roll, pitch, yaw = euler_angles
    if degrees:
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
    cr = np.cos(roll)
    sr = np.sin(roll)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    if extrinsic:
        return np.array(
            [
                [(cp * cr), ((cr * sp * sy) - (cy * sr)), ((cr * cy * sp) + (sr * sy))],
                [(cp * sr), ((cy * cr) + (sr * sp * sy)), ((cy * sp * sr) - (cr * sy))],
                [-sp, (cp * sy), (cy * cp)],
            ]
        )
    else:
        return np.array(
            [
                [(cp * cy), (-cp * sy), sp],
                [((cy * sr * sp) + (cr * sy)), ((cr * cy) - (sr * sp * sy)), (-cp * sr)],
                [((-cr * cy * sp) + (sr * sy)), ((cy * sr) + (cr * sp * sy)), (cr * cp)],
            ]
        )


def quat_to_euler_angles(quat: np.ndarray, degrees: bool = False, extrinsic: bool = True) -> np.ndarray:
    """Convert input quaternion to Euler XYZ or ZYX angles.

    Args:
        quat (np.ndarray): Input quaternion (w, x, y, z).
        degrees (bool, optional): Whether returned angles should be in degrees. Defaults to False.
        extrinsic (bool, optional): True if the euler angles follows the extrinsic angles
                   convention (equivalent to ZYX ordering but returned in the reverse) and False if it follows
                   the intrinsic angles conventions (equivalent to XYZ ordering).
                   Defaults to True.


    Returns:
        np.ndarray: Euler XYZ angles (intrinsic form) if extrinsic is False and Euler XYZ angles (extrinsic form) if extrinsic is True.
    """
    return matrix_to_euler_angles(quat_to_rot_matrix(quat), degrees=degrees, extrinsic=extrinsic)


def euler_angles_to_quat(euler_angles: np.ndarray, degrees: bool = False, extrinsic: bool = True) -> np.ndarray:
    """Convert Euler angles to quaternion.

    Args:
        euler_angles (np.ndarray):  Euler XYZ angles.
        degrees (bool, optional): Whether input angles are in degrees. Defaults to False.
        extrinsic (bool, optional): True if the euler angles follows the extrinsic angles
                   convention (equivalent to ZYX ordering but returned in the reverse) and False if it follows
                   the intrinsic angles conventions (equivalent to XYZ ordering).
                   Defaults to True.

    Returns:
        np.ndarray: quaternion (w, x, y, z).
    """
    mat = np.array(euler_to_rot_matrix(euler_angles, degrees=degrees, extrinsic=extrinsic))
    return rot_matrix_to_quat(mat)

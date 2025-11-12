from omni.isaac.core.utils.rotations import *

def euler_to_quat(euler):
    return euler_angles_to_quat(euler)


def rot_matrix_to_lookat_direction(rot_matrix, forward_direction=np.array([0, 0, -1])):
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


def axis_theta_to_rot_matrix(axis, theta):
    """
    计算绕任意轴旋转的旋转矩阵。
    
    参数:
        axis: 旋转轴的向量（非单位向量也可以，函数内部会标准化）
        theta: 旋转角度（弧度制）
    
    返回:
        旋转矩阵
    """
    # 将旋转轴标准化为单位向量
    axis = np.array(axis) / np.linalg.norm(axis)
    kx, ky, kz = axis
    
    # 计算反对称矩阵 K
    K = np.array([
        [0, -kz, ky],
        [kz, 0, -kx],
        [-ky, kx, 0]
    ])
    
    # 计算旋转矩阵 R
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    
    return R

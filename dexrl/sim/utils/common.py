import os
import numpy as np
import platform
import torch as th
import re
import subprocess
# from omni.isaac.nucleus import get_assets_root_path
import configparser
from dataclasses import dataclass
from pxr import Sdf
import omni.isaac.core.utils.prims as prims_utils
from cryptography.fernet import Fernet
import torch
import pxr
# ROOT_PATH = pathlib.Path(__file__).parents[3]
# print(ROOT_PATH)
# global_config = configparser.ConfigParser()
# global_config.read(ROOT_PATH / 'global_config.ini')
from omni.isaac.core.utils.rotations import euler_angles_to_quat

from dexrl import global_config

omnigibson_asset_path = os.path.join(global_config.data_path, "Assets/OmniGibson")
omnigibson_key_path = os.path.join(global_config.data_path, "Assets/OmniGibson/omnigibson.key")


# class Normalizer_N1_1:
#     def __init__(self, min_value: torch.Tensor, max_value: torch.Tensor):
#         self.min_value = min_value
#         self.max_value = max_value

#         self.scale = self.max_value - self.min_value
#     #     self.bias = (max_value + min_value) / 2

#     def normalize(self, denormalized_value):
#         return 2 * (denormalized_value - self.min_value) / (self.scale) - 1
    
#     def denormalize(self, normalized_value):
#         return (normalized_value + 1) * (self.scale) / 2 + self.min_value


# class Normalizer_0_1(Normalizer_N1_1):
#     def normalize(self, denormalized_value):
#         return (denormalized_value - self.min_value) / (self.scale)
    
#     def denormalize(self, normalized_value):
#         return normalized_value * self.scale + self.min_value


# class Normalizer_N1_1(Normalizer_N1_1):
#     def normalize(self, denormalized_value):
#         return 2 * (denormalized_value - self.min_value) / (self.scale) - 1
    
#     def denormalize(self, normalized_value):
#         return (normalized_value + 1) * (self.scale) / 2 + self.min_value


def rotation_matrix_from_vectors(vec1, vec2):
    """
    找到将 vec1 旋转到 vec2 的旋转矩阵。
    
    参数:
    vec1 -- 初始方向向量
    vec2 -- 目标方向向量
    
    返回:
    R -- 旋转矩阵
    """
    # 将输入向量转换为 NumPy 数组
    vec1 = np.array(vec1, dtype=float)
    vec2 = np.array(vec2, dtype=float)
    
    # 归一化向量
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    
    # 计算旋转轴（叉积）
    axis = np.cross(vec1, vec2)
    axis_norm = np.linalg.norm(axis)
    
    # 如果两个向量完全相同或完全相反，直接返回单位矩阵或反射矩阵
    if axis_norm < 1e-6:  # 阈值用于判断是否共线
        if np.allclose(vec1, vec2):
            return np.eye(3)  # 同方向，返回单位矩阵
        else:
            # 反方向，构造一个反射矩阵
            normal = np.cross(vec1, [1, 0, 0])  # 假设 vec1 不是 [0, 0, 1]
            if np.linalg.norm(normal) < 1e-6:
                normal = np.cross(vec1, [0, 1, 0])  # vec1 是 [0, 0, 1]，使用另一个方向
            normal = normal / np.linalg.norm(normal)
            return np.eye(3) - 2 * np.outer(normal, normal)
    
    # 归一化旋转轴
    axis = axis / axis_norm
    
    # 计算旋转角度（点积）
    angle = np.arccos(np.dot(vec1, vec2))
    
    # 构造旋转矩阵（罗德里格斯公式）
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    
    return R


def euler_yz_angles_to_quats(euler_yz_angles, rotation=0):
    return euler_angles_to_quat((rotation,euler_yz_angles[0], euler_yz_angles[1]))

def format_array(array):
    result = ""
    if isinstance(array, float):
        return f"{array:.2f}"
    for i in range(len(array)):
        result += f"{array[i]:.2f},"
    return result[:-1]

def disable_physics(prim_path):
    change_prim_property(prim_path, "physics:collisionEnabled", False)
    change_prim_property(prim_path, "physxRigidBody:disableGravity", True)

def enable_physics(prim_path):
    change_prim_property(prim_path, "physics:collisionEnabled", True)
    change_prim_property(prim_path, "physxRigidBody:disableGravity", False)


def omnigibson_object_fix_base(prim_path):
    prim = prims_utils.get_prim_at_path(prim_path)
    # prim.RemoveAPI(pxr.UsdPhysics.ArticulationRootAPI)
    prim.RemoveAPI(pxr.UsdPhysics.RigidBodyAPI)
    # change_prim_property(prim_path+"/base_link", "physxArticulation:articulationEnabled", False)
    # change_prim_property(prim_path+"/base_link", "physics:rigidBodyEnabled", False)


def calculate_lookat_position(camera_position, rotation_matrix, distance=1.0):
    """
    计算相对相机正对方向指定距离的坐标

    参数:
        camera_position (numpy.array): 相机的坐标 (x, y, z)
        rotation_matrix (numpy.array): 相机的旋转矩阵 (3x3)
        distance (float): 目标点距离相机的距离，默认为1.0米

    返回:
        numpy.array: 目标点的坐标 (x, y, z)
    """
    # 相机正对方向的单位向量
    forward_direction = np.array([0, 0, -1])  # 相机坐标系中的Z轴方向
    # 将相机坐标系中的单位向量转换到世界坐标系中
    forward_direction_world =  rotation_matrix @ forward_direction
    # 计算目标点的坐标
    target_position = camera_position + distance * forward_direction_world
    return target_position


def change_prim_property(prim_path,property_name,value):
    """If it exists, set the ``physics:rigidBodyEnabled`` attribute on the USD Prim at the given path

    .. note::

        If the prim does not have the physics Rigid Body property added, calling this function will have no effect

    Args:
        _value (Any): Value to set ``physics:rigidBodyEnabled`` attribute to
        prim_path (str): The path to the USD Prim

    Example:

    .. code-block:: python

        >>> import omni.isaac.core.utils.physics as physics_utils
        >>>
        >>> physics_utils.set_rigid_body_enabled(False, "/World/Cube")
    """
    import omni.kit.commands

    omni.kit.commands.execute(
        "ChangeProperty", prop_path=Sdf.Path(f"{prim_path}.{property_name}"), value=value, prev=None
    )



def get_pid_by_port(port):
    """
    获取指定端口的占用进程 ID (PID)。
    :param port: 要检测的端口号
    :return: 占用该端口的进程 ID (PID)，如果没有占用则返回 None
    """
    system = platform.system()
    pid = None

    if system == "Linux":
        # 在 Linux 系统中使用 lsof 命令
        try:
            result = subprocess.run(
                ["lsof", "-i", f":{port}", "-t"],  # -t 选项直接输出 PID
                capture_output=True,
                text=True,
                check=True
            )
            pid = result.stdout.strip()
        except subprocess.CalledProcessError:
            # 如果没有找到占用端口的进程，lsof 会返回非零退出码
            pass

    elif system == "Windows":
        # 在 Windows 系统中使用 netstat 和 findstr 命令
        try:
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                check=True
            )
            output = result.stdout
            # 使用正则表达式匹配指定端口的行
            pattern = re.compile(rf"\s*TCP\s+\S+:{port}\s+\S+\s+\S+\s+(\d+)\s*")
            match = pattern.search(output)
            if match:
                pid = match.group(1)
        except subprocess.CalledProcessError:
            # 如果命令执行失败，捕获异常
            pass

    return pid



def kill_process_using_port(port):
    """如果端口被占用，结束占用端口的进程"""
    pid = get_pid_by_port(port)
    if pid:
        print(f"端口 {port} 被进程 {pid} 占用, 正在杀死进程。")
        kill_process(pid)
    else:
        print(f"端口 {port} 未被占用。")

def kill_process(pid):
    pid=int(pid)
    if platform.system() == "Windows":
        os.system(f"taskkill /PID {pid} /F")
    else:
        os.system(f"kill -9 {pid}")
    
    print(f"进程 {pid} 已杀死。")



def download_file_from_url(url, save_path):
    """
    从URL下载文件并保存到指定路径
    :param url: 文件的URL地址
    :param save_path: 保存文件的路径
    :return: 是否下载成功
    """
    try:
        import requests
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查请求是否成功
        
        # 确保保存路径的目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 以二进制写入模式保存文件
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"下载文件时发生错误: {e}")
        return False

def download_isaacsim_asset(path_to_robot_usd):
    """下载IsaacSim的资产"""
    asset_path = os.path.join(os.path.dirname(__file__), 'assets')
    path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/Franka/franka.usd"


def get_omnibt_asset(omnibt_asset_path):
    """获取omnibt的资产路径"""
    local_path = os.path.join(cfg.omnibt_asset_path, omnibt_asset_path)
    return local_path

def get_isaacsim_asset(isaacsim_asset_path):
    """获取IsaacSim的资产路径"""
    local_path = os.path.join(cfg.isaacsim_asset_path, isaacsim_asset_path)

    if os.path.exists(local_path):
        return local_path
    else:
        nucleus_path = os.path.join(get_assets_root_path(), isaacsim_asset_path)
        return nucleus_path

def decrypt_file(encrypted_filename, decrypted_filename):
    with open(cfg.omnigibson_key_path, "rb") as filekey:
        key = filekey.read()
    fernet = Fernet(key)

    with open(encrypted_filename, "rb") as enc_f:
        encrypted = enc_f.read()

    decrypted = fernet.decrypt(encrypted)

    with open(decrypted_filename, "wb") as decrypted_file:
        decrypted_file.write(decrypted)


def get_object_path(category,object_name):
    return os.path.join(global_config.assets_path, f"objects/{category}/{object_name}/{object_name}.usd")

def get_omnigibson_asset(category="bottom_cabinet", object_name="jhymlr"):
    folder_path = f"og_dataset/objects/{category}/{object_name}/usd/"
    target_path = os.path.join(global_config.omnigibson_asset_path, folder_path, f"{object_name}.usd")
    encrypted_path = os.path.join(global_config.omnigibson_asset_path, folder_path, f"{object_name}.encrypted.usd")
    if not os.path.exists(target_path):
        assert os.path.exists(encrypted_path), f"OmniGibson 物体 {encrypted_path} 不存在"
        decrypt_file(encrypted_path, target_path)
    # else:
    #     # 删除target_path
    #     os.remove(target_path)
    #     decrypt_file(encrypted_path, target_path)
    return target_path

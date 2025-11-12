from omnigibson.sensors.vision_sensor import VisionSensor
import omnigibson.utils.transform_utils as T
import numpy as np
import torch

class OGCamera:
    """
    Defines the camera class
    """
    def __init__(self, og_env, config) -> None:        
        self.cam = insert_camera(name=config['name'], og_env=og_env, width=config['resolution'], height=config['resolution'])
        self.cam.set_position_orientation(config['position'], config['orientation'])
        self.intrinsics = get_cam_intrinsics(self.cam)
        self.extrinsics = get_cam_extrinsics(self.cam)

    def get_params(self):
        """
        Get the intrinsic and extrinsic parameters of the camera
        """
        return {"intrinsics": self.intrinsics, "extrinsics": self.extrinsics}
    
    def get_obs(self):
        """
        Gets the image observation from the camera.
        Assumes have rendered befor calling this function.
        No semantic handling here for now.
        """
        obs = self.cam.get_obs()
        ret = {}
        ret["rgb"] = obs[0]["rgb"][:,:,:3]  # H, W, 3
        ret["depth"] = obs[0]["depth_linear"]  # H, W
        ret["points"] = pixel_to_3d_points(ret["depth"], self.intrinsics, self.extrinsics)  # H, W, 3
        ret["seg"] = obs[0]["seg_semantic"]  # H, W
        ret["seg_instance"] = obs[0]["seg_instance"]  # H, W
        ret["seg_instance_id"] = obs[0]["seg_instance_id"]  # H, W
        ret["intrinsic"] = self.intrinsics
        ret["extrinsic"] = self.extrinsics
        
        # 添加边界框数据
        if "bbox_2d_tight" in obs[0]:
            ret["bbox_2d_tight"] = obs[0]["bbox_2d_tight"]
        if "bbox_2d_loose" in obs[0]:
            ret["bbox_2d_loose"] = obs[0]["bbox_2d_loose"]
        if "bbox_3d" in obs[0]:
            ret["bbox_3d"] = obs[0]["bbox_3d"]
        return ret

def insert_camera(name, og_env, width=480, height=480):
    try:
        cam = VisionSensor(
            prim_path=f"/World/{name}",
            name=name,
            image_width=width,
            image_height=height,
            modalities=["rgb", "depth_linear", "seg_semantic","seg_instance","seg_instance_id", "bbox_2d_tight", "bbox_2d_loose", "bbox_3d"]
            #modalities=["rgb", "depth_linear", "seg_semantic","seg_instance","seg_instance_id"]
        )
    except TypeError:
        cam = VisionSensor(
            relative_prim_path=f"/{name}",
            name=name,
            image_width=width,
            image_height=height,
            modalities=["rgb", "depth_linear", "seg_semantic","seg_instance","seg_instance_id", "bbox_2d_tight", "bbox_2d_loose", "bbox_3d"]
            #modalities=["rgb", "depth_linear", "seg_semantic","seg_instance","seg_instance_id"]
        )
    
    try:
        cam.load()
    except TypeError:
        cam.load(og_env.scene)
    cam.initialize()
    return cam

def get_cam_intrinsics(cam):
    """
    Get the intrinsics matrix for a VisionSensor object
    ::param cam: VisionSensor object
    ::return intrinsics: 3x3 numpy array
    """
    img_width = cam.image_width
    img_height = cam.image_height

    if img_width != img_height:
        raise ValueError("Only square images are supported")

    apert = cam.prim.GetAttribute("horizontalAperture").Get()
    focal_len_in_pixel = cam.focal_length * img_width / apert

    intrinsics = np.eye(3)
    intrinsics[0,0] = focal_len_in_pixel
    intrinsics[1,1] = focal_len_in_pixel
    intrinsics[0,2] = img_width / 2
    intrinsics[1,2] = img_height / 2

    return intrinsics

def get_cam_extrinsics(cam):
    return T.pose_inv(T.pose2mat(cam.get_position_orientation()))

def pixel_to_3d_points(depth_image, intrinsics, extrinsics):
    # if torch.is_tensor(intrinsics):
    #     intrinsics = intrinsics.detach().cpu().numpy()
    # if torch.is_tensor(extrinsics):
    #     extrinsics = extrinsics.detach().cpu().numpy()
    # Get the shape of the depth image
    H, W = depth_image.shape

    # Create a grid of (x, y) coordinates corresponding to each pixel in the image
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')

    # Unpack the intrinsic parameters
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Convert pixel coordinates to normalized camera coordinates
    if torch.is_tensor(depth_image):
        depth_image = depth_image.detach().cpu().numpy()
    z = depth_image
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy

    # Stack the coordinates to form (H, W, 3)
    camera_coordinates = np.stack((x, y, z), axis=-1)

    # Reshape to (H*W, 3) for matrix multiplication
    camera_coordinates = camera_coordinates.reshape(-1, 3)

    # Convert to homogeneous coordinates (H*W, 4)
    camera_coordinates_homogeneous = np.hstack((camera_coordinates, np.ones((camera_coordinates.shape[0], 1))))

    # additional conversion to og convention
    T_mod = np.array([[1., 0., 0., 0., ],
              [0., -1., 0., 0.,],
              [0., 0., -1., 0.,],
              [0., 0., 0., 1.,]])
    camera_coordinates_homogeneous = camera_coordinates_homogeneous @ T_mod

    # Apply extrinsics to get world coordinates
    # world_coordinates_homogeneous = camera_coordinates_homogeneous @ extrinsics.T
    world_coordinates_homogeneous = T.pose_inv(extrinsics) @ (camera_coordinates_homogeneous.T)
    world_coordinates_homogeneous = world_coordinates_homogeneous.T

    # Convert back to non-homogeneous coordinates
    world_coordinates = world_coordinates_homogeneous[:, :3] / world_coordinates_homogeneous[:, 3, np.newaxis]

    # Reshape back to (H, W, 3)
    world_coordinates = world_coordinates.reshape(H, W, 3)

    return world_coordinates

def point_to_pixel(pt, intrinsics, extrinsics):
    """
    pt -- (N, 3) 3d points in world frame
    intrinsics -- (3, 3) intrinsics matrix
    extrinsics -- (4, 4) extrinsics matrix
    """
    pt_in_cam = extrinsics @ np.hstack((pt, np.ones((pt.shape[0], 1)))).T # (4, N)
    # multiply y, z by -1
    pt_in_cam[1, :] *= -1
    pt_in_cam[2, :] *= -1
    pt_in_cam = pt_in_cam[:3, :]
    pt_in_cam = intrinsics @ pt_in_cam
    pt_in_cam /= pt_in_cam[2, :]

    return pt_in_cam[:2, :].T


def pixel_to_world(obs, camera_info, pixel_x,  pixel_y):
    p2w = pixel_to_3d_points(obs['depth'], camera_info['intrinsics'], camera_info['extrinsics'])
    return p2w[pixel_y,pixel_x]


# # 将方向向量转为欧拉角度
# def direction_to_euler(direction):
#     import transforms3d.euler as euler
#     import transforms3d.quaternions as quaternions
#     import transforms3d.axangles as axangles

#     # 将方向向量转换为旋转矩阵
#     rotation_matrix = axangles.axangle2mat(direction, 0)
#     # 将旋转矩阵转换为四元数
#     grasp_quaternion = quaternions.mat2quat(rotation_matrix)
#     # 将四元数转换为欧拉角
#     return euler.quat2euler(grasp_quaternion)
# import torch as th
# grasp_direction = th.tensor([-0.5659, -0.1251, -0.8150])
# print(direction_to_euler(grasp_direction))
def direction_vector_to_euler_angles(unit_vector):
    # 计算 pitch (绕 y 轴)
    pitch = np.arctan2(unit_vector[2], unit_vector[0])  # z 和 x 的夹角
    # pitch_degrees = np.degrees(pitch)

    # 计算 yaw (绕 z 轴)
    yaw = np.arctan2(unit_vector[1], unit_vector[0])  # y 和 x 的夹角
    # yaw_degrees = np.degrees(yaw)

    # Roll 通常为零，因为我们假设初始方向是 (1, 0, 0)
    roll = 0.0
    # roll_degrees = np.degrees(roll)
    
    return [roll,pitch, yaw]
    
# 示例
# grasp_direction = np.array([0.5, 0.5, 0.7071])
# alpha, beta, gamma = direction_vector_to_euler_angles(grasp_direction)
# print(f"欧拉角: {alpha}, {beta}, {gamma}")
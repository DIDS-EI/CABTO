from dexrl.sim.scenarios.base_scenario import *
from omni.isaac.core.objects import  VisualCuboid
from omni.kit.viewport.utility import get_active_viewport_and_window
import numpy as np
from dexrl.sim import utils
from dexrl import global_config
import carb
from omni.isaac.sensor import Camera
from omni.ui import DockPosition, Workspace
from omni.kit.viewport.utility import create_viewport_window
from omni.isaac.core.simulation_context import SimulationContext
from .._cfg import ExtSimCfg
from dexrl.utils import format_array

class ExtScenario:
    scenario_cls:Scenario = Scenario
    scenario_cfg_cls:ScenarioCfg = ScenarioCfg
    script_generator = None
    cfg: ExtSimCfg
    viewport_camera_pos_lookat = None


    def __init__(self,cfg:ExtSimCfg):
        self.cfg = cfg
        self.set_scenario_cfg()
        self.scenario:Scenario = self.scenario_cls(self.scenario_cfg)
        self.script_generator = self.script()

    def set_scenario_cfg(self):
        self.scenario_cfg = self.scenario_cfg_cls(
            device=self.cfg.device,
        )

    def load_assets(self):
        self.scenario.load_assets()
        self.stage = self.scenario.stage
        self.robot:Robot = self.scenario.robot

        self.load_viewport()
        self.load_vision()
        self.load_ext_objects()

    def load_viewport(self):
        self.viewport_api,self.viewport_window = get_active_viewport_and_window()
        self.viewport_camera_path = self.viewport_api.camera_path
        self.viewport_camera_prim = self.stage.GetPrimAtPath(self.viewport_camera_path)

    def load_vision(self):
        if self.cfg.enable_camera_view:
            self.camera_window = create_viewport_window(
                "Top Camera",
                camera_path="/Dex/base_camera_rgb/base_camera_rgb",
                width=128,
                height=128
            )

            if not self.scenario.cfg.enable_vision:
                carb.log_info("enable_vision is not set in scenario cfg, set it to True")
                self.scenario.cfg.enable_vision = True
                self.scenario.load_vision()
                
        # FIX: [Warning] [carb] Plugin interface for a client: omni.hydratexture.plugin was already released

    def load_ext_objects(self):
        self.cube_left = VisualCuboid(
            name="RedCubeLeft",
            position=np.array([(0.30000001192092896, 0.3302656302933757, 1.1211619158771668)]),
            # orientation=euler_angles_to_quats(np.array([-140, 8, 14])),
            prim_path="/World/red_cube_left",
            size=0.05,
            color=np.array([1, 0, 0]),
        )

        self.cube_right = VisualCuboid(
            name="RedCubeRight",
            # position=np.array((-0.1, -0.4, 1.05)),
            position=np.array((-0.28,-0.40,1.06)),
            # orientation=np.array((-3.1,-2.7,-3)),
            # orientation=np.array([-0.14581,-0.14581,-0.03285,-0.42832]),
            # orientation=euler_angles_to_quats(np.array([-3.1, 2.7,-3])),
            # orientation=euler_angles_to_quats(np.array([0, 0, np.pi/4])),
            # orientation = np.array([-0.14581,-0.14581,-0.03285,-0.42832]),#np.array([-16.777,3.805,3.805]), #np.array([-0.14581,-0.14581,-0.03285,-0.42832]),
            # 1.39,-0.87 -0.24
            prim_path="/World/red_cube_right",
            size=0.05,
            color=np.array([1, 0, 0]),
        )

        self.cube_list:List[VisualCuboid] = [self.cube_left, self.cube_right]
        self.cube_use_prim_list:List[pxr.Usd.Prim] = [
            get_prim_at_path(self.cube_left.prim_path),
            get_prim_at_path(self.cube_right.prim_path)]
       
    def reset(self):
        self.scenario.reset()

    def setup(self):
        self.scenario.setup()
        if self.viewport_camera_pos_lookat is not None:
            set_camera_view(eye=self.viewport_camera_pos_lookat[:3], target=self.viewport_camera_pos_lookat[3:])
        self.setup_vision()

    def setup_vision(self):
        if self.cfg.enable_camera_view:
            self.dock_space = Workspace.get_window("DockSpace")
            self.camera_window.dock_in(self.dock_space,dock_position=DockPosition.LEFT, ratio=0.2)


    def step(self, action):
        return self.scenario.step(action)

    def set_world(self, world):
        self.scenario.set_world(world)
    
    def close(self):
        self.scenario.close()
        if self.cfg.enable_camera_view:
            print("destroy viewport window")
            self.camera_window.destroy()

    def ext_update(self):
        try:
            result = next(self.script_generator)
        except StopIteration:
            return True

    def script(self):
        yield ()


class RightArmHandFollowCubeExtScenario(ExtScenario):
    """右臂手跟随立方体
    """
    scenario_cls = Scenario
    viewport_camera_pos_lookat = -0.14,-2.78,1.69,-0.17,-1.80,1.50


    def load_ext_objects(self):
        self.cube_left = VisualCuboid(
            name="RedCubeLeft",
            position=np.array([(0.30000001192092896, 0.3302656302933757, 1.1211619158771668)]),
            # orientation=euler_angles_to_quats(np.array([-140, 8, 14])),
            prim_path="/World/red_cube_left",
            size=0.05,
            color=np.array([1, 0, 0]),
        )

        self.cube_right = VisualCuboid(
            name="RedCubeRight",
            # position=np.array((-0.1, -0.4, 1.05)),
            position=np.array((-0.306,-0.266,1.06)),
            orientation = utils.rot.euler_angles_to_quat(np.deg2rad([0,0,-30])),#np.array([-16.777,3.805,3.805]), #np.array([-0.14581,-0.14581,-0.03285,-0.42832]),
            # 1.39,-0.87 -0.24
            prim_path="/World/red_cube_right",
            size=0.05,
            color=np.array([1, 0, 0]),
        )

        self.blue_cube_right = VisualCuboid(
            name="BlueCubeRight",
            # position=np.array((-0.1, -0.4, 1.05)),
            position=np.array((-0.28,-0.40,1.06)),
            # orientation = np.array([-0.14581,-0.14581,-0.03285,-0.42832]),#np.array([-16.777,3.805,3.805]), #np.array([-0.14581,-0.14581,-0.03285,-0.42832]),
            # 1.39,-0.87 -0.24
            prim_path="/World/blue_cube_right",
            size=0.05,
            color=np.array([0, 0, 1]),
        )

        self.cube_list:List[VisualCuboid] = [self.cube_left, self.cube_right]
        self.cube_use_prim_list:List[pxr.Usd.Prim] = [
            get_prim_at_path(self.cube_left.prim_path),
            get_prim_at_path(self.cube_right.prim_path)]
       
    def script(self):

        while True:

            mixed_pose = self.robot.get_mixed_pose()
            rot_matrix = utils.rot.euler_to_rot_matrix(mixed_pose[3:6])
            lookat_direction = utils.rot.rot_matrix_to_lookat_direction(rot_matrix,forward_direction=np.array([0,0,1]))
            lookat = lookat_direction*0.2+mixed_pose[:3]

            self.blue_cube_right.set_world_pose(position=lookat)

            for i in range(1,len(self.robot.arm_hands)):
                pos_target,quat_target = self.cube_list[i].get_world_pose()
                # print(self.cube_use_prim_list[i].GetAttribute("xformOp:rotateXZY").Get())
                # pos_target[2] -= 
                euler_target = utils.rot.quat_to_euler_angles(quat_target)
                euler_target[1] += np.pi/2
                euler_target[0] -= 1

                # # 通过指向物体计算欧拉角
                self.scenario.robot.arm_hands[i].arm.set_ik_target_real(np.concatenate([pos_target, euler_target]))
                current_pos, status = self.scenario.robot.arm_hands[i].arm.reach_joint_target()
                # if status != JointControlStatus.REACHED_TARGET:
                #     print(f"current_pos: {current_pos}, status: {status}")
            yield ()



class RightArmHandCollectDataExtScenario(ExtScenario):
    """右臂手收集数据
    """
    scenario_cls = Scenario

    action_list = [
        # [0.25,-0.41,1.00, 0.05,1.37,  0,      1,1,1],
        [0.3,-0.34,1.05, 0.05,1.35,   0.4,      1,1,1],
        [0.3,-0.24,1.05, 0.05,1.35,   0.4,      1,1,1],
        # [0.255,-0.34,1.00, 0.05,1.37,   0,      1,1,1],
        # [0.252,-0.30,1.00, 0.05,1.37,   0,      1,1,1],
        # [0.25,-0.29,1.00, 0.05,1.37,    0,      0,0.8,1],
        # [0.25,-0.29,1.00, 0.05,1.37,    0,      0.25,0.8,1],
        # [0.25,-0.29,1.05, 0.05,1.37,    0,      0.25,0.8,1],
        # [0.25,-0.29,1.1, 0.05,1.37,     0,       0.25,0.6,1],
        # [0.25,-0.29,1.15, 0.05,1.37,    0,      0.25,0.6,1],
    ]
    
    def setup_viewport_camera(self):
        set_camera_view(eye=[1.11,-0.03,1.33], target=[0.16,-0.10,1.02])

    def load_ext_objects(self):
        self.red_cube_right = VisualCuboid(
            name="RedCubeRight",
            position=np.array([(0.2878412468910885, -0.39393220715837446, 1.0488733532586387)]),
            prim_path="/World/red_cube_right",
            size=0.05,
            color=np.array([1, 0, 0]),
        )
       

    def script(self):
        import pickle
        joint_pos_list = []

        self.action_length = len(self.action_list)

        self.current_replay_action_index = 0
        self.current_replay_action_step = 0
        
        yield from self.task_reset()
        
        while self.current_replay_action_index < self.action_length:
            if self.current_replay_action_step == 0:
                action = np.array(self.action_list[self.current_replay_action_index])
                self.current_joint_pos_action = action
                print(f"new action {action} ")
                self.robot.right_arm_hand.set_real_ik_action_9(self.current_joint_pos_action) 
                self.red_cube_right.set_world_pose(position=action[:3])

            # step
            print(f"current_action_index: {self.current_replay_action_index}, current_action_step: {self.current_replay_action_step}")

            current_pos, status = self.robot.right_arm_hand.reach_joint_target()
            # arm_action = self.robot.right_arm_hand.step()
            # hand_action =self.set_finger_action_3(self.finger_action)
            self.current_replay_action_step += 1
            # print(arm_action)
            # print(hand_action)

            # 记录当前 joint_pos
            # joint_pos_list.append((
            #     np.concatenate([arm_action.joint_positions,hand_action.joint_positions]),
            #     np.concatenate([arm_action.joint_velocities,np.zeros(11)])
            # ))
            if status == JointControlStatus.STUCK:
                carb.log_error(f"Joint Stucked !")

            if status == JointControlStatus.REACHED_TARGET:
                print(f"reach target !")
                self.current_replay_action_step = 0
                self.current_replay_action_index += 1

            yield ()

        # 保存 joint_pos_list
        with open(f"{global_config.root_path}/outputs/pkl/joint_pos_list.pkl", "wb") as f:
            pickle.dump(joint_pos_list, f)
        print(f"saved joint_pos_list to joint_pos_list.pkl")




class RightArmHandFollowCubeNewPoseExtScenario(ExtScenario):
    """右臂手跟随立方体
    """
    scenario_cls = Scenario
    viewport_camera_pos_lookat = -0.14,-2.78,1.69,-0.17,-1.80,1.50


    def load_ext_objects(self):

        self.pos_cube = VisualCuboid(
            name="pos_cube",
            # position=np.array((-0.1, -0.4, 1.05)),
            position=np.array((-0.306,-0.266,1.06)),
            # orientation = utils.rot.euler_angles_to_quat(np.deg2rad([0,0,-30])),#np.array([-16.777,3.805,3.805]), #np.array([-0.14581,-0.14581,-0.03285,-0.42832]),
            # 1.39,-0.87 -0.24
            prim_path="/World/pos_cube",
            size=0.05,
            color=np.array([1, 0, 0]),
        )

        self.rot_cube = VisualCuboid(
            name="rot_cube",
            position=np.array((-0.1,-0.40,1.05)),
            # orientation=euler_angles_to_quats(np.array([-140, 8, 14])),
            prim_path="/World/rot_cube",
            size=0.05,
            color=np.array([0, 0, 1]),
        )

        self.euler_y_cube = VisualCuboid(
            name="euler_y_cube",
            # position=np.array((-0.1, -0.4, 1.05)),
            position=np.array((0.0,-0.40,1.05)),
            # orientation = utils.rot.euler_angles_to_quat([0,0.6,0]),#np.array([-16.777,3.805,3.805]), #np.array([-0.14581,-0.14581,-0.03285,-0.42832]),
            # 1.39,-0.87 -0.24
            prim_path="/World/euler_y_cube",
            size=0.05,
            color=np.array([0, 0, 1]),
        )

        self.euler_z_cube = VisualCuboid(
            name="euler_z_cube",
            position=np.array((0.1, -0.4, 1.05)),
            orientation=utils.rot.euler_angles_to_quat([0,0.,-0.6]),
            # orientation = np.array([-0.14581,-0.14581,-0.03285,-0.42832]),#np.array([-16.777,3.805,3.805]), #np.array([-0.14581,-0.14581,-0.03285,-0.42832]),
            # 1.39,-0.87 -0.24
            prim_path="/World/euler_z_cube",
            size=0.05,
            color=np.array([0, 0, 1]),
        )

        self.lookat_cube = VisualCuboid(
            name="lookat_cube",
            position=np.array((0.1, -0.4, 1.05)),
            # orientation=utils.rot.euler_angles_to_quat([0,0.,-0.6]),
            # orientation = np.array([-0.14581,-0.14581,-0.03285,-0.42832]),#np.array([-16.777,3.805,3.805]), #np.array([-0.14581,-0.14581,-0.03285,-0.42832]),
            # 1.39,-0.87 -0.24
            prim_path="/World/lookat_cube",
            size=0.05,
            color=np.array([0, 1, 0]),
        )

        # self.cube_list:List[VisualCuboid] = [self.cube_left, self.pos_cube]
        # self.cube_use_prim_list:List[pxr.Usd.Prim] = [
        #     get_prim_at_path(self.cube_left.prim_path),
        #     get_prim_at_path(self.pos_cube.prim_path)]
       
    def script(self):

        while True:
            current_pose = self.robot.forward_kinematics(self.robot.right_arm_hand.arm.get_joint_positions())
            current_pos = current_pose[:3] # 位置
            current_quat = current_pose[3:7] # 旋转
            mixed_pose = self.robot.get_mixed_pose()
            rot_matrix = utils.rot.euler_to_rot_matrix(mixed_pose[3:6])
            lookat_direction = utils.rot.rot_matrix_to_lookat_direction(rot_matrix,forward_direction=np.array([0,0,1]))
            # lookat = lookat_direction*0.2+mixed_pose[:3]
            pos_target,_ = self.pos_cube.get_world_pose()
            _,quat_target = self.rot_cube.get_world_pose()
            euler = utils.rot.quat_to_euler_angles(quat_target)
            rot = euler[0]
            _,quat_target = self.euler_y_cube.get_world_pose()
            euler = utils.rot.quat_to_euler_angles(quat_target)
            euler_y = euler[1] + np.pi/2
            _,quat_target = self.euler_z_cube.get_world_pose()
            euler = utils.rot.quat_to_euler_angles(quat_target)
            euler_z = euler[2]

            yz_rot_matrix = utils.rot.euler_to_rot_matrix([0,euler_y,euler_z])
            lookat_direction = utils.rot.rot_matrix_to_lookat_direction(yz_rot_matrix,forward_direction=np.array([0,0,1]))

            x_rot_matrix = utils.rot.axis_theta_to_rot_matrix(lookat_direction,rot)
            final_rot_matrix = x_rot_matrix @ yz_rot_matrix
            final_quat = utils.rot.rot_matrix_to_quat(final_rot_matrix)

            print(final_quat)

            # print(utils.rot.rotate_quat(current_quat,lookat_direction,rot))
            # final_rot_matrix = rot_matrix @  rot
            # print(utils.rot.rot_matrix_to_quat(final_rot_matrix))
            # x_lookat_direction = utils.rot.rot_matrix_to_lookat_direction(rot_matrix,forward_direction=np.array([0,0,1]))



            # quat = utils.rot.euler_angles_to_quat([0, euler_y, euler_z])
            # rotation = R.from_quat(quat)
            # axis = rotation.as_rotvec() / np.linalg.norm(rotation.as_rotvec())
            # new_quat = rotate_quat_by_angle(quat, axis, -rot)


            # print(new_quat)
            # self.lookat_cube.set_world_pose(position=x_lookat_direction*0.2+mixed_pose[:3])

            # print(self.cube_use_prim_list[i].GetAttribute("xformOp:rotateXZY").Get())
            # pos_target[2] -= 
            # euler_target = utils.rot.quat_to_euler_angles(quat_target)
            # euler_target[1] += np.pi/2
            # euler_target[0] -= 1

            # # 通过指向物体计算欧拉角
            self.scenario.robot.right_arm_hand.arm.set_ik_target_real(np.concatenate([pos_target, [rot,euler_y,euler_z]]))
            current_pos, status = self.scenario.robot.right_arm_hand.arm.reach_joint_target()
            # if status != JointControlStatus.REACHED_TARGET:
            #     print(f"current_pos: {current_pos}, status: {status}")
            yield ()





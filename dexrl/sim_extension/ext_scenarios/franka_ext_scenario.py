from dexrl.sim.scenarios.franka import *
from omni.isaac.core.objects import  VisualCuboid
from omni.kit.viewport.utility import get_active_viewport_and_window
import numpy as np
from omni.ui import DockPosition, Workspace
from dexrl.sim.scenarios.franka import FrankaScenario,FrankaScenarioCfg
from omni.isaac.core.utils.types import ArticulationAction
try:
    from .._cfg import ExtSimCfg
except:
    from .._cfg import ExtSimCfg

# from ..generation.generated import *
from dexrl.sim.scenarios.franka import FrankaScenario,FrankaScenarioCfg
from dexrl.sim.scenarios.rl_franka_sim import RLFrankaScenario,RLFrankaScenarioCfg

class FrankaExtScenario:
    scenario_cls:FrankaScenario = FrankaScenario
    scenario_cfg_cls:FrankaScenarioCfg = FrankaScenarioCfg
    script_generator = None
    cfg: ExtSimCfg
    viewport_camera_pos_lookat = None


    def __init__(self,cfg:ExtSimCfg):
        self.cfg = cfg
        self.set_scenario_cfg()
        self.scenario:FrankaScenario = self.scenario_cls(self.scenario_cfg)
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
            from omni.kit.viewport.utility import create_viewport_window
            self.camera_window = create_viewport_window(
                "Top Camera",
                camera_path="/Dex/base_camera_rgb/base_camera_rgb",
                width=128,
                height=128
            )

            if not self.scenario.cfg.enable_vision:
                import carb
                carb.log_info("enable_vision is not set in scenario cfg, set it to True")
                self.scenario.cfg.enable_vision = True
                self.scenario.load_vision()

    def load_ext_objects(self):
        pass

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
        # pose = np.array([0,-0.2,0.15,0,0,0])
        # self.scenario.robot.kinematics_solver.compute_inverse_kinematics(pose)
        # self.scenario.robot.kinematics_solver.compute_forward_kinematics(self.scenario.robot.kinematics_solver.joint_target)
        pos = np.array([0,-0.4,0.1])
        quat = utils.rot.euler_angles_to_quat([0,np.pi,0])
        
        yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)
        print(f"pos: {pos}, quat: {quat}")
        
        close_gripper = self.scenario.close_gripper()
        yield from close_gripper
        
        pos = np.array([0.1,-0.6,0.2])
        quat = utils.rot.euler_angles_to_quat([0,np.pi,0])
        
        yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)

        open_gripper = self.scenario.open_gripper()
        yield from open_gripper
        




class RLFrankaExtScenario(FrankaExtScenario):
    """RLFranka场景
    """
    scenario: RLFrankaScenario
    scenario_cls = RLFrankaScenario
    scenario_cfg_cls = RLFrankaScenarioCfg
    viewport_camera_pos_lookat = None

    def script(self):
        # pose = np.array([0,-0.2,0.15,0,0,0])
        # self.scenario.robot.kinematics_solver.compute_inverse_kinematics(pose)
        # self.scenario.robot.kinematics_solver.compute_forward_kinematics(self.scenario.robot.kinematics_solver.joint_target)
        pos = np.array([0,-0.40253, 0.15]) #np.array([-0.12638, -0.63211, 0.2])
        # pos = np.array([0,-0.4,0.15])
        quat = utils.rot.euler_angles_to_quat([0,np.pi,0])
        
        yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)
        print(f"pos: {pos}, quat: {quat}")
        
        close_gripper = self.scenario.close_gripper()
        yield from close_gripper
        
        pos = np.array([0,-0.40253, 0.25])
        yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)
        print(f"pos: {pos}, quat: {quat}")
        
        x = 0.16168 #0.083-0.004+0.00126
        y= -0.48
        
        # (0.1641213297843933, -0.4776080846786499, 0.15057548880577087)
        
        # x = 0 #0.083-0.004+0.00126
        # y= -0.7
        
        pos = np.array([x,y,0.25])
        quat = utils.rot.euler_angles_to_quat([0,np.pi,0])
        
        yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)

        pos = np.array([x,y,0.16])
        # pos = np.array([x,y,0.16])
        quat = utils.rot.euler_angles_to_quat([0,np.pi,0])
        
        yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)


        open_gripper = self.scenario.open_gripper()
        yield from open_gripper
        
        pos = np.array([x,y,0.25])
        quat = utils.rot.euler_angles_to_quat([0,np.pi,0])
        
        yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)


class FrankaFollowCubeExtScenario(FrankaExtScenario):
    """右臂手跟随立方体
    """
    scenario_cls = FrankaScenario
    # viewport_camera_pos_lookat = -0.14,-2.78,1.69,-0.17,-1.80,1.50

    def load_ext_objects(self):

        self.pos_cube = VisualCuboid(
            name="pos_cube",
            # position=np.array((-0.1, -0.4, 1.05)),
            position=np.array((0,-0.4,0.3)),
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



    def script(self):
        # pos_cube 的 三个轴 旋转角度
        pos,quat = self.pos_cube.get_world_pose()
        init_euler = utils.rot.quat_to_euler_angles(quat)
        print(f"init_euler: {init_euler}")


        while True:
            joint = self.robot.articulation.get_joint_positions()
            arm_joint = joint[0:7]
            print(f"current_joint: {arm_joint}")
            current_pose = self.robot.forward_kinematics(arm_joint)
            print(f"current_pose: {current_pose}")
            pos,current_cube_quat = self.pos_cube.get_world_pose()
            pos = np.array([-pos[1],pos[0],pos[2]])
            quat = utils.rot.euler_angles_to_quat(np.array([0,np.pi,0]))

            # current_pose 的 三个轴 旋转角度
            current_cube_euler = utils.rot.quat_to_euler_angles(current_cube_quat)
            # print(f"current_euler: {current_euler}")
            delta_euler = current_cube_euler - init_euler
            current_quat = utils.rot.euler_angles_to_quat(np.array([0+delta_euler[0],np.pi+delta_euler[1],0+delta_euler[2]]))

            articulation_action,success = self.robot.inverse_kinematics(pos,current_quat)
            if success: 
                print(f"target_joint_pos: {articulation_action.joint_positions}")
                # articulation_action = ArticulationAction(
                #     joint_positions=np.concatenate([target_joint_pos,joint[7:]]),
                # )
                self.robot.articulation.apply_action(articulation_action)
            else:
                print("inverse kinematics failed")

            yield





class FrankaGenerationExtScenario(FrankaExtScenario):
    """右臂手跟随立方体
    """
    scenario_cls = FrankaScenario
    current_gripper_pos = 0.04
    current_action_name = None
    current_action_generator = None
    

    def create_ground_executor_map(self):
        import btvla.cfg
        from btvla.pddl.domain_problem import DomainProblem
        from ..generation import generated


        domain_path = btvla.cfg.root_path+"/btagent/test_scene_generation/domain.pddl"
        problem_path = btvla.cfg.root_path+"/btagent/test_scene_generation/problem.pddl"
        self.domprob = DomainProblem(domain_path,problem_path)

        # 提取所有 action
        action_map = {}
        all_ground_ops = []
        for op_name,op in self.domprob.domain.operators.items():
            ground_op_list = list(self.domprob.ground_operator(op_name))
            all_ground_ops.extend(ground_op_list)
            for ground_op in ground_op_list:
                ground_op_name = ground_op.get_ground_name()
                # action_map[ground_op_name] = executer_map["Action"][op_name](self,ground_op)
                action_map[ground_op_name] = getattr(generated,op_name)(self,ground_op)


        # 提取所有 condition
        all_condition_tuple = set()
        for op in all_ground_ops:
            all_condition_tuple.update(op.precondition_pos)
            all_condition_tuple.update(op.effect_pos)

        all_condition_tuple_list = sorted(list(all_condition_tuple))

        condition_map = {}
        for condition_tuple in all_condition_tuple_list:
            # condition_tuple 格式: ("On", "obj", "right_table") 或 ("In", "obj", "left_hand")
            # 过滤掉谓词名称（key 值），只传递参数部分，保持顺序
            # variable_list = [item for item in condition_tuple if item not in condition_executer_class_map.keys()]
            print(f"condition_tuple: {condition_tuple}")
            # executor = executer_map["Condition"][condition_tuple[0]](self, condition_tuple)
            executor = getattr(generated,condition_tuple[0])(self, condition_tuple)
            condition_map[executor.ground_name] = executor

        self.ground_executor_map = {"Action":action_map,"Condition":condition_map}
        print(self.ground_executor_map)


    def create_action_list(self):
        self.action_list = []
        self.create_ground_executor_map()
        plan_path = btvla.cfg.root_path+"/btagent/test_scene_generation/output.plan"
        with open(plan_path, "r") as file:
            plan = file.readlines()
        for line in plan[:-1]:
            action_tuple = line.strip()[1:-1].split(" ")
            ground_action_name = f"{action_tuple[0]}({','.join(action_tuple[1:])})"
            self.action_list.append(self.ground_executor_map["Action"][ground_action_name])
        self.action_index = 0
        self.current_action = None
        return self.action_list

        

    def create_bt(self):
        from btvla.behavior_tree.behavior_tree import BehaviorTree
        pass

    def load_ext_objects(self):
        self.obj_map = {
            "red_cube": self.scenario.red_cube,
            "blue_cube": self.scenario.blue_cube,
            "green_cube": self.scenario.green_cube,
        }

        self.create_action_list()


    def ext_update(self):
        if self.current_action is None and self.action_index < len(self.action_list):
            self.current_action = self.action_list[self.action_index]
            self.current_action_generator = self.current_action.get_action_generator()
            
        try:
            result = next(self.current_action_generator)
        except StopIteration:
            self.current_action = None
            self.action_index += 1


    def open_gripper(self):
        self.current_gripper_pos = 0.04

        articulation_action = ArticulationAction(joint_positions=[self.current_gripper_pos]*2,joint_indices=(7,8))
        self.robot.articulation.apply_action(articulation_action)
        while not np.allclose(self.robot.articulation.get_joint_positions()[7:], np.array([self.current_gripper_pos, self.current_gripper_pos]), atol=0.002):
            yield
        
    def close_gripper(self):
        self.current_gripper_pos = 0.02
        articulation_action = ArticulationAction(joint_positions=[self.current_gripper_pos]*2,joint_indices=(7,8))
        self.robot.articulation.apply_action(articulation_action)
        stuck_count = 0
        while not np.allclose(self.robot.articulation.get_joint_positions()[7:], np.array([self.current_gripper_pos, self.current_gripper_pos]), atol=0.004):
            last_gripper_pos = self.robot.articulation.get_joint_positions()[7:]
            yield
            current_gripper_pos = self.robot.articulation.get_joint_positions()[7:]
            # print(current_gripper_pos,last_gripper_pos)
            if np.allclose(last_gripper_pos, current_gripper_pos, atol=0.001):
                # print("gripper stuck")
                stuck_count += 1
                if stuck_count > 20:
                    break
            else:
                stuck_count = 0
            last_gripper_pos = current_gripper_pos


    def move_to_object(self,obj_name):
        obj = self.obj_map[obj_name]
        pos,quat = obj.get_world_pose()
        target_pos = np.array([pos[0],pos[1],pos[2]+0.04])
        yield from self.move_to_pos(target_pos,50)
        target_pos = np.array([pos[0],pos[1],pos[2]+0.0])
        yield from self.move_to_pos(target_pos,50)

    def move_on_object(self,obj_name):
        obj = self.obj_map[obj_name]
        pos,quat = obj.get_world_pose()
        target_pos = np.array([pos[0],pos[1],pos[2]+0.1])
        yield from self.move_to_pos(target_pos,200)


    def move_to_table(self):
        target_pos = np.array([0,-0.5,0.1])
        yield from self.move_to_pos(target_pos,50)

    def move_to_pos(self,pos,time_range=120):
        quat = utils.rot.euler_angles_to_quat([0,np.pi,0])
        yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)
        
        # self.robot.articulation.set_joint_velocities(np.zeros(9))
        # self.robot.articulation.set_joint_efforts(np.zeros(9))

        # quat = utils.rot.euler_angles_to_quat([0,np.pi,3/4*np.pi])
        # target_joint_pos,success = self.robot.inverse_kinematics(pos,quat)
        # # self.robot.articulation.set_joint_positions(np.concatenate([target_joint_pos,[self.current_gripper_pos]*2]))
        # # yield
        # if success:
        #     articulation_action = ArticulationAction(joint_positions=target_joint_pos,joint_indices=range(7),
        #                                              joint_velocities=np.zeros(7),joint_efforts=np.zeros(7))
        #     self.robot.articulation.apply_action(articulation_action)
        # steps = 0
        # while not np.allclose(self.robot.articulation.get_joint_positions()[:7], target_joint_pos, atol=0.001):
        #     print(self.robot.articulation.get_joint_positions()[:7]-target_joint_pos)
        #     steps += 1
        #     if steps > time_range:
        #         break
        #     yield


    def script(self):
        yield
        # yield from self.open_gripper()
        # yield from self.move_to_object("red_cube")
        # yield from self.close_gripper()
        # yield from self.move_on_object("green_cube")
        # yield from self.open_gripper()
        # yield from self.move_to_object("blue_cube")
        # yield from self.close_gripper()
        # yield from self.move_on_object("red_cube")
        # yield from self.open_gripper()
            # yield from self.close_gripper()

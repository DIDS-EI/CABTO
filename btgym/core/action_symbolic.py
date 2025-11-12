from btgym.core.simulator import Simulator
from btgym.utils.logger import log
import math
import omnigibson.utils.transform_utils as T
import torch as th
import random
from omnigibson.action_primitives.symbolic_semantic_action_primitives import SymbolicSemanticActionPrimitives,SymbolicSemanticActionPrimitiveSet
from omnigibson.action_primitives.starter_semantic_action_primitives import m,PlanningContext, ActionPrimitiveError,indented_print

m.GRASP_APPROACH_DISTANCE = 0.3
m.MAX_STEPS_FOR_GRASP_OR_RELEASE = 50
m.MAX_STEPS_FOR_HAND_MOVE_JOINT = 100
m.OPENNESS_FRACTION_TO_OPEN = 80
m.OPENNESS_THRESHOLD_TO_CLOSE = 10


class ActionPrimitives(SymbolicSemanticActionPrimitives):
    def __init__(self, simulator:Simulator):
        self.simulator = simulator
        self.og_sim = simulator.og_sim
        super().__init__(self.og_sim)


    def _grasp(self, obj):
        """
        Yields action for the robot to navigate to object if needed, then to grasp it

        Args:
            DatasetObject: Object for robot to grasp

        Returns:
            th.tensor or None: Action array for one step for the robot to grasp or None if grasp completed
        """
        # Don't do anything if the object is already grasped.
        obj_in_hand = self._get_obj_in_hand()
        if obj_in_hand is not None:
            if obj_in_hand == obj:
                return
            else:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                    "Cannot grasp when your hand is already full",
                    {"target object": obj.name, "object currently in hand": obj_in_hand.name},
                )

        # Get close
        yield from self._navigate_to_obj(obj)

        # Perform forced assisted grasp
        obj.set_position_orientation(position=self.robot.get_eef_position(self.arm))
        self.robot._establish_grasp(self.arm, (obj, obj.root_link), obj.get_position_orientation()[0])

        # Execute for a moment
        yield from self._settle_robot()

        # Verify
        if self._get_obj_in_hand() is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "Grasp completed, but no object detected in hand after executing grasp",
                {"target object": obj.name},
            )

        if self._get_obj_in_hand() != obj:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "An unexpected object was detected in hand after executing grasp. Consider releasing it",
                {"expected object": obj.name, "actual object": self._get_obj_in_hand().name},
            )

    def _navigate_to_obj(self, obj, pose_on_obj=None, **kwargs):
        """
        Yields action to navigate the robot to be in range of the pose

        Args:
            obj (StatefulObject): object to be in range of
            pose_on_obj (Iterable): (pos, quat) pose

        Returns:
            th.tensor or None: Action array for one step for the robot to navigate in range or None if it is done navigating
        """
        pose = self._sample_pose_near_object(obj, pose_on_obj=pose_on_obj, **kwargs)
        x,y,yaw = pose
        pose = (th.tensor([x, y, 0.0], dtype=th.float32), T.euler2quat(th.tensor([0, 0, yaw], dtype=th.float32)))

        self.robot.set_position_orientation(pose[0],pose[1])

        self.simulator.set_camera_lookat_robot()

        yield self._empty_action()


    def _sample_pose_near_object(self, obj, pose_on_obj=None, **kwargs):
        """
        Returns a 2d pose for the robot within in the range of the object and where the robot is not in collision with anything

        Args:
            obj (StatefulObject): Object to sample a 2d pose near
            pose_on_obj (Iterable of arrays or None): The pose to sample near

        Returns:
            2-tuple:
                - 3-array: (x,y,z) Position in the world frame
                - 4-array: (x,y,z,w) Quaternion orientation in the world frame
        """
        with PlanningContext(self.env, self.robot, self.robot_copy, "simplified") as context:
            for _ in range(m.MAX_ATTEMPTS_FOR_SAMPLING_POSE_NEAR_OBJECT):
                if pose_on_obj is None:
                    pos_on_obj = self._sample_position_on_aabb_side(obj)
                    pose_on_obj = [pos_on_obj, th.tensor([0, 0, 0, 1])]

                distance_lo, distance_hi = 0.2, 2
                distance = (th.rand(1) * (distance_hi - distance_lo) + distance_lo).item()
                yaw_lo, yaw_hi = -math.pi, math.pi
                yaw = th.rand(1) * (yaw_hi - yaw_lo) + yaw_lo
                avg_arm_workspace_range = th.mean(self.robot.arm_workspace_range[self.arm])
                pose_2d = th.cat(
                    [
                        pose_on_obj[0][0] + distance * th.cos(yaw),
                        pose_on_obj[0][1] + distance * th.sin(yaw),
                        yaw + math.pi - avg_arm_workspace_range,
                    ]
                )
                # Check room
                obj_rooms = (
                    obj.in_rooms
                    if obj.in_rooms
                    else [self.env.scene._seg_map.get_room_instance_by_point(pose_on_obj[0][:2])]
                )
                if self.env.scene._seg_map.get_room_instance_by_point(pose_2d[:2]) not in obj_rooms:
                    indented_print("Candidate position is in the wrong room.")
                    continue

                if not self._test_pose(pose_2d, context, pose_on_obj=pose_on_obj, **kwargs):
                    continue

                indented_print("Found valid position near object.")
                return pose_2d

            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.SAMPLING_ERROR,
                "Could not find valid position near object.",
                {
                    "target object": obj.name,
                    "target pos": obj.get_position_orientation()[0],
                    "pose on target": pose_on_obj,
                },
            )



class ActionSymbolic:
    action_primitive_item = SymbolicSemanticActionPrimitiveSet.GRASP

    def __init__(self,object_name):
        self.object_name = object_name
        self.object = None
        self.is_stoped = False

    def pre_start(self,action_primitives:ActionPrimitives, simulator:Simulator):
        self.object = simulator.og_sim.task.object_scope[self.object_name]
        # self.object = simulator.scene.object_registry("name", self.object_name)
        self.action_primitives = action_primitives
        self.simulator = simulator
        log(f"Action Start: {self.__class__.__name__}({self.object_name})")
        self.check_stop()
    def start(self,action_primitives:ActionPrimitives, simulator:Simulator):
        self.pre_start(action_primitives, simulator)

        self.action_control_generator = self.action_primitives.apply_ref(
                self.action_primitive_item, self.object,attempts=1)

    def check_stop(self):
        self.is_stoped = False

    def step(self):
        try:
            action = next(self.action_control_generator)
            if action is not None:
                self.simulator.add_control(action)
            else:
                log('Action is None!!')
        except Exception as e:
            log(f"Action Error: {self.__class__.__name__}({self.object_name}) ==> {e}")
            self.start(self.action_primitives, self.simulator)


class Grasp(ActionSymbolic):
    action_primitive_item = SymbolicSemanticActionPrimitiveSet.GRASP

    def check_stop(self):
        obj_in_hand = self.action_primitives._get_obj_in_hand()
        if obj_in_hand is not None:
            log(f"Object in hand: {obj_in_hand.name}")
            log(f"Action Stoped: {self.__class__.__name__}({self.object_name})")
            self.is_stoped = True



def generate_action_class(action_primitive):
    class_name = action_primitive.name
    action_class = type(class_name, (ActionSymbolic,), {'action_primitive_item': action_primitive})
    return action_class


action_symbolic_map = {
    "grasp": generate_action_class(SymbolicSemanticActionPrimitiveSet.GRASP),
    "place_on_top": generate_action_class(SymbolicSemanticActionPrimitiveSet.PLACE_ON_TOP),
    "place_inside": generate_action_class(SymbolicSemanticActionPrimitiveSet.PLACE_INSIDE),
    "open": generate_action_class(SymbolicSemanticActionPrimitiveSet.OPEN),
    "close": generate_action_class(SymbolicSemanticActionPrimitiveSet.CLOSE),
    "toggle_on": generate_action_class(SymbolicSemanticActionPrimitiveSet.TOGGLE_ON), 
    "toggle_off": generate_action_class(SymbolicSemanticActionPrimitiveSet.TOGGLE_OFF),
    "soak_under": generate_action_class(SymbolicSemanticActionPrimitiveSet.SOAK_UNDER),
    "soak_inside": generate_action_class(SymbolicSemanticActionPrimitiveSet.SOAK_INSIDE),
    "wipe": generate_action_class(SymbolicSemanticActionPrimitiveSet.WIPE),
    "cut": generate_action_class(SymbolicSemanticActionPrimitiveSet.CUT),
    "place_near_heating_element": generate_action_class(SymbolicSemanticActionPrimitiveSet.PLACE_NEAR_HEATING_ELEMENT),
    "navigate_to": generate_action_class(SymbolicSemanticActionPrimitiveSet.NAVIGATE_TO),
    "release": generate_action_class(SymbolicSemanticActionPrimitiveSet.RELEASE),
}

from btgym.core.simulator import Simulator
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitiveSet, StarterSemanticActionPrimitives
from btgym.utils.logger import log
import math

from btgym.core.simulator import Simulator
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitiveSet, StarterSemanticActionPrimitives,set_base_and_detect_collision
from btgym.utils.logger import log
import math
import omnigibson.utils.transform_utils as T
import torch as th
import random
from omnigibson.action_primitives.starter_semantic_action_primitives import m,indented_print,\
    ActionPrimitiveError,detect_robot_collision_in_sim,multi_dim_linspace,PlanningContext,\
    object_states,get_grasp_position_for_open,get_grasp_poses_for_object_sticky

from omnigibson.utils.grasping_planning_utils import JointType, _get_relevant_joints,grasp_position_for_open_on_revolute_joint,grasp_position_for_open_on_prismatic_joint
import omnigibson as og

m.GRASP_APPROACH_DISTANCE = 0.3
m.MAX_STEPS_FOR_GRASP_OR_RELEASE = 50
m.MAX_STEPS_FOR_HAND_MOVE_JOINT = 100
m.OPENNESS_FRACTION_TO_OPEN = 80
m.OPENNESS_THRESHOLD_TO_CLOSE = 10
m.MAX_ATTEMPTS_FOR_SAMPLING_POSE_NEAR_OBJECT=1000
def get_grasp_poses_for_object_sticky(target_obj):
    """
    Obtain a grasp pose for an object from top down, to be used with sticky grasping.

    Args:
        target_object (StatefulObject): Object to get a grasp pose for

    Returns:
        List of grasp candidates, where each grasp candidate is a tuple containing the grasp pose and the approach direction.
    """
    bbox_center_in_world, bbox_quat_in_world, bbox_extent_in_base_frame, _ = target_obj.get_base_aligned_bbox(
        visual=False
    )

    grasp_center_pos = bbox_center_in_world
    towards_object_in_world_frame = bbox_center_in_world - grasp_center_pos
    towards_object_in_world_frame /= th.norm(towards_object_in_world_frame)

    grasp_quat = T.euler2quat(th.tensor([0, math.pi / 2, 0], dtype=th.float32))

    grasp_pose = (grasp_center_pos, grasp_quat)
    grasp_candidate = [(grasp_pose, towards_object_in_world_frame)]

    return grasp_candidate


def get_grasp_position_for_open(robot, target_obj, should_open, relevant_joint=None, num_waypoints="default"):
    """
    Computes the grasp position for opening or closing a joint.

    Args:
      robot: the robot object
      target_obj: the object to open/close a joint of
      should_open: a boolean indicating whether we are opening or closing
      relevant_joint: the joint to open/close if we want to do a particular one in advance
      num_waypoints: the number of waypoints to interpolate between the start and end poses (default is "default")

    Returns:
      None (if no grasp was found), or Tuple, containing:
        relevant_joint: the joint that is being targeted for open/close by the returned grasp
        offset_grasp_pose_in_world_frame: the grasp pose in the world frame
        waypoints: the interpolated waypoints between the start and end poses
        approach_direction_in_world_frame: the approach direction in the world frame
        grasp_required: a boolean indicating whether a grasp is required for the opening/closing based on which side of the joint we are
        required_pos_change: the required change in position of the joint to open/close
    """
    # Pick a moving link of the object.
    relevant_joints = [relevant_joint] if relevant_joint is not None else _get_relevant_joints(target_obj)[1]
    if len(relevant_joints) == 0:
        raise ValueError("Cannot open/close object without relevant joints.")

    # Make sure what we got is an appropriately open/close joint.
    # relevant_joints = relevant_joints[0]
    # relevant_joints = relevant_joints[th.randperm(relevant_joints.size(0))]
    selected_joint = None
    for joint in relevant_joints:
        current_position = joint.get_state()[0][0]
        joint_range = joint.upper_limit - joint.lower_limit
        openness_fraction = (current_position - joint.lower_limit) / joint_range
        if (should_open and openness_fraction < m.OPENNESS_FRACTION_TO_OPEN) or (
            not should_open and openness_fraction > m.OPENNESS_THRESHOLD_TO_CLOSE
        ):
            selected_joint = joint
            break

    if selected_joint is None:
        return None

    if selected_joint.joint_type == JointType.JOINT_REVOLUTE:
        return (selected_joint,) + grasp_position_for_open_on_revolute_joint(
            robot, target_obj, selected_joint, should_open
            # robot, target_obj, selected_joint, should_open, num_waypoints=num_waypoints
        )
    elif selected_joint.joint_type == JointType.JOINT_PRISMATIC:
        return (selected_joint,) + grasp_position_for_open_on_prismatic_joint(
            robot, target_obj, selected_joint, should_open
            # robot, target_obj, selected_joint, should_open, num_waypoints=num_waypoints
        )
    else:
        raise ValueError("Unknown joint type encountered while generating joint position.")


class ActionPrimitives(StarterSemanticActionPrimitives):
    def __init__(self, simulator:Simulator):
        self.simulator = simulator
        self.og_sim = simulator.og_sim
        self._grasp = self._grasp_starter
        super().__init__(self.og_sim, enable_head_tracking=False)

    def _navigate_to_obj(self, obj, pose_on_obj=None, **kwargs):
        """
        Yields action to navigate the robot to be in range of the pose

        Args:
            obj (StatefulObject): object to be in range of
            pose_on_obj (Iterable): (pos, quat) pose

        Returns:
            th.tensor or None: Action array for one step for the robot to navigate in range or None if it is done navigating
        """

        # Allow grasping from suboptimal extents if we've tried enough times.
        grasp_poses = get_grasp_poses_for_object_sticky(obj)
        grasp_pose, object_direction = grasp_poses[0]

        # Prepare data for the approach later.
        approach_pos = grasp_pose[0]
        approach_pose = (approach_pos, grasp_pose[1])

        # get object radius
        bbox_center_in_world, bbox_quat_in_world, bbox_extent_in_base_frame, _ = obj.get_base_aligned_bbox(
            visual=False
        )
        distance_lo=0.2
        distance_hi=2
        # distance_lo = min(bbox_extent_in_base_frame[0],bbox_extent_in_base_frame[1]) + 0.3
        # distance_hi = max(bbox_extent_in_base_frame[0],bbox_extent_in_base_frame[1]) + 0.5
        # print(f"distance_lo: {distance_lo}, distance_hi: {distance_hi}")

        pose = self._sample_pose_near_object(obj, pose_on_obj=approach_pose, distance_lo=distance_lo,distance_hi=distance_hi,**kwargs)
        # pose = self._sample_pose_near_object(obj, pose_on_obj=pose_on_obj, distance_lo=distance_lo,distance_hi=distance_hi,**kwargs)

        
        x,y,yaw = pose
        pose = (th.tensor([x, y, 0.0], dtype=th.float32), T.euler2quat(th.tensor([0, 0, yaw], dtype=th.float32)))

        self.robot.set_position_orientation(pose[0],pose[1])

        # get object position
        # obj_pos = obj.get_position()
        
        # # calculate camera target position - look at object
        # self.simulator.set_camera_lookat(obj_pos)

        self.simulator.set_camera_lookat_robot()
        # self.simulator.idle_step(1)
        self.simulator.set_camera_lookat_pos(approach_pos)


    def get_object_face_tensor(self,obj,pos,horizontal=True):
        aabb_center, aabb_extent = obj.aabb_center, obj.aabb_extent
        obj_center_pos = aabb_center
        # assume object is always horizontal or vertical
        x_offset = pos[0]-obj_center_pos[0]
        y_offset = pos[1]-obj_center_pos[1]
        # convert (x_offset,y_offset) to the nearest x or y axis
        if abs(x_offset)>abs(y_offset):
            x_offset = 0
            y_offset = th.sign(y_offset)
        else:
            y_offset = 0
            x_offset = th.sign(x_offset)

        u = th.tensor([x_offset, y_offset,0])
        return u

    def _navigate_to_pos(self,obj,pos,horizontal=True,offset=(1.2,-0.2)):
        if not horizontal:
            pose = self._sample_pose_near_pos(pos,distance_lo=0.4,distance_hi=2)
        else:
            # u = self.get_object_face_tensor(obj,pos)
            u = th.tensor([0,1,0],dtype=th.float32)
            x_offset,y_offset,_ = u
            # take (x_offset,y_offset) as the positive direction of pos, get the transformation matrix from pos coordinate system to world coordinate system
            pos_to_world_matrix = th.tensor([[u[0],-u[1],pos[0]],
                                            [u[1],u[0],pos[1]],
                                             [0.,0,1]], dtype=th.float32)
            
            # convert (1.5,-0.3) in pos coordinate system to world coordinate system
            x,y,_ = pos_to_world_matrix@th.tensor([offset[0],offset[1],1], dtype=th.float32)
            # get the angle between (-y_offset,-x_offset) and (1,0)
            yaw = math.atan2(-y_offset,x_offset)

            # pose = self._sample_pose_near_pos(pos,distance_lo=2,distance_hi=3,yaw_lo=-math.pi/2,yaw_hi=math.pi/2)
        
            pose = (th.tensor([x, y, 0.0], dtype=th.float32), T.euler2quat(th.tensor([0, 0, yaw], dtype=th.float32)))


            self.robot.set_position_orientation(pose[0],pose[1])
            self.simulator.set_camera_lookat_robot()
            # self.simulator.idle_step(1)
            self.simulator.set_camera_lookat_pos(pos)

            # with PlanningContext(self.env, self.robot, self.robot_copy, "simplified") as context:
            #     for _ in range(m.MAX_ATTEMPTS_FOR_SAMPLING_POSE_NEAR_OBJECT):
            #         pose_on_obj = [pos, th.tensor([0, 0, 0, 1])]

            #         distance = (th.rand(1) * (distance_hi - distance_lo) + distance_lo).item()
            #         yaw_lo, yaw_hi = -math.pi, math.pi
            #         yaw = th.rand(1) * (yaw_hi - yaw_lo) + yaw_lo
            #         avg_arm_workspace_range = th.mean(self.robot.arm_workspace_range[self.arm])
            #         pose_2d = th.cat(
            #             [
            #                 pose_on_obj[0][0] + distance * th.cos(yaw),
            #                 pose_on_obj[0][1] + distance * th.sin(yaw),
            #                 yaw + math.pi - avg_arm_workspace_range,
            #             ]
            #         )

            #         if not self._test_pose(pose_2d, context, pose_on_obj=pose_on_obj, **kwargs):
            #             continue

            #         indented_print("Found valid position near object.")
            #         return pose_2d

            #     raise ActionPrimitiveError(
            #         ActionPrimitiveError.Reason.SAMPLING_ERROR,
            #         "Could not find valid position near object.",
            #         {
            #             "pose on target": pose_on_obj,
            #         },
            #     )





    def _grasp_symbolic(self, obj):
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

        yield from self._execute_grasp()
        # Perform forced assisted grasp
        obj.set_position_orientation(position=self.robot.get_eef_position(self.arm))
        # self.robot._establish_grasp(self.arm, (obj, obj.root_link), obj.get_position_orientation()[0])

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

    def _grasp_starter(self, obj):
        """
        Yields action for the robot to navigate to object if needed, then to grasp it

        Args:
            StatefulObject: Object for robot to grasp

        Returns:
            th.tensor or None: Action array for one step for the robot to grasp or None if grasp completed
        """
        # Update the tracking to track the object.
        self._tracking_object = obj

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

        # Open the hand first
        indented_print("Opening hand before grasping")
        yield from self._execute_release()

        # Allow grasping from suboptimal extents if we've tried enough times.
        indented_print("Sampling grasp pose")
        grasp_poses = get_grasp_poses_for_object_sticky(obj)
        grasp_pose, object_direction = random.choice(grasp_poses)

        # Prepare data for the approach later.
        approach_pos = grasp_pose[0] + object_direction * m.GRASP_APPROACH_DISTANCE
        approach_pose = (approach_pos, grasp_pose[1])

        # If the grasp pose is too far, navigate.
        indented_print("Navigating to grasp pose if needed")
        # yield from self._navigate_if_needed(obj, pose_on_obj=grasp_pose)
        yield from self._navigate_to_obj(obj, pose_on_obj=grasp_pose)

        indented_print("Moving hand to grasp pose")
        yield from self._move_hand(grasp_pose)

        # We can pre-grasp in sticky grasping mode.
        indented_print("Pregrasp squeeze")
        yield from self._execute_grasp()

        # Since the grasp pose is slightly off the object, we want to move towards the object, around 5cm.
        # It's okay if we can't go all the way because we run into the object.
        indented_print("Performing grasp approach")
        yield from self._move_hand_linearly_cartesian(approach_pose, stop_on_contact=True)
        # yield from self._move_hand_linearly_cartesian(approach_pose, stop_on_contact=True)

        # Step once to update
        empty_action = self._empty_action()
        yield self._postprocess_action(empty_action)

        indented_print("Checking grasp")
        if self._get_obj_in_hand() is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "Grasp completed, but no object detected in hand after executing grasp",
                {"target object": obj.name},
            )

        indented_print("Moving hand back")
        yield from self._reset_hand()

        indented_print("Done with grasp")

        if self._get_obj_in_hand() != obj:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "An unexpected object was detected in hand after executing grasp. Consider releasing it",
                {"expected object": obj.name, "actual object": self._get_obj_in_hand().name},
            )


    def _move_hand_linearly(
        self, dir,distance=0.5, stop_on_contact=False, ignore_failure=False, stop_if_stuck=False,ignore_obj_in_hand=False
    ):
        """
        Yields action for the robot to move its arm to reach the specified target pose by moving the eef along a line in cartesian
        space from its current pose

        Args:
            target_pose (Iterable of array): Position and orientation arrays in an iterable for pose
            stop_on_contact (boolean): Determines whether to stop move once an object is hit
            ignore_failure (boolean): Determines whether to throw error for not reaching final joint positions

        Returns:
            th.tensor or None: Action array for one step for the robot to move arm or None if its at the target pose
        """
        dis_offset = 0.02
        # To make sure that this happens in a roughly linear fashion, we will divide the trajectory
        # into 1cm-long pieces
        collision_counts = 0
        start_pos,start_orn = self.robot.eef_links[self.arm].get_position_orientation()
        while True:
            if detect_robot_collision_in_sim(self.robot, ignore_obj_in_hand=ignore_obj_in_hand):
                collision_counts += 1
                if collision_counts > 3:
                    break
            else:
                collision_counts = 0

            current_pos, current_orn = self.robot.eef_links[self.arm].get_position_orientation()
            target_pos = current_pos + dir * dis_offset
            current_distance = th.norm(target_pos-start_pos)
            if current_distance >= distance:
                break
            joint_pos = self._convert_cartesian_to_joint_space((target_pos, current_orn))
            yield from self._move_hand_direct_joint(
                joint_pos, stop_on_contact=stop_on_contact, ignore_failure=ignore_failure
            )
    

    def _move_hand_direct_joint(self, joint_pos, stop_on_contact=False, ignore_failure=False):
        """
        Yields action for the robot to move its arm to reach the specified joint positions by directly actuating with no planner

        Args:
            joint_pos (th.tensor): Array of joint positions for the arm
            stop_on_contact (boolean): Determines whether to stop move once an object is hit
            ignore_failure (boolean): Determines whether to throw error for not reaching final joint positions

        Returns:
            th.tensor or None: Action array for one step for the robot to move arm or None if its at the joint positions
        """

        # Store the previous eef pose for checking if we got stuck
        prev_eef_pos = th.zeros(3)

        # All we need to do here is save the target joint position so that empty action takes us towards it
        controller_name = f"arm_{self.arm}"
        self._arm_targets[controller_name] = joint_pos

        for i in range(m.MAX_STEPS_FOR_HAND_MOVE_JOINT):
            current_joint_pos = self.robot.get_joint_positions()[self._manipulation_control_idx]
            diff_joint_pos = joint_pos - current_joint_pos
            if th.max(th.abs(diff_joint_pos)).item() < m.JOINT_POS_DIFF_THRESHOLD:
                return
            if stop_on_contact and detect_robot_collision_in_sim(self.robot, ignore_obj_in_hand=False):
                return
            # check if the eef stayed in the same pose for sufficiently long
            if (
                og.sim.get_sim_step_dt() * i > m.TIME_BEFORE_JOINT_STUCK_CHECK
                and th.max(th.abs(self.robot.get_eef_position(self.arm) - prev_eef_pos)).item() < 0.0001
            ):
                # We're stuck!
                break

            # Since we set the new joint target as the arm_target, the empty action will take us towards it.
            action = self._empty_action()

            prev_eef_pos = self.robot.get_eef_position(self.arm)
            yield self._postprocess_action(action)

        if not ignore_failure:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "Your hand was obstructed from moving to the desired joint position",
            )


    def _sample_pose_near_pos(self, pos,distance_lo=0.2,distance_hi=2,yaw_lo=-math.pi,yaw_hi=math.pi,**kwargs):
        with PlanningContext(self.env, self.robot, self.robot_copy, "simplified") as context:
            for _ in range(m.MAX_ATTEMPTS_FOR_SAMPLING_POSE_NEAR_OBJECT):
                pose_on_obj = [pos, th.tensor([0, 0, 0, 1])]

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

                if not self._test_pose(pose_2d, context, pose_on_obj=pose_on_obj, **kwargs):
                    continue

                indented_print("Found valid position near object.")
                return pose_2d

            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.SAMPLING_ERROR,
                "Could not find valid position near object.",
                {
                    "pose on target": pose_on_obj,
                },
            )

    def _sample_pose_near_object(self, obj, pose_on_obj=None, distance_lo=0.2,distance_hi=2,**kwargs):
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
                    # pos_on_obj, aabb_extent = obj.aabb_center, obj.aabb_extent
                    
                    pose_on_obj = [pos_on_obj, th.tensor([0, 0, 0, 1])]

                # distance_lo, distance_hi = 0.2, 2
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
                # if self.env.scene._seg_map.get_room_instance_by_point(pose_2d[:2]) not in obj_rooms:
                #     indented_print("Candidate position is in the wrong room.")
                #     continue

                if not self._test_pose(pose_2d, context, pose_on_obj=pose_on_obj, **kwargs):
                    continue

                indented_print("Found valid position near object.")
                print("distance:",distance)
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

    # TODO: Why do we need to pass in the context here?
    def _test_pose(self, pose_2d, context, pose_on_obj=None):
        """
        Determines whether the robot can reach the pose on the object and is not in collision at the specified 2d pose

        Args:
            pose_2d (Iterable): (x, y, yaw) 2d pose
            context (Context): Planning context reference
            pose_on_obj (Iterable of arrays): Pose on the object in the world frame

        Returns:
            bool: True if the robot is in a valid pose, False otherwise
        """
        pose = self._get_robot_pose_from_2d_pose(pose_2d)
        if pose_on_obj is not None:
            relative_pose = T.relative_pose_transform(*pose_on_obj, *pose)
            if not self._target_in_reach_of_robot_relative(relative_pose):
                return False

        if set_base_and_detect_collision(context, pose):
            indented_print("Candidate position failed collision test.")
            return False
        return True


    def _open_or_close(self, obj, should_open):
        # Update the tracking to track the eef.
        self._tracking_object = self.robot

        if self._get_obj_in_hand():
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "Cannot open or close an object while holding an object",
                {"object in hand": self._get_obj_in_hand().name},
            )

        # Open the hand first
        yield from self._execute_release()

        for _ in range(m.MAX_ATTEMPTS_FOR_OPEN_CLOSE):
            # TODO: This needs to be fixed. Many assumptions (None relevant joint, 3 waypoints, etc.)
            if should_open:
                grasp_data = get_grasp_position_for_open(self.robot, obj, should_open, None)
            else:
                grasp_data = get_grasp_position_for_open(self.robot, obj, should_open, None, num_waypoints=3)

            if grasp_data is None:
                # We were trying to do something but didn't have the data.
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.SAMPLING_ERROR,
                    "Could not sample grasp position for target object",
                    {"target object": obj.name},
                )

            relevant_joint, grasp_pose, target_poses, object_direction, grasp_required, pos_change = grasp_data
            if abs(pos_change) < 0.1:
                indented_print("Yaw change is small and done,", pos_change)
                return

            # Prepare data for the approach later.
            approach_pos = grasp_pose[0] + object_direction * m.OPEN_GRASP_APPROACH_DISTANCE
            approach_pose = (approach_pos, grasp_pose[1])

            # If the grasp pose is too far, navigate
            yield from self._navigate_if_needed(obj, pose_on_obj=grasp_pose)

            yield from self._move_hand(grasp_pose, stop_if_stuck=True)

            # We can pre-grasp in sticky grasping mode only for opening
            if should_open:
                yield from self._execute_grasp()

            # Since the grasp pose is slightly off the object, we want to move towards the object, around 5cm.
            # It's okay if we can't go all the way because we run into the object.
            yield from self._navigate_if_needed(obj, pose_on_obj=approach_pose)

            if should_open:
                yield from self._move_hand_linearly_cartesian(
                    approach_pose, ignore_failure=False, stop_on_contact=True, stop_if_stuck=True
                )
            else:
                yield from self._move_hand_linearly_cartesian(
                    approach_pose, ignore_failure=False, stop_if_stuck=True
                )

            # Step once to update
            empty_action = self._empty_action()
            yield self._postprocess_action(empty_action)

            for i, target_pose in enumerate(target_poses):
                yield from self._move_hand_linearly_cartesian(target_pose, ignore_failure=False, stop_if_stuck=True)

            # Moving to target pose often fails. This might leave the robot's motors with torques that
            # try to get to a far-away position thus applying large torques, but unable to move due to
            # the sticky grasp joint. Thus if we release the joint, the robot might suddenly launch in an
            # arbitrary direction. To avoid this, we command the hand to apply torques with its current
            # position as its target. This prevents the hand from jerking into some other position when we do a release.
            yield from self._move_hand_linearly_cartesian(
                self.robot.eef_links[self.arm].get_position_orientation(), ignore_failure=True, stop_if_stuck=True
            )

            if should_open:
                yield from self._execute_release()
                yield from self._move_base_backward()

            # try:
            #     # TODO: This needs to be fixed. Many assumptions (None relevant joint, 3 waypoints, etc.)
            #     if should_open:
            #         grasp_data = get_grasp_position_for_open(self.robot, obj, should_open, None)
            #     else:
            #         grasp_data = get_grasp_position_for_open(self.robot, obj, should_open, None, num_waypoints=3)

            #     if grasp_data is None:
            #         # We were trying to do something but didn't have the data.
            #         raise ActionPrimitiveError(
            #             ActionPrimitiveError.Reason.SAMPLING_ERROR,
            #             "Could not sample grasp position for target object",
            #             {"target object": obj.name},
            #         )

            #     relevant_joint, grasp_pose, target_poses, object_direction, grasp_required, pos_change = grasp_data
            #     if abs(pos_change) < 0.1:
            #         indented_print("Yaw change is small and done,", pos_change)
            #         return

            #     # Prepare data for the approach later.
            #     approach_pos = grasp_pose[0] + object_direction * m.OPEN_GRASP_APPROACH_DISTANCE
            #     approach_pose = (approach_pos, grasp_pose[1])

            #     # If the grasp pose is too far, navigate
            #     yield from self._navigate_if_needed(obj, pose_on_obj=grasp_pose)

            #     yield from self._move_hand(grasp_pose, stop_if_stuck=True)

            #     # We can pre-grasp in sticky grasping mode only for opening
            #     if should_open:
            #         yield from self._execute_grasp()

            #     # Since the grasp pose is slightly off the object, we want to move towards the object, around 5cm.
            #     # It's okay if we can't go all the way because we run into the object.
            #     yield from self._navigate_if_needed(obj, pose_on_obj=approach_pose)

            #     if should_open:
            #         yield from self._move_hand_linearly_cartesian(
            #             approach_pose, ignore_failure=False, stop_on_contact=True, stop_if_stuck=True
            #         )
            #     else:
            #         yield from self._move_hand_linearly_cartesian(
            #             approach_pose, ignore_failure=False, stop_if_stuck=True
            #         )

            #     # Step once to update
            #     empty_action = self._empty_action()
            #     yield self._postprocess_action(empty_action)

            #     for i, target_pose in enumerate(target_poses):
            #         yield from self._move_hand_linearly_cartesian(target_pose, ignore_failure=False, stop_if_stuck=True)

            #     # Moving to target pose often fails. This might leave the robot's motors with torques that
            #     # try to get to a far-away position thus applying large torques, but unable to move due to
            #     # the sticky grasp joint. Thus if we release the joint, the robot might suddenly launch in an
            #     # arbitrary direction. To avoid this, we command the hand to apply torques with its current
            #     # position as its target. This prevents the hand from jerking into some other position when we do a release.
            #     yield from self._move_hand_linearly_cartesian(
            #         self.robot.eef_links[self.arm].get_position_orientation(), ignore_failure=True, stop_if_stuck=True
            #     )

            #     if should_open:
            #         yield from self._execute_release()
            #         yield from self._move_base_backward()

            # except ActionPrimitiveError as e:
            #     indented_print(e)
            #     if should_open:
            #         yield from self._execute_release()
            #         yield from self._move_base_backward()
            #     else:
            #         yield from self._move_hand_backward()

        if obj.states[object_states.Open].get_value() != should_open:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "Despite executing the planned trajectory, the object did not open or close as expected. Maybe try again",
                {"target object": obj.name, "is it currently open": obj.states[object_states.Open].get_value()},
            )


class ActionStarter:
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
        self._start()

    def check_stop(self):
        self.is_stoped = False

    def step(self):
        action = next(self.action_control_generator)
        if action is not None:
            self.simulator.add_control(action)
        else:
            log('Action is None!!')
        # try:
        #     action = next(self.action_control_generator)
        #     if action is not None:
        #         self.simulator.add_control(action)
        #     else:
        #         log('Action is None!!')
        # except Exception as e:
        #     log(f"Action Error: {self.__class__.__name__}({self.object_name}) ==> {e}")
        #     self.start(self.action_primitives, self.simulator)

            # self.check_stop()
            # if self.is_stoped:
            #     return
            # else:
            #     log(f"Action Error: {self.__class__.__name__}({self.object_name}) ==> {e}")
            #     self.start(self.action_primitives, self.simulator)

            #     if self.action_primitives._get_obj_in_hand() is None:
            #         log(f"Action Error: {self.__class__.__name__}({self.object_name}) ==> {e}")
            # else:
            #     log(f"Action Error: {self.__class__.__name__}({self.object_name}) ==> {e}")
            #     self.start(self.action_primitives, self.simulator)
            # self.simulator.reset()


class Grasp(ActionStarter):
    def __init__(self, object_name):
        super().__init__(object_name)

    def _start(self):
        self.action_control_generator = self.action_primitives.apply_ref(
            StarterSemanticActionPrimitiveSet.GRASP, self.object,attempts=1)

    def check_stop(self):
        obj_in_hand = self.action_primitives._get_obj_in_hand()
        if obj_in_hand is not None:
            log(f"Object in hand: {obj_in_hand.name}")
            log(f"Action Stoped: {self.__class__.__name__}({self.object_name})")
            self.is_stoped = True


class PlaceOnTop(ActionStarter):
    def __init__(self, object_name):
        super().__init__(object_name)

    def _start(self):
        self.action_control_generator = self.action_primitives.apply_ref(
            StarterSemanticActionPrimitiveSet.PLACE_ON_TOP, self.object,attempts=1)

    def check_stop(self):
        obj_in_hand = self.action_primitives._get_obj_in_hand()
        if obj_in_hand is None:
            log(f"No object in hand.")
            log(f"Action Stoped: {self.__class__.__name__}({self.object_name})")
            self.is_stoped = True




action_starter_map = {
    "grasp": Grasp,
    "place_on_top": PlaceOnTop,
    # "release": Release,
    # "place_inside": PlaceInside,
    # "open": Open,
    # "close": Close, 
    # "navigate_to": NavigateTo,
    # "toggle_on": ToggleOn,
    # "toggle_off": ToggleOff,
}

import grpc
from concurrent import futures
import multiprocessing as mp
from typing import Any, Dict, Type
from multiprocessing import Pipe

import btgym.simulator.simulator_pb2 as simulator_pb2
import btgym.simulator.simulator_pb2_grpc as simulator_pb2_grpc
from btgym.simulator.simulator import Simulator
import numpy as np
import time

class RPCMethod:
    """RPC method registry"""
    registry = {}

    def __init__(self, response_type: Type):
        self.response_type = response_type

    def __call__(self, func):
        method_name = func.__name__
        RPCMethod.registry[method_name] = {
            'handler': func,
            'response_type': self.response_type
        }

        def servicer_method(servicer, request, context):
            try:
                servicer._track_connection(context)
                servicer.pipe_conn.send((method_name, request))
                result = servicer.pipe_conn.recv()
                return self.response_type(**result) if result else self.response_type()
            except Exception as e:
                print(f"Error in RPC method {method_name}: {str(e)}")
                return self.response_type()

        RPCMethod.registry[method_name]['servicer_method'] = servicer_method
        return func

class SimulatorServicer(simulator_pb2_grpc.SimulatorServiceServicer):
    def __init__(self, pipe_conn):
        self.pipe_conn = pipe_conn
        self._active_connections = set()
        
    def _track_connection(self, context):
        peer = context.peer()
        self._active_connections.add(peer)
        context.add_callback(lambda: self._handle_disconnect(peer))
        return peer
    
    def _handle_disconnect(self, peer):
        self._active_connections.remove(peer)
        self.pipe_conn.send(('client_disconnected', peer))

class SimulatorCommandHandler:
    def __init__(self, simulator: Simulator):
        self.simulator = simulator

    def handle_command(self, command: str, request: Any) -> Dict:
        if command not in RPCMethod.registry:
            return {}
        
        try:
            handler = RPCMethod.registry[command]['handler']
            return handler(self, request) or {}
        except Exception as e:
            print(f"Error handling command {command}: {str(e)}")
            return {}

    @RPCMethod(simulator_pb2.Empty)
    def LoadBehaviorTask(self, request):
        self.simulator.load_behavior_task(request.task_name)
        return None

    @RPCMethod(simulator_pb2.Empty)
    def LoadCustomTask(self, request):
        self.simulator.load_custom_task(request.task_name,scene_file_name=request.scene_file_name)
        return None

    @RPCMethod(simulator_pb2.SampleCustomTaskResponse)
    def SampleCustomTask(self, request):
        json_path = self.simulator.sample_custom_task(request.task_name,scene_name=request.scene_name)
        return {'json_path': json_path}

    @RPCMethod(simulator_pb2.Empty)
    def LoadScene(self, request):
        self.simulator.load_scene(request.scene_name)
        return None


    @RPCMethod(simulator_pb2.Empty)
    def NavigateToObject(self, request):
        self.simulator.navigate_to_object(request.object_name)
        return None


    @RPCMethod(simulator_pb2.Empty)
    def InitActionPrimitives(self, request):
        self.simulator.init_action_primitives()
        return None
        
    @RPCMethod(simulator_pb2.SceneNameResponse)
    def GetSceneName(self, request) -> Dict:
        return {'scene_name': self.simulator.get_scene_name()}

    @RPCMethod(simulator_pb2.RobotPosResponse)
    def GetRobotPos(self, request) -> Dict:
        return {'position': self.simulator.get_robot_pos()}

    @RPCMethod(simulator_pb2.GetRobotJointStatesResponse)
    def GetRobotJointStates(self, request) -> Dict:
        joint_positions = self.simulator.get_joint_states()
        return {'joint_states': joint_positions.tolist() if isinstance(joint_positions, np.ndarray) else joint_positions}

    @RPCMethod(simulator_pb2.Empty)
    def SetRobotJointStates(self, request) -> Dict:
        self.simulator.set_joint_states(request.joint_states)
        return None

    @RPCMethod(simulator_pb2.EEFPoseResponse)
    def GetRobotEEFPose(self, request) -> Dict:
        pose = self.simulator.get_end_effector_pose()
        return {'eef_pose': pose.tolist() if isinstance(pose, np.ndarray) else pose}

    @RPCMethod(simulator_pb2.RelativeEEFPoseResponse)
    def GetRelativeEEFPose(self, request) -> Dict:
        pose = self.simulator.get_relative_eef_pose()
        return {'relative_eef_pose': pose.tolist() if isinstance(pose, np.ndarray) else pose}

    @RPCMethod(simulator_pb2.TaskObjectsResponse)
    def GetTaskObjects(self, request) -> Dict:
        objects = self.simulator.get_task_objects()
        return {'object_names': objects}

    @RPCMethod(simulator_pb2.Empty)
    def GraspObject(self, request) -> Dict:
        self.simulator.grasp_object(request.object_name)
        return None

    @RPCMethod(simulator_pb2.Empty)
    def ReachPose(self, request) -> Dict:
        self.simulator.reach_pose(request.pose, request.is_local)
        return None

    @RPCMethod(simulator_pb2.Empty)
    def SaveCameraImage(self, request) -> Dict:
        self.simulator.save_camera_image(request.output_path)
        return None

    @RPCMethod(simulator_pb2.Empty)
    def SetTargetVisualPose(self, request) -> Dict:
        self.simulator.set_target_visual_pose(request.pose)
        return None

    @RPCMethod(simulator_pb2.GetCameraInfoResponse)
    def GetCameraInfo(self, request) -> Dict:
        camera_info = self.simulator.get_camera_info_server()
        return camera_info

    @RPCMethod(simulator_pb2.GetObsResponse)
    def GetObs(self, request) -> Dict:
        obs = self.simulator.get_obs_server()
        return obs

    @RPCMethod(simulator_pb2.Empty)
    def SetCameraLookatPos(self, request) -> Dict:
        self.simulator.set_camera_lookat_pos(request.pos)

    @RPCMethod(simulator_pb2.Empty)
    def Close(self, request) -> Dict:
        self.simulator.close()

    @RPCMethod(simulator_pb2.GetObjectPosResponse)
    def GetObjectPos(self, request) -> Dict:
        return self.simulator.get_object_pos(request.object_name)

    @RPCMethod(simulator_pb2.GraspObjectByPosResponse)
    def GraspObjectByPos(self, request) -> Dict:
        if hasattr(request,'is_local'):
            success = self.simulator.grasp_object_by_pose(request.pos,request.object_name,request.is_local)
        else:
            success = self.simulator.grasp_object_by_pose(request.pos,request.object_name)
        return {'success': success}

    @RPCMethod(simulator_pb2.PoseToLocalResponse)
    def PoseToLocal(self, request) -> Dict:
        pose = self.simulator.pose_to_local(request.pose)
        return {'pose': pose}

# dynamically add methods to ServicerClass
for method_name, method_info in RPCMethod.registry.items():
    setattr(SimulatorServicer, method_name, method_info['servicer_method'])

def main_process_loop(pipe_conn):
    simulator = Simulator()
    print("Simulator process started, waiting for client connection")
    handler = SimulatorCommandHandler(simulator)
    
    while True:
        try:
            if pipe_conn.poll():  # check if there is new message
                command, request = pipe_conn.recv()
                if command == 'client_disconnected':
                    print(f"Client disconnected: {request}")
                    continue
                    
                result = handler.handle_command(command, request)
                pipe_conn.send(result)
            else:
                simulator.idle_step()
        except Exception as e:
            print(f"Error in main process loop: {str(e)}")

def serve():
    parent_conn, child_conn = Pipe()
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = SimulatorServicer(parent_conn)
    simulator_pb2_grpc.add_SimulatorServiceServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:51051')
    server.start()
    
    try:
        # while True:
        process = mp.Process(target=main_process_loop, args=(child_conn,))
        process.start()
        print("Simulator process is starting")
        # while process.is_alive():
        #     time.sleep(0.1)
        server.wait_for_termination()
        print("Simulator process is closed")
    except KeyboardInterrupt:
        server.stop(0)
        process.terminate()
        process.join()
    finally:
        parent_conn.close()
        child_conn.close()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    serve()
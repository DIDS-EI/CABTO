
from dataclasses import dataclass
import math

@dataclass
class state:
    target_pos = [0,0,0]
    target_euler = [0, math.pi/2, math.pi/2]
    target_local_pose = [0,0,0, 0, math.pi/2, math.pi/2]



@dataclass
class Obs:
    rgb = None
    depth = None
    seg_semantic = None
    proprio = None
    gripper_open = None
    eef_pose = None
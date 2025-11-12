import os
import sys
# import torch 
# print(torch.__version__) # 'torch2.4+cu118'
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../../../..'))
print("root_path: ",root_path)
sys.path.append(root_path)

from .ext_sim import *
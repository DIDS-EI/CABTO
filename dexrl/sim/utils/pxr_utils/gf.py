from pxr import Gf
import numpy as np

def quat_gf_to_numpy(quat:Gf.Quatd):
    return np.array([quat.real, quat.imaginary[0], quat.imaginary[0], quat.imaginary[0]])

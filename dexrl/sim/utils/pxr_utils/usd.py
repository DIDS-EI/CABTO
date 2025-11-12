import numpy as np
import pxr
from pxr import Usd
from omni.isaac.core.utils.prims import get_prim_at_path as isaacsim_get_prim_at_path
from dexrl.sim.utils.pxr_utils.gf import quat_gf_to_numpy


def get_prim_by_path(prim_path:str):
    return isaacsim_get_prim_at_path(pxr.Sdf.Path(prim_path))


def get_prim_position(prim:Usd.Prim):
    translate_attr = prim.GetAttribute("xformOp:translate")
    if translate_attr and translate_attr.Get() is not None:
        return translate_attr.Get()
    else:
        # 如果没有位置属性或属性值为None，返回零向量
        return np.array([0.0, 0.0, 0.0])

def get_prim_quat(prim:Usd.Prim):
    orient_attr = prim.GetAttribute("xformOp:orient")
    if orient_attr and orient_attr.Get() is not None:
        return quat_gf_to_numpy(orient_attr.Get())
    else:
        # 如果没有方向属性或属性值为None，返回单位四元数
        return np.array([1.0, 0.0, 0.0, 0.0])

def get_prim_euler(prim:Usd.Prim):
    rotate_attr = prim.GetAttribute("xformOp:rotateXYZ")
    if rotate_attr and rotate_attr.Get() is not None:
        return rotate_attr.Get() * np.pi/180
    else:
        # 如果没有旋转属性或属性值为None，返回零向量
        return np.array([0.0, 0.0, 0.0])

def get_prim_property_names(prim:Usd.Prim):
    return prim.GetPropertyNames()
    # print(self.viewport_camera_prim.GetPropertyNames())
    # print(self.viewport_camera_prim.GetAttribute("xformOp:rotateXYZ").Get())
    # self.viewport_camera_prim.AddXformOp(UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.PrecisionDouble, "")
    # self.viewport_camera_xform_prim = XFormPrim(str(self.viewport_camera_path))
    # print(self.viewport_api.transform)
    # print(self.viewport_api.get_active_camera())



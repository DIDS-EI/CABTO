import os
import numpy as np
from omni.isaac.core.prims import XFormPrim, XFormPrimView
from omni.isaac.core.utils.prims import add_reference_to_stage, get_prim_at_path
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats
from pxr import Gf
from dexrl import global_config
from dexrl.sim.utils import get_object_path
from omni.isaac.core.utils.stage import get_current_stage
import pxr


class BaseObject:
    usd_path = None
    xform_prim: XFormPrim

    def create_prim(self):
        NotImplementedError("create_prim is not implemented")

    def get_world_pose(self):
        return self.xform_prim.get_world_pose()

    def set_world_pose(self, position, orientation):
        self.xform_prim.set_world_pose(position, orientation)
    
    def setup_collision_api(self, prim_path, approximation_type="SDF", mass=None):
        """
        设置碰撞体API
        
        Args:
            prim_path: 物体prim路径
            approximation_type: 碰撞体近似类型
                - "SDF": SDF Mesh (最精确，支持复杂几何体)
                - "convexHull": 凸包 (性能好，适合简单物体)
                - "convexDecomposition": 凸分解 (平衡精度和性能)
                - "box": 包围盒 (最快，精度最低)
                - "sphere": 球体 (最快，适合球形物体)
            mass: 物体质量 (可选)
        """
        from pxr import UsdPhysics, Sdf
        stage = get_current_stage()
        
        # 获取几何体prim
        geometry_prim = get_prim_at_path(f"{prim_path}/geometry")
        if geometry_prim:
            # 应用碰撞API
            collision_api = UsdPhysics.CollisionAPI.Apply(geometry_prim)
            
            # 设置碰撞近似类型
            collision_api.CreateApproximationAttr().Set(approximation_type)
            
            # 设置碰撞过滤组
            collision_api.CreateFilteredGroupsAttr().Set(["default"])
            
            # 设置碰撞偏移 (可选)
            collision_api.CreateOffsetAttr().Set((0.0, 0.0, 0.0))
            
            # 设置质量 (如果指定了mass)
            if mass is not None:
                mass_api = UsdPhysics.MassAPI.Apply(geometry_prim)
                mass_api.CreateMassAttr().Set(mass)
            
            print(f"已为 {prim_path} 设置 {approximation_type} 碰撞体近似")
            return collision_api
        else:
            print(f"警告: 未找到几何体prim: {prim_path}/geometry")
            return None



class ThickCoconutMilk(BaseObject):

    usd_path = f"{global_config.assets_path}/objects/thick_coconut_milk/thick_coconut_milk.usd"
    scale_ratio = 0.002

    def __init__(self, scale=None, position=None, orientation=None,mass=0.2,name="thick_coconut_milk"):
        self.prim_path = f"/World/{name}"
        self.name = name
        self.scale = scale
        self.position = position
        self.mass = mass
        self.orientation = orientation if orientation is not None else euler_angles_to_quats([0, 0, 0])

    def create_prim(self):
        add_reference_to_stage(self.usd_path, self.prim_path)
        self.xform_prim = XFormPrim(
            prim_path=self.prim_path,
            name=self.name,
            scale=[self.scale_ratio,self.scale_ratio,0.0013],
            position=self.position,
            orientation=self.orientation,
        )
        # self.xform_prim.set_world_pose(
        #     position=np.array([0.6, 0,0.96]),
        #     orientation=euler_angles_to_quats([0, 0, 0])  # 默认朝向
        # )
        # prim = get_prim_at_path("/World/target/geometry")
        # scale = prim.GetAttribute('xformOp:scale')
        # print(f"scale: {scale}")
        # prim.GetAttribute('xformOp:translate').Set(Gf.Vec3f(0.06,0.,0.1))
        # prim.GetAttribute('xformOp:scale').Set(Gf.Vec3f(0.2,0.1,0.2))
        return self.xform_prim
    
    def reset(self):
        self.xform_prim.set_world_pose(position=self.position, orientation=self.orientation)


class GreenTea(BaseObject):

    usd_path = f"{global_config.assets_path}/objects/GreenTea/usd/B48V1const.usd"
    scale_ratio = 0.001

    def __init__(self, scale=None, position=None, orientation=None,mass=0.2):
        self.prim_path = "/World/obstacle"
        self.name = "obstacle"
        self.scale = scale
        self.position = position
        self.mass = mass
        self.orientation = orientation if orientation is not None else euler_angles_to_quats([0, 0, 0])

    def create_prim(self):
        add_reference_to_stage(self.usd_path, self.prim_path)
        self.xform_prim = XFormPrim(
            prim_path="/World/obstacle",
            name="obstacle",
            scale=[self.scale_ratio,0.001,self.scale_ratio],
            position=self.position,
            orientation=self.orientation,
        )
        # self.xform_prim.set_world_pose(
        #     position=np.array([0.6, 0,0.96]),
        #     orientation=euler_angles_to_quats([0, 0, 0])  # 默认朝向
        # )
        # prim = get_prim_at_path("/World/target/geometry")
        # scale = prim.GetAttribute('xformOp:scale')
        # print(f"scale: {scale}")
        # prim.GetAttribute('xformOp:translate').Set(Gf.Vec3f(0.06,0.,0.1))
        # prim.GetAttribute('xformOp:scale').Set(Gf.Vec3f(0.2,0.1,0.2))
        return self.xform_prim
    
    def reset(self):
        self.xform_prim.set_world_pose(position=self.position, orientation=self.orientation)
class Milk(BaseObject):

    usd_path = f"{global_config.assets_path}/objects/milk/B31V2const.usd"
    scale_ratio_x = 0.000924
    scale_ratio_y = 0.0008775
    scale_ratio_z = 0.0009776

    def __init__(self, scale=None, position=None, orientation=None,mass=0.2):
        self.prim_path = "/World/milk"
        self.name = "milk"
        self.scale = scale
        self.position = position
        self.mass = mass
        self.orientation = orientation if orientation is not None else euler_angles_to_quats([0, 0, 0])

    def create_prim(self):
        add_reference_to_stage(self.usd_path, self.prim_path)
        self.xform_prim = XFormPrim(
            prim_path="/World/milk",
            name="milk",
            scale=[self.scale_ratio_x,self.scale_ratio_y,self.scale_ratio_z],
            position=self.position,
            orientation=self.orientation,
        )
        # self.xform_prim.set_world_pose(
        #     position=np.array([0.6, 0,0.96]),
        #     orientation=euler_angles_to_quats([0, 0, 0])  # 默认朝向
        # )
        # prim = get_prim_at_path("/World/target/geometry")
        # scale = prim.GetAttribute('xformOp:scale')
        # print(f"scale: {scale}")
        # prim.GetAttribute('xformOp:translate').Set(Gf.Vec3f(0.06,0.,0.1))
        # prim.GetAttribute('xformOp:scale').Set(Gf.Vec3f(0.2,0.1,0.2))
        return self.xform_prim
    
    def reset(self):
        self.xform_prim.set_world_pose(position=self.position, orientation=self.orientation)

 
 
class Lego(BaseObject):

    usd_path = f"{global_config.assets_path}/objects/lego/1x2/lego_green_2.usd" #1x2.obj 1x2.urdf
    scale_ratio_x = 0.01
    scale_ratio_y = 0.01
    scale_ratio_z = 0.01

    def __init__(self, scale=None, position=None, orientation=None,mass=0.2,color="green", prim_path=None, name=None):
        self.prim_path = prim_path if prim_path is not None else "/World/target"
        self.name = name if name is not None else "target"
        self.scale = scale
        self.position = position
        self.mass = mass
        self.orientation = orientation if orientation is not None else euler_angles_to_quats([0, 0, 0])
        self.usd_path = f"{global_config.assets_path}/objects/lego/1x2/lego_{color}_2.usd"
        
        
    def create_prim(self):
        add_reference_to_stage(self.usd_path, self.prim_path)
        # self.prim = get_prim_at_path(self.prim_path)
        self.xform_prim = XFormPrim(
            prim_path=self.prim_path,
            name=self.name,
            scale=[self.scale_ratio_x,self.scale_ratio_y,self.scale_ratio_z],
            position=self.position,
            orientation=self.orientation,
        )
        
        # # 设置物理属性 - 先检查属性是否存在
        # mass_attr = self.prim.GetAttribute('physics:mass')
        # if not mass_attr:
        #     # 如果属性不存在，创建它
        #     from pxr import Sdf
        #     mass_attr = self.prim.CreateAttribute('physics:mass', Sdf.ValueTypeNames.Float)
        # mass_attr.Set(self.mass)
        
        # # 设置纯绿色材质
        # from pxr import UsdShade, Sdf, Gf
        
        # # 获取stage
        # stage = get_current_stage()
        
        # # 创建材质
        # material_path = f"{self.prim_path}/Material"
        # material = UsdShade.Material.Define(stage, material_path)
        
        # # 创建着色器
        # shader_path = f"{material_path}/Shader"
        # shader = UsdShade.Shader.Define(stage, shader_path)
        # shader.CreateIdAttr("UsdPreviewSurface")
        
        # # 设置纯绿色
        # shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 1.0, 0.0))  # 纯绿色
        # shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        # shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        
        # # 连接着色器到材质
        # material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        
        # # 将材质绑定到几何体
        # geometry_prim = get_prim_at_path(f"{self.prim_path}/geometry")
        # if geometry_prim:
        #     UsdShade.MaterialBindingAPI(geometry_prim).Bind(material)
        
        return self.xform_prim
    
    def reset(self):
        self.xform_prim.set_world_pose(position=self.position, orientation=self.orientation)


class BigLego(BaseObject):

    usd_path = f"{global_config.assets_path}/objects/lego/1x2/lego_green_2.usd" #1x2.obj 1x2.urdf
    scale_ratio_x = 0.012
    scale_ratio_y = 0.012
    scale_ratio_z = 0.012

    def __init__(self, scale=None, position=None, orientation=None,mass=0.2,color="green", prim_path=None, name=None, rgb=None):
        self.prim_path = prim_path if prim_path is not None else "/World/big_lego"
        self.name = name if name is not None else "big_lego"
        self.scale = scale
        self.position = position
        self.mass = mass
        self.orientation = orientation if orientation is not None else euler_angles_to_quats([0, 0, 0])
        self.usd_path = f"{global_config.assets_path}/objects/lego/1x2/lego_{color}_2.usd"
        self.rgb = rgb
        
        
    def create_prim(self):
        add_reference_to_stage(self.usd_path, self.prim_path)
        # self.prim = get_prim_at_path(self.prim_path)
        self.xform_prim = XFormPrim(
            prim_path=self.prim_path,
            name=self.name,
            scale=[self.scale_ratio_x,self.scale_ratio_y,self.scale_ratio_z],
            position=self.position,
            orientation=self.orientation,
        )
        if self.rgb is not None:
            # 使用pxr设置颜色
            from pxr import UsdShade, Sdf, Gf
            stage = get_current_stage()
            
            # 创建材质
            material_path = f"{self.prim_path}/Material"
            material = UsdShade.Material.Define(stage, material_path)
            
            # 创建着色器
            shader_path = f"{material_path}/Shader"
            shader = UsdShade.Shader.Define(stage, shader_path)
            shader.CreateIdAttr("UsdPreviewSurface")
            
            # 设置颜色
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(self.rgb[0], self.rgb[1], self.rgb[2]))
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
            
            # 连接着色器到材质
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            
            # 将材质绑定到几何体
            geometry_prim = get_prim_at_path(f"{self.prim_path}/geometry")
            if geometry_prim:
                UsdShade.MaterialBindingAPI(geometry_prim).Bind(material)
        
        return self.xform_prim
    
    def reset(self):
        self.xform_prim.set_world_pose(position=self.position, orientation=self.orientation)

class Canned(BaseObject):
    usd_path = f"{global_config.assets_path}/objects/canned/A2V1const.usd"
    scale_ratio_x = 0.000924
    scale_ratio_y = 0.0008775
    scale_ratio_z = 0.0009776

    def __init__(self, scale=None, position=None, orientation=None,mass=0.2,name="canned"):
        self.prim_path = f"/World/{name}"
        self.name = name
        self.scale = scale
        self.position = position
        self.mass = mass
        self.orientation = orientation if orientation is not None else euler_angles_to_quats([0, 0, 0])

    def create_prim(self):
        add_reference_to_stage(self.usd_path, self.prim_path)
        self.xform_prim = XFormPrim(
            prim_path=self.prim_path,
            name=self.name,
            scale=[self.scale_ratio_x,self.scale_ratio_y,self.scale_ratio_z],
            position=self.position,
            orientation=self.orientation,
        )
        return self.xform_prim
    
    def reset(self):
        self.xform_prim.set_world_pose(position=self.position, orientation=self.orientation)

class Carton(BaseObject):

    # usd_path = f"{global_config.assets_path}/objects/drawer/drawer.usd"
    usd_path = f"{global_config.assets_path}/objects/carton/cdmmwy.usd"
    scale_ratio_x = 1
    scale_ratio_y = 1
    scale_ratio_z = 1

    def __init__(self, scale=None, position=None, orientation=None,mass=0.2):
        self.prim_path = "/World/carton"
        self.name = "carton"
        self.scale = scale
        self.position = position
        self.mass = mass
        self.orientation = orientation if orientation is not None else euler_angles_to_quats([0, 0, 0])

    def create_prim(self):
        add_reference_to_stage(self.usd_path, self.prim_path)
        self.xform_prim = XFormPrim(
            prim_path="/World/carton",
            name="carton",
            scale=[self.scale_ratio_x,self.scale_ratio_y,self.scale_ratio_z],
            position=self.position,
            orientation=self.orientation,
        )
        return self.xform_prim
    
    def reset(self):
        self.xform_prim.set_world_pose(position=self.position, orientation=self.orientation)    
    
class Tea(BaseObject):

    usd_path = f"{global_config.assets_path}/objects/tea/B48V1const.usd" #1x2.obj 1x2.urdf
    scale_ratio_x = 0.000924
    scale_ratio_y = 0.0008775
    scale_ratio_z = 0.0006776 #0.0009776

    def __init__(self, scale=None, position=None, orientation=None,mass=0.2, prim_path=None, name=None):
        self.prim_path = prim_path if prim_path is not None else f"/World/{name}"
        self.name = name if name is not None else f"{name}"
        self.scale = scale
        self.position = position
        self.mass = mass
        self.orientation = orientation if orientation is not None else euler_angles_to_quats([0, 0, 0])
        
        
    def create_prim(self):
        add_reference_to_stage(self.usd_path, self.prim_path)
        # self.prim = get_prim_at_path(self.prim_path)
        self.xform_prim = XFormPrim(
            prim_path=self.prim_path,
            name=self.name,
            scale=[self.scale_ratio_x,self.scale_ratio_y,self.scale_ratio_z],
            position=self.position,
            orientation=self.orientation,
        )

        return self.xform_prim
    
    def reset(self):
        self.xform_prim.set_world_pose(position=self.position, orientation=self.orientation)
    
        
# myx plate
class Boiler(BaseObject):
    usd_path = f"{global_config.assets_path}/objects/Boiler/usd/E13V1const.usd" #1x2.obj 1x2.urdf
    scale_ratio_x = 0.000924
    scale_ratio_y = 0.0008775
    scale_ratio_z = 0.0006776 #0.0009776

    def __init__(self, scale=None, position=None, orientation=None,mass=0.2, prim_path=None, name=None):
        self.prim_path = prim_path if prim_path is not None else f"/World/{name}"
        self.name = name if name is not None else f"{name}"
        self.scale = scale
        self.position = position
        self.mass = mass
        self.orientation = orientation if orientation is not None else euler_angles_to_quats([0, 0, 0])
        
        
    def create_prim(self):
        add_reference_to_stage(self.usd_path, self.prim_path)
        # self.prim = get_prim_at_path(self.prim_path)
        self.xform_prim = XFormPrim(
            prim_path=self.prim_path,
            name=self.name,
            scale=[self.scale_ratio_x,self.scale_ratio_y,self.scale_ratio_z],
            position=self.position,
            orientation=self.orientation,
        )

        return self.xform_prim
    
    def reset(self):
        self.xform_prim.set_world_pose(position=self.position, orientation=self.orientation)
# myx plate
class ChopBoard(BaseObject):
    usd_path = f"{global_config.assets_path}/objects/ChopBoard/usd/E10V1const.usd" #1x2.obj 1x2.urdf
    scale_ratio_x = 0.000924
    scale_ratio_y = 0.0008775
    scale_ratio_z = 0.0006776 #0.0009776

    def __init__(self, scale=None, position=None, orientation=None,mass=0.2, prim_path=None, name=None):
        self.prim_path = prim_path if prim_path is not None else f"/World/{name}"
        self.name = name if name is not None else f"{name}"
        self.scale = scale
        self.position = position
        self.mass = mass
        self.orientation = orientation if orientation is not None else euler_angles_to_quats([0, 0, 0])
        
        
    def create_prim(self):
        add_reference_to_stage(self.usd_path, self.prim_path)
        # self.prim = get_prim_at_path(self.prim_path)
        self.xform_prim = XFormPrim(
            prim_path=self.prim_path,
            name=self.name,
            scale=[self.scale_ratio_x,self.scale_ratio_y,self.scale_ratio_z],
            position=self.position,
            orientation=self.orientation,
        )

        return self.xform_prim
    
    def reset(self):
        self.xform_prim.set_world_pose(position=self.position, orientation=self.orientation)

class Cup(BaseObject):
    usd_path = f"{global_config.assets_path}/objects/E20V1/usd/E20V1const.usd" #1x2.obj 1x2.urdf
    scale_ratio_x = 0.000924
    scale_ratio_y = 0.0008775
    scale_ratio_z = 0.0006776 #0.0009776

    def __init__(self, scale=None, position=None, orientation=None,mass=0.2, prim_path=None, name=None):
        self.prim_path = prim_path if prim_path is not None else f"/World/{name}"
        self.name = name if name is not None else f"{name}"
        self.scale = scale
        self.position = position
        self.mass = mass
        self.orientation = orientation if orientation is not None else euler_angles_to_quats([0, 0, 0])
        
        
    def create_prim(self):
        add_reference_to_stage(self.usd_path, self.prim_path)
        # self.prim = get_prim_at_path(self.prim_path)
        self.xform_prim = XFormPrim(
            prim_path=self.prim_path,
            name=self.name,
            scale=[self.scale_ratio_x,self.scale_ratio_y,self.scale_ratio_z],
            position=self.position,
            orientation=self.orientation,
        )

        return self.xform_prim
    
    def reset(self):
        self.xform_prim.set_world_pose(position=self.position, orientation=self.orientation)


class BreadPI(BaseObject):

    usd_path = f"{global_config.assets_path}/objects/BreadPI/usd/A19V2const.usd" #1x2.obj 1x2.urdf
    scale_ratio_x = 0.000924
    scale_ratio_y = 0.0008775
    scale_ratio_z = 0.0006776 #0.0009776

    def __init__(self, scale=None, position=None, orientation=None,mass=0.2, prim_path=None, name=None):
        self.prim_path = prim_path if prim_path is not None else f"/World/{name}"
        self.name = name if name is not None else f"{name}"
        self.scale = scale
        self.position = position
        self.mass = mass
        self.orientation = orientation if orientation is not None else euler_angles_to_quats([0, 0, 0])
        
        
    def create_prim(self):
        add_reference_to_stage(self.usd_path, self.prim_path)
        # self.prim = get_prim_at_path(self.prim_path)
        self.xform_prim = XFormPrim(
            prim_path=self.prim_path,
            name=self.name,
            scale=[self.scale_ratio_x,self.scale_ratio_y,self.scale_ratio_z],
            position=self.position,
            orientation=self.orientation,
        )

        return self.xform_prim
    
    def reset(self):
        self.xform_prim.set_world_pose(position=self.position, orientation=self.orientation)
  

class Bowl(BaseObject):

    usd_path = f"{global_config.assets_path}/objects/bowl/bowl.usd" #1x2.obj 1x2.urdf
    scale_ratio_x = 0.000924*1.5
    scale_ratio_y = 0.0008775*1.5
    scale_ratio_z = 0.0009776*1.5

    def __init__(self, scale=None, position=None, orientation=None,mass=0.2, prim_path=None, name="bowl"):
        self.prim_path = f"/World/{name}"
        self.name = name
        self.scale = scale
        self.position = position
        self.mass = mass
        self.orientation = orientation if orientation is not None else euler_angles_to_quats([0, 0, 0])
        
        
    def create_prim(self):
        add_reference_to_stage(self.usd_path, self.prim_path)
        # self.prim = get_prim_at_path(self.prim_path)
        self.xform_prim = XFormPrim(
            prim_path=self.prim_path,
            name=self.name,
            scale=[self.scale_ratio_x,self.scale_ratio_y,self.scale_ratio_z],
            position=self.position,
            orientation=self.orientation,
        )

        return self.xform_prim
    
    def reset(self):
        self.xform_prim.set_world_pose(position=self.position, orientation=self.orientation)
      
class Pan(BaseObject):

    usd_path = f"{global_config.assets_path}/objects/pan/E13V1const.usd" #1x2.obj 1x2.urdf
    scale_ratio_x = 0.000924*1.5
    scale_ratio_y = 0.0008775*1.5
    scale_ratio_z = 0.0009776*1.5

    def __init__(self, scale=None, position=None, orientation=None,mass=0.2, prim_path=None, name=None):
        self.prim_path = prim_path if prim_path is not None else "/World/Pan"
        self.name = name if name is not None else "Pan"
        self.scale = scale
        self.position = position
        self.mass = mass
        self.orientation = orientation if orientation is not None else euler_angles_to_quats([0, 0, 0])
        
        
    def create_prim(self):
        add_reference_to_stage(self.usd_path, self.prim_path)
        # self.prim = get_prim_at_path(self.prim_path)
        self.xform_prim = XFormPrim(
            prim_path=self.prim_path,
            name=self.name,
            scale=[self.scale_ratio_x,self.scale_ratio_y,self.scale_ratio_z],
            position=self.position,
            orientation=self.orientation,
        )

        return self.xform_prim
    
    def reset(self):
        self.xform_prim.set_world_pose(position=self.position, orientation=self.orientation)
    
    
class Turner(BaseObject):

    usd_path = f"{global_config.assets_path}/objects/turner/E13V1const.usd" #1x2.obj 1x2.urdf
    scale_ratio_x = 0.000924*1.5
    scale_ratio_y = 0.0008775*1.5
    scale_ratio_z = 0.0009776*1.5

    def __init__(self, scale=None, position=None, orientation=None,mass=0.2, prim_path=None, name=None):
        self.prim_path = prim_path if prim_path is not None else "/World/Turner"
        self.name = name if name is not None else "Turner"
        self.scale = scale
        self.position = position
        self.mass = mass
        self.orientation = orientation if orientation is not None else euler_angles_to_quats([0, 0, 0])
        
        
    def create_prim(self):
        add_reference_to_stage(self.usd_path, self.prim_path)
        # self.prim = get_prim_at_path(self.prim_path)
        self.xform_prim = XFormPrim(
            prim_path=self.prim_path,
            name=self.name,
            scale=[self.scale_ratio_x,self.scale_ratio_y,self.scale_ratio_z],
            position=self.position,
            orientation=self.orientation,
        )

        return self.xform_prim
    
    def reset(self):
        self.xform_prim.set_world_pose(position=self.position, orientation=self.orientation)
      


class BlueCup(BaseObject):

    usd_path = f"{global_config.assets_path}/objects/BlueCup/usd/E18V1const.usd" #1x2.obj 1x2.urdf
    # usd_path = f"{global_config.assets_path}/objects/E20V1/usd/E20V1const.usd" #1x2.obj 1x2.urdf
    scale_ratio_x = 0.000924*1.5
    scale_ratio_y = 0.0008775*1.5
    scale_ratio_z = 0.0009776*1.5

    def __init__(self, scale=None, position=None, orientation=None,mass=0.2, prim_path=None, name="BlueCup"):
        self.prim_path = f"/World/{name}"
        self.name = name
        self.scale = scale
        self.position = position
        self.mass = mass
        self.orientation = orientation if orientation is not None else euler_angles_to_quats([0, 0, 0])
        
        
    def create_prim(self):
        add_reference_to_stage(self.usd_path, self.prim_path)
        # self.prim = get_prim_at_path(self.prim_path)
        self.xform_prim = XFormPrim(
            prim_path=self.prim_path,
            name=self.name,
            scale=[self.scale_ratio_x,self.scale_ratio_y,self.scale_ratio_z],
            position=self.position,
            orientation=self.orientation,
        )

        return self.xform_prim
    
    def reset(self):
        self.xform_prim.set_world_pose(position=self.position, orientation=self.orientation)




#############
# og datasets
########
class Apple(BaseObject):

    usd_path = f"{global_config.assets_path}/objects/og_apple/usd/omzprq.usd" #1x2.obj 1x2.urdf
    scale_ratio_x = 0.7
    scale_ratio_y = 0.7
    scale_ratio_z = 0.7

    def __init__(self, scale=None, position=None, orientation=None,mass=0.2, prim_path=None, name="apple"):
        self.prim_path = f"/World/{name}"
        self.name = name
        self.scale = scale
        self.position = position
        self.mass = mass
        self.orientation = orientation if orientation is not None else euler_angles_to_quats([0, 0, 0])
        
        
    def create_prim(self):
        add_reference_to_stage(self.usd_path, self.prim_path)
        # self.prim = get_prim_at_path(self.prim_path)
        self.xform_prim = XFormPrim(
            prim_path=self.prim_path,
            name=self.name,
            scale=[self.scale_ratio_x,self.scale_ratio_y,self.scale_ratio_z],
            position=self.position,
            orientation=self.orientation,
        )

        return self.xform_prim
    
    def reset(self):
        self.xform_prim.set_world_pose(position=self.position, orientation=self.orientation)
  


class Shrimp(BaseObject):

    usd_path = f"{global_config.assets_path}/objects/og_shrimp/usd/mnpgev.usd" #1x2.obj 1x2.urdf
    scale_ratio_x = 1.5
    scale_ratio_y = 1.5
    scale_ratio_z = 1.5

    def __init__(self, scale=None, position=None, orientation=None,mass=0.2, prim_path=None, name="banana"):
        self.prim_path = f"/World/{name}"
        self.name = name
        self.scale = scale
        self.position = position
        self.mass = mass
        self.orientation = orientation if orientation is not None else euler_angles_to_quats([0, 0, 0])
        
        
    def create_prim(self):
        add_reference_to_stage(self.usd_path, self.prim_path)
        # self.prim = get_prim_at_path(self.prim_path)
        self.xform_prim = XFormPrim(
            prim_path=self.prim_path,
            name=self.name,
            scale=[self.scale_ratio_x,self.scale_ratio_y,self.scale_ratio_z],
            position=self.position,
            orientation=self.orientation,
        )

        return self.xform_prim
    
    def reset(self):
        self.xform_prim.set_world_pose(position=self.position, orientation=self.orientation)
  



class Potato(BaseObject):

    usd_path = f"{global_config.assets_path}/objects/og_potato/usd/bpwohr.usd" #1x2.obj 1x2.urdf
    scale_ratio_x = 1
    scale_ratio_y = 1
    scale_ratio_z = 1

    def __init__(self, scale=None, position=None, orientation=None,mass=0.2, prim_path=None, name="potato"):
        self.prim_path = f"/World/{name}"
        self.name = name
        self.scale = scale
        self.position = position
        self.mass = mass
        self.orientation = orientation if orientation is not None else euler_angles_to_quats([0, 0, 0])
        
        
    def create_prim(self):
        add_reference_to_stage(self.usd_path, self.prim_path)
        # self.prim = get_prim_at_path(self.prim_path)
        self.xform_prim = XFormPrim(
            prim_path=self.prim_path,
            name=self.name,
            scale=[self.scale_ratio_x,self.scale_ratio_y,self.scale_ratio_z],
            position=self.position,
            orientation=self.orientation,
        )

        return self.xform_prim
    
    def reset(self):
        self.xform_prim.set_world_pose(position=self.position, orientation=self.orientation)
  
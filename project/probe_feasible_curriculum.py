"""Opt-in easier task definitions. Geometry/config may change ONLY before reset.

No object qpos edits, robot rehome, success flags or snap during task execution.
Default grasp-weld and oracle tracking remain explicit assistance. Hand-written
reference policies establish feasibility; they are NOT generated-policy results.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import types
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parent
STAGE=ROOT.parent/"CABTO"/"exp4_bt_tasks"/"stage2_scripted"
sys.path.insert(0,str(STAGE))
import scene_dual_common as dc
from exp4_env import Exp4Env
from arm_skills import ArmSkills
from probe_dual_no_snap import Scout, containment
from probe_five_scene_preflight import camera_params, fit_task_cameras, penetrations, visibility, task_bodies


def vec(xs): return " ".join(str(float(x)) for x in xs)

def validate_config(c):
    t=c["table"];e=c["evaluation"]
    assert 0<t["thickness"]<t["top_z"], "table must be above floor"
    assert all(v>0 for v in t["full_size_xy"])
    assert 0<e["stability_seconds"]<=10, "nonempty bounded stability window required"
    assert 0<e["position_tolerance_m"]<=.03
    assert 0<=e["layout_jitter_m"]<=.02
    assert [r["suffix"] for r in c["robots"]]==["_L","_R"]
    assert c["pour"]["ball_radius"]<c["pour"]["can_inner_radius"]
    assert c["packing"]["tray_wall"]>0


def build_scene(config,task,seed=0):
    validate_config(config)
    c=deepcopy(config); h=c["table"]["top_z"]; p=c[task]
    visual=c.get("visual_style",{});storage_style=task=="packing" and visual.get("profile")=="paper_storage"
    physical_cardboard=storage_style and bool(visual.get("physical_cardboard",False))
    rng=np.random.default_rng(seed)
    jitter=c["evaluation"]["layout_jitter_m"] if seed else 0.0
    bodies=[]; welds=[]; names=[]
    def weld(name,arm):
        welds.append(f'<weld name="grasp_weld{arm}_{name}" body1="hand{arm}" body2="{name}" active="false" solimp="0.95 0.99 0.001" solref="0.005 1"/>')
    def block(name,xy,half,mat,arms):
        xy=np.asarray(xy)+rng.uniform(-jitter,jitter,2)
        bodies.append(f'<body name="{name}" pos="{vec([*xy,h+half[2]+.002])}"><freejoint name="{name}_free"/><geom type="box" size="{vec(half)}" material="{mat}" mass="0.04" friction="1.0 0.02 0.001"/></body>')
        names.append(name)
        for arm in arms: weld(name,arm)
    def marker(name,xy,rgb):
        bodies.append(f'<body name="{name}" pos="{vec([*xy,h+.0005])}"><geom type="cylinder" size="0.04 0.0005" rgba="{rgb} 0.6" contype="0" conaffinity="0"/></body>')
    def vessel(name,xy,radius,height,free,mat):
        wall=p["can_wall"]; bottom=p["can_floor"] if free else p["cup_floor"]
        origin=p.get("handle_height",height/2) if free else 0.0
        geom=f'<geom type="cylinder" size="{radius+wall} {bottom/2}" pos="0 0 {bottom/2-origin}" material="{mat}" mass="0.035" solref="{p["contact_timeconstant"]} 1" solimp="0.99 0.99 0.001"/>'
        # Overlapping tangential boxes form closed walls, not disconnected posts.
        n=32; rr=radius+wall/2; tangent=rr*math.tan(math.pi/n)*1.04
        for i in range(n):
            a=2*math.pi*i/n
            geom+=f'<geom type="box" size="{wall/2} {tangent} {height/2}" pos="{rr*math.cos(a)} {rr*math.sin(a)} {height/2-origin}" quat="{math.cos(a/2)} 0 0 {math.sin(a/2)}" material="{mat}" mass="0.001" friction="0.4 0.01 0.001"/>'
        if free:
            # Body origin is the grasp handle; keep the palm away from the outlet.
            # This is rigid scene geometry, not a runtime grasp/object teleport.
            off=p.get("handle_offset",0.0)
            if off:
                node=ET.fromstring("<body>"+geom+"</body>")
                for g in node.findall("geom"):
                    xyz=list(map(float,g.get("pos").split()));xyz[0]-=off;g.set("pos",vec(xyz))
                geom="".join(ET.tostring(g,encoding="unicode") for g in node)
                geom+=f'<geom type="box" size="0.012 0.014 0.012" pos="0 0 0" material="{mat}" mass="0.015"/>'
                length=off-radius
                geom+=f'<geom type="box" size="{length/2} 0.007 0.007" pos="{-length/2} 0 0" material="{mat}" mass="0.010"/>'
                stem=max(.005,origin-height/2)
                geom+=f'<geom type="box" size="0.007 0.007 {stem/2}" pos="{-length} 0 {-stem/2}" material="{mat}" mass="0.010"/>'
        z=h+origin+.002 if free else h
        bodies.append(f'<body name="{name}" pos="{vec([*xy,z])}">'+(f'<freejoint name="{name}_free"/>' if free else "")+geom+'</body>')
        if free:names.append(name)
    if task=="relay":
        block("box0",p["start_xy"],p["object_half_size"],"mat_emerald",["_L","_R"])
        marker("relay_station",p["station_xy"],"0.95 0.75 0.15")
        marker("destination",p["destination_xy"],"0.2 0.6 0.9")
        for row in p.get("extra_objects",[]):
            block(row["name"],row["start_xy"],p["object_half_size"],"mat_blue",["_L","_R"])
            marker("destination_"+row["name"],row["destination_xy"],"0.2 0.6 0.9")
    elif task=="packing":
        if storage_style:
            half=np.asarray(p["item_half_size"],float)
            for i,xy0 in enumerate(p["sources_xy"]):
                name=f"item{i}";xy=np.asarray(xy0)+rng.uniform(-jitter,jitter,2)
                yaw=math.radians(p.get("source_yaws_deg",[0.0,0.0])[i]);quat=vec([math.cos(yaw/2),0,0,math.sin(yaw/2)])
                core=f'<geom name="{name}_collision_core" type="box" size="{vec(half)}" material="mat_toy_green" mass="0.04" friction="1.0 0.02 0.001"/>'
                if i==0:
                    detail=''.join(f'<geom name="{name}_stud_{j}" type="cylinder" size="0.0062 0.0035" pos="{sx} 0 {half[2]+.0035}" material="mat_toy_green_light" mass="0" contype="0" conaffinity="0"/>' for j,sx in enumerate((-.009,.009)))
                else:
                    detail='<geom name="item1_hinge_bar" type="box" size="0.015 0.006 0.005" pos="0 0 0.020" material="mat_toy_green_light" mass="0" contype="0" conaffinity="0"/>'
                    detail+=''.join(f'<geom name="item1_stud_{j}" type="cylinder" size="0.0058 0.0035" pos="{sx} 0 0.028" material="mat_toy_green_light" mass="0" contype="0" conaffinity="0"/>' for j,sx in enumerate((-.008,.008)))
                    detail+='<geom name="item1_axle_L" type="cylinder" size="0.0045 0.005" pos="0 0.021 0.006" quat="0.7071068 0.7071068 0 0" material="mat_toy_green_light" mass="0" contype="0" conaffinity="0"/><geom name="item1_axle_R" type="cylinder" size="0.0045 0.005" pos="0 -0.021 0.006" quat="0.7071068 0.7071068 0 0" material="mat_toy_green_light" mass="0" contype="0" conaffinity="0"/>'
                bodies.append(f'<body name="{name}" pos="{vec([*xy,h+half[2]+.002])}" quat="{quat}"><freejoint name="{name}_free"/>{core}{detail}</body>')
                names.append(name);weld(name,("_L","_R")[i])
        else:
            for i,xy in enumerate(p["sources_xy"]): block(f"item{i}",xy,p["item_half_size"],("mat_green","mat_blue")[i],[("_L","_R")[i]])
        ix,iy=p["tray_inner_half_size"];w=p["tray_wall"];height=p["tray_height"]
        free=p.get("tray_free",False);mass=p.get("tray_mass",0.6);tray_friction=vec(p.get("tray_friction",[.6,.01,.001]))
        def ga(frac):return f' mass="{mass*frac:.4f}" friction="{tray_friction}"' if free else ''
        handled=bool(p.get("dual_lift_handles"))
        floor_frac=.40 if physical_cardboard and handled else .46 if handled else .6
        if storage_style:
            carton_outer="mat_cardboard_outer";carton_inner="mat_cardboard_inner"
            geoms=f'<geom name="tray_floor" type="box" size="{ix+w} {iy+w} {w/2}" pos="0 0 {w/2}" material="{carton_inner}"{ga(floor_frac)}/>'
            for sign in (-1,1):
                geoms+=f'<geom name="tray_wall_x_{sign}" type="box" size="{w/2} {iy+w} {height/2}" pos="{sign*(ix+w/2)} 0 {height/2}" material="{carton_outer}"{ga(0.1)}/>'
                geoms+=f'<geom name="tray_wall_y_{sign}" type="box" size="{ix+w} {w/2} {height/2}" pos="0 {sign*(iy+w/2)} {height/2}" material="{carton_outer}"{ga(0.1)}/>'
            if physical_cardboard:
                # Two short rigid flaps are part of the same free box body.  Each
                # overlaps a physical hinge cylinder and the top edge of its wall;
                # unlike v2 sites, they carry mass and participate in collision.
                fl=p["flap_half_length"];fw=p["flap_half_width"];ft=p["flap_half_thickness"]
                angle=math.radians(p["flap_angle_deg"]);cx=ix+w+fl*math.cos(angle);cz=height+fl*math.sin(angle)
                for sign,label in ((-1,"left"),(1,"right")):
                    q=vec([math.cos(angle/2),0,-sign*math.sin(angle/2),0])
                    geoms+=f'<geom name="carton_flap_{label}" type="box" size="{fl} {fw} {ft}" pos="{sign*cx} 0 {cz}" quat="{q}" material="{carton_outer}"{ga(.02)}/>'
                    geoms+=f'<geom name="carton_hinge_{label}" type="cylinder" size="{ft*1.3} {fw}" pos="{sign*(ix+w)} 0 {height}" quat="0.7071068 0.7071068 0 0" material="{carton_outer}"{ga(.01)}/>'
            else:
                # v2 presentation-only flaps retained for historical replay.
                geoms+=f'<site name="carton_flap_left" type="box" size="0.042 {iy} 0.0015" pos="{-ix-.032} 0 {height+.027}" quat="0.9238795 0 -0.3826834 0" rgba="0.80 0.62 0.37 1"/>'
                geoms+=f'<site name="carton_flap_right" type="box" size="0.042 {iy} 0.0015" pos="{ix+.032} 0 {height+.027}" quat="0.9238795 0 0.3826834 0" rgba="0.80 0.62 0.37 1"/>'
            geoms+='<site name="carton_tape" type="box" size="0.018 0.081 0.0008" pos="0 0 0.007" rgba="0.92 0.76 0.48 1"/><site name="carton_label" type="box" size="0.025 0.001 0.013" pos="0 -0.086 0.038" rgba="0.94 0.88 0.70 1"/><site name="carton_print_1" type="box" size="0.002 0.0008 0.009" pos="-0.010 -0.087 0.038" rgba="0.20 0.16 0.11 1"/><site name="carton_print_2" type="box" size="0.002 0.0008 0.009" pos="0 -0.087 0.038" rgba="0.20 0.16 0.11 1"/><site name="carton_print_3" type="box" size="0.002 0.0008 0.009" pos="0.010 -0.087 0.038" rgba="0.20 0.16 0.11 1"/>'
        else:
            carton_outer="mat_carton"
            geoms=f'<geom type="box" size="{ix+w} {iy+w} {w/2}" pos="0 0 {w/2}" material="mat_carton_in"{ga(floor_frac)}/>'
            for sign in (-1,1):
                geoms+=f'<geom type="box" size="{w/2} {iy+w} {height/2}" pos="{sign*(ix+w/2)} 0 {height/2}" material="mat_carton"{ga(0.1)}/>'
                geoms+=f'<geom type="box" size="{ix+w} {w/2} {height/2}" pos="0 {sign*(iy+w/2)} {height/2}" material="mat_carton"{ga(0.1)}/>'
        fj='<freejoint name="tray_free"/>' if free else ''
        if handled:
            hy=p["lift_handle_y"];hz=p["lift_handle_z"];joint_y=iy+w+.008
            geoms+=f'<geom name="tray_handle_L" type="box" size="0.022 0.010 0.010" pos="0 {hy} {hz}" material="{carton_outer}"{ga(0.035)}/>'
            geoms+=f'<geom name="tray_handle_R" type="box" size="0.022 0.010 0.010" pos="0 {-hy} {hz}" material="{carton_outer}"{ga(0.035)}/>'
            geoms+=f'<geom name="tray_handle_joint_L" type="box" size="0.022 0.008 0.010" pos="0 {joint_y} {hz}" material="{carton_outer}"{ga(0.035)}/>'
            geoms+=f'<geom name="tray_handle_joint_R" type="box" size="0.022 0.008 0.010" pos="0 {-joint_y} {hz}" material="{carton_outer}"{ga(0.035)}/>'
        bodies.append(f'<body name="tray" pos="{vec([*p["tray_xy"],h])}">{fj}{geoms}</body>')
        if free:names.append("tray")
        if free and p.get("tray_dual_weld"):
            weld("tray","_L");weld("tray","_R")
        if "shelf_xy" in p:
            sx,sy=p["shelf_half_size_xy"];top=p["shelf_top_z"]
            if storage_style:
                height=top-h;platform=.018;leg=.018
                # Keep the v1 solid collision pedestal exactly; make it invisible
                # and overlay a visual-only open wood/metal shelf with sites.
                shelf=f'<geom name="shelf_collision_core" type="box" size="{sx} {sy} {height/2}" pos="0 0 {height/2}" material="mat_carton_in" rgba="0 0 0 0" friction="1.0 0.02 0.001"/>'
                shelf+=f'<site name="shelf_top" type="box" size="{sx} {sy} {platform}" pos="0 0 {height-platform}" material="mat_shelf_wood"/>'
                shelf+=f'<site name="shelf_lower" type="box" size="{sx-.018} {sy-.018} 0.010" pos="0 0 0.030" material="mat_shelf_wood_dark"/>'
                for dx in (-sx+leg,sx-leg):
                    for dy in (-sy+leg,sy-leg):
                        shelf+=f'<site name="shelf_leg_{dx}_{dy}" type="box" size="{leg} {leg} {height/2-platform}" pos="{dx} {dy} {height/2-platform}" material="mat_shelf_metal"/>'
                shelf+=f'<site name="shelf_front_trim" type="box" size="{sx} 0.006 0.008" pos="0 {-sy+.006} {height-.012}" rgba="0.20 0.12 0.075 1"/>'
                bodies.append(f'<body name="shelf" pos="{vec([*p["shelf_xy"],h])}">{shelf}</body>')
            else:
                bodies.append(f'<body name="shelf" pos="{vec([*p["shelf_xy"],h])}"><geom type="box" size="{sx} {sy} {(top-h)/2}" pos="0 0 {(top-h)/2}" material="mat_carton_in" friction="1.0 0.02 0.001"/></body>')
    else:
        held=p.get("held_basin")
        if held:
            xy=np.asarray(held["source_xy"])+rng.uniform(-jitter,jitter,2)
            vessel("canL",xy,p["can_inner_radius"],p["can_height"],True,"mat_red")
            if held.get("source_stability_foot_radius"):
                fr=held["source_stability_foot_radius"];ft=held["source_stability_foot_thickness"]
                foot=f'<geom name="source_stability_foot" type="cylinder" size="{fr} {ft/2}" pos="{-p["handle_offset"]} 0 {ft/2-p["handle_height"]}" material="mat_red" mass="0.035" friction="1.2 0.02 0.002"/>'
                bodies[-1]=bodies[-1].replace("</body>",foot+"</body>")
            weld("canL","_L")
            r=p["ball_radius"];ballxy=xy-np.array([p.get("handle_offset",0.0),0])
            bodies.append(f'<body name="ballL" pos="{vec([*ballxy,h+.002+p["can_floor"]+r+.002])}"><freejoint name="ballL_free"/><geom type="sphere" size="{r}" material="mat_blue" mass="0.008" condim="{p.get("ball_contact_dimension",3)}" solref="{p["contact_timeconstant"]} 1" solimp="0.99 0.99 0.001" friction="{vec(p["ball_friction"])}"/></body>')
            names.append("ballL")
            if held.get("source_dock_inner_half_size"):
                dx,dy=held["source_dock_inner_half_size"];dw=held["source_dock_wall"];dh=held["source_dock_height"]
                dock=f'<geom type="box" size="{dx+dw} {dy+dw} {dw/2}" pos="0 0 {dw/2}" material="mat_carton_in" friction="1.2 0.02 0.002"/>'
                for sign in (-1,1):
                    dock+=f'<geom type="box" size="{dw/2} {dy+dw} {dh/2}" pos="{sign*(dx+dw/2)} 0 {dh/2}" material="mat_carton"/>'
                    dock+=f'<geom type="box" size="{dx+dw} {dw/2} {dh/2}" pos="0 {sign*(dy+dw/2)} {dh/2}" material="mat_carton"/>'
                dock_xy=np.asarray(held.get("source_drop_center_xy",ballxy))
                bodies.append(f'<body name="source_dock" pos="{vec([*dock_xy,h])}">{dock}</body>')
            br=held["basin_inner_radius"];bw=held["basin_wall"];bh=held["basin_height"]
            bf=held["basin_floor"];bo=held["basin_handle_offset"];bz=held["basin_handle_height"]
            geom=f'<geom type="cylinder" size="{br+bw} {bf/2}" pos="{-bo} 0 {bf/2-bz}" material="mat_cup" mass="0.18" solref="{p["contact_timeconstant"]} 1" solimp="0.99 0.99 0.001"/>'
            n=32;rr=br+bw/2;tangent=rr*math.tan(math.pi/n)*1.04
            for k in range(n):
                a=2*math.pi*k/n
                geom+=f'<geom type="box" size="{bw/2} {tangent} {bh/2}" pos="{rr*math.cos(a)-bo} {rr*math.sin(a)} {bh/2-bz}" quat="{math.cos(a/2)} 0 0 {math.sin(a/2)}" material="mat_cup" mass="0.004" friction="0.5 0.02 0.002"/>'
            length=max(.005,bo-br)
            geom+=f'<geom type="box" size="0.014 0.016 0.014" pos="0 0 0" material="mat_carton" mass="0.03"/>'
            geom+=f'<geom type="box" size="{length/2} 0.009 0.009" pos="{-length/2} 0 0" material="mat_carton" mass="0.02"/>'
            stem=max(.005,bz-bh/2)
            geom+=f'<geom type="box" size="0.009 0.009 {stem/2}" pos="{-length} 0 {-stem/2}" material="mat_carton" mass="0.02"/>'
            center=np.asarray(held["basin_start_center_xy"])+rng.uniform(-jitter,jitter,2)
            handle=center+np.array([bo,0.0])
            bodies.append(f'<body name="basin" pos="{vec([*handle,h+bz+.002])}"><freejoint name="basin_free"/>{geom}</body>')
            names.append("basin");weld("basin","_R")
        else:
            for i,s in enumerate(("L","R")):
                xy=np.asarray(p["sources_xy"][i])+rng.uniform(-jitter,jitter,2)
                vessel(f"can{s}",xy,p["can_inner_radius"],p["can_height"],True,"mat_red" if i==0 else "mat_green")
                weld(f"can{s}","_"+s)
                r=p["ball_radius"];ballxy=xy-np.array([p.get("handle_offset",0.0),0])
                bodies.append(f'<body name="ball{s}" pos="{vec([*ballxy,h+.002+p["can_floor"]+r+.002])}"><freejoint name="ball{s}_free"/><geom type="sphere" size="{r}" material="mat_blue" mass="0.008" condim="{p.get("ball_contact_dimension",3)}" solref="{p["contact_timeconstant"]} 1" solimp="0.99 0.99 0.001" friction="{vec(p["ball_friction"])}"/></body>')
                names.append(f"ball{s}")
                vessel(f"cup{s}",p["receivers_xy"][i],p["cup_inner_radius"],p["cup_height"],False,"mat_cup")
    root=ET.fromstring(dc.make_dual_xml("curriculum_"+task,"\n".join(bodies),"\n".join(welds)))
    assets=ROOT/"mujoco_workspace/franka_articulated/assets/panda_assets"
    root.find("compiler").set("meshdir",str(assets))
    if storage_style:
        asset=root.find("asset")
        for name,rgba in (("mat_toy_green","0.12 0.66 0.06 1"),("mat_toy_green_light","0.24 0.82 0.10 1"),
                          ("mat_cardboard_outer","0.72 0.50 0.25 1"),("mat_cardboard_inner","0.84 0.65 0.39 1"),
                          ("mat_studio_table","0.72 0.74 0.75 1"),("mat_table_edge","0.33 0.27 0.23 1"),
                          ("mat_shelf_wood","0.30 0.17 0.10 1"),("mat_shelf_wood_dark","0.20 0.11 0.07 1"),
                          ("mat_shelf_metal","0.16 0.19 0.21 1")):
            ET.SubElement(asset,"material",{"name":name,"rgba":rgba,"specular":"0.18","shininess":"0.20"})
        sky=asset.find("texture[@type='skybox']")
        if sky is not None:sky.set("rgb1","0.34 0.67 0.82");sky.set("rgb2","0.24 0.52 0.70")
        grid=asset.find("texture[@name='grid']")
        if grid is not None:grid.set("rgb1","0.24 0.55 0.72");grid.set("rgb2","0.30 0.63 0.79")
        table_mat=asset.find("material[@name='table']")
        if table_mat is not None:table_mat.set("rgba","0.72 0.74 0.75 1")
        dark_mat=asset.find("material[@name='table_dark']")
        if dark_mat is not None:dark_mat.set("rgba","0.23 0.25 0.27 1")
    table=root.find("./worldbody/body[@name='table']")
    t=c["table"];th=t["thickness"]/2
    table.set("pos",vec([*t["center_xy"],h-th]));table.find("geom").set("size",vec([*(np.asarray(t["full_size_xy"])/2),th]))
    if storage_style:
        tx,ty=np.asarray(t["full_size_xy"])/2
        ET.SubElement(table,"site",{"name":"table_front_edge","type":"box","size":vec([tx,.008,.018]),"pos":vec([0,-ty+.008,-.010]),"rgba":"0.30 0.25 0.22 1"})
        ET.SubElement(table,"site",{"name":"table_back_edge","type":"box","size":vec([tx,.008,.018]),"pos":vec([0,ty-.008,-.010]),"rgba":"0.30 0.25 0.22 1"})
    # Geometry support/legs are reconstructed for any configured table size/height.
    world=root.find("worldbody")
    for g in list(world.findall("geom")):
        if g.get("name")!="floor":world.remove(g)
    for dx in (-t["full_size_xy"][0]/2+.06,t["full_size_xy"][0]/2-.06):
        for dy in (-t["full_size_xy"][1]/2+.06,t["full_size_xy"][1]/2-.06):
            ET.SubElement(world,"geom",{"type":"box","size":vec([.025,.025,(h-t["thickness"])/2]),"pos":vec([t["center_xy"][0]+dx,t["center_xy"][1]+dy,(h-t["thickness"])/2]),"material":"table_dark"})
    if storage_style:
        for x in np.arange(-2.5,2.6,.5):
            ET.SubElement(world,"geom",{"name":f"studio_grid_x_{x}","type":"box","size":"0.004 2.5 0.0006","pos":vec([x,0,.001]),"rgba":"0.91 0.98 1 0.80","contype":"0","conaffinity":"0"})
        for y in np.arange(-2.5,2.6,.5):
            ET.SubElement(world,"geom",{"name":f"studio_grid_y_{y}","type":"box","size":"2.5 0.004 0.0006","pos":vec([0,y,.001]),"rgba":"0.91 0.98 1 0.80","contype":"0","conaffinity":"0"})
    for robot in c["robots"]:
        node=root.find(f"./worldbody/body[@name='link0{robot['suffix']}']")
        if node is None:raise ValueError("cannot locate robot base")
        node.set("pos",vec([*robot["base_xy"],h+robot["base_above_table"]]))
        a=robot["yaw"];node.set("quat",vec([math.cos(a/2),0,0,math.sin(a/2)]))
    xml=ET.tostring(root,encoding="unicode")
    xml=fit_task_cameras(xml,task_bodies(xml))
    if storage_style:
        styled=ET.fromstring(xml);target=np.asarray(visual["camera_target"],float)
        for name,key in (("overview","camera_overview"),("front","camera_front")):
            cam=styled.find(f"./worldbody/camera[@name='{name}']");pos=np.asarray(visual[key],float)
            cam.set("pos",vec(pos));cam.set("xyaxes",vec(camera_params(pos,target)));cam.set("fovy","44" if name=="overview" else "48")
        xml=ET.tostring(styled,encoding="unicode")
    return xml,names


class Curriculum(Scout):
    def __init__(self,config,task,out,seed=0):
        self.config=config;self.p=config[task];self.h=config["table"]["top_z"]
        self.task=task;self.out=out;out.mkdir(parents=True,exist_ok=True)
        xml,names=build_scene(config,task,seed);(out/"scene.xml").write_text(xml)
        (out/"config.json").write_text(json.dumps(config,indent=2))
        (out/"experiment_source.py").write_bytes(Path(__file__).read_bytes())
        self.env=Exp4Env(xml,[(r["suffix"],r["yaw"]) for r in config["robots"]],names,render=True,img_size=560)
        self.env.reset();self.env.start_record();self.env.record_frame("overview")
        self.events=[];self.steps=[];self.arms=self.env.arms
        half=self.p.get("object_half_size",self.p.get("item_half_size",[0,0,self.p.get("can_height",.08)/2]))[2]
        self.skills=[ArmSkills(self.env,a,recorder=lambda:self.env.record_frame("overview"),rec_every=3,block_half=half) for a in self.arms]
        for a in self.arms:
            a.grasp_z_tol=.09;original=a._set_weld
            def logged(arm,name,active,_original=original):
                self.events.append({"arm":arm.s,"object":name,"active":bool(active),"sim_time":float(self.env.d.time),"object_position":self.pos(name).tolist(),"finger_mid":arm._finger_mid().tolist()})
                return _original(name,active)
            a._set_weld=types.MethodType(logged,a)
        self.result={"task":task,"task_definition":self.p["label"],"source":"handwritten_reference","seed":seed,
            "perception":"oracle","control":"grasp_weld_assisted","object_snap":False,"arm_rehome_during_execution":False,
            "steps":self.steps,"assistance_events":self.events,"success":False,"failure":None,
            "scene_sha256":hashlib.sha256(xml.encode()).hexdigest(),
            "initial_penetrations":penetrations(self.env.m,self.env.d,task_bodies(xml)),
            "initial_visibility":visibility(self.env.m,self.env.d,task_bodies(xml),size=560),
            "source_hashes":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/"probe_dual_no_snap.py",STAGE/"arm_skills.py",STAGE/"exp4_env.py"]}}
        self.env.save_png(str(out/"before.png"))
    def pick(self,i,name,z):
        sk=self.skills[i];p=self.pos(name).copy()
        self.go(i,[p[0],p[1],self.h+.24],1.0);self.go(i,[p[0],p[1],z],1.0)
        sk.grasp(self.pos(name)[:2],n=28)
        if not self.arms[i].is_holding(name):raise RuntimeError(f"{name} not held")
        sk.lift(top=self.h+.30);sk.record_grab_offset(name)
        if not self.arms[i].is_holding(name):raise RuntimeError(f"{name} lost on lift")
        self.mark("picked",object=name,arm=i)
    def transport(self,i,name,xy,z=None):
        return super().transport(i,name,xy,z=self.h+.30 if z is None else z)
    def place(self,i,name,xy,top):
        sk=self.skills[i];self.transport(i,name,xy);sk.record_grab_offset(name)
        sk.descend_place_tracked(np.asarray(xy),top,name,gap=.003,suppress_feedforward=True)
        sk.place_release()
        now=sk.ee();self.go(i,[now[0],now[1],self.h+.30],1.0)
        self.mark("placed",object=name,arm=i)
    def park(self,i):
        xy=(self.p["donor_park_xy"] if self.task=="relay" and i==0 else
            [self.config["robots"][i]["base_xy"][0],.28 if i==0 else -.28])
        self.go(i,[*xy,self.h+.32],1.0)
    def released(self,name):return not any(a.is_holding(name) for a in self.arms)
    def near_table(self,name,xy,half):
        p=self.pos(name)
        return bool(np.linalg.norm(p[:2]-xy)<self.config["evaluation"]["position_tolerance_m"] and abs(p[2]-(self.h+half))<.008 and self.released(name))
    def stable(self,predicate):
        steps=math.ceil(self.config["evaluation"]["stability_seconds"]/.032);ok=True
        for _ in range(steps):self.hold(1);ok=bool(predicate()) and ok
        self.result["stability_seconds"]=steps*.032
        return ok
    def relay(self):
        if self.p.get("extra_objects"):
            raise ValueError("this handwritten reference covers one object only; use the multi-object BT runner")
        p=self.p;name="box0";half=p["object_half_size"][2]
        self.result["initial_goal_false"]=not self.near_table(name,p["destination_xy"],half)
        if not self.result["initial_goal_false"]:raise ValueError("invalid trial: goal already true")
        self.pick(0,name,self.pos(name)[2]+.015)
        self.place(0,name,p["station_xy"],self.h);self.park(0);self.hold(16)
        station=self.near_table(name,p["station_xy"],half);self.mark("station_released",ok=station)
        if not station:raise RuntimeError("donor did not leave a stable released object at relay station")
        self.pick(1,name,self.pos(name)[2]+.015)
        self.place(1,name,p["destination_xy"],self.h);self.park(1)
        self.result["success"]=self.stable(lambda:self.near_table(name,p["destination_xy"],half))
        self.result["final_error_mm"]=float(np.linalg.norm(self.pos(name)[:2]-p["destination_xy"])*1000)
    def packing(self):
        p=self.p
        def checks():return {f"item{i}":containment(self.env.get_object_pose(f"item{i}"),(self.env.get_body_pos("tray"),np.array([1,0,0,0])),p["item_half_size"],p["tray_inner_half_size"],floor=p["tray_wall"],ceiling=p["tray_height"]) for i in (0,1)}
        self.result["initial_goal_false"]=not any(v["inside"] for v in checks().values())
        if not self.result["initial_goal_false"]:raise ValueError("invalid trial: an item starts in tray")
        for i in (0,1):
            name=f"item{i}";self.pick(i,name,self.pos(name)[2]+.015)
            xy=self.env.get_body_pos("tray")[:2]+p["slots_xy"][i]
            self.place(i,name,xy,self.h+p["tray_wall"]);self.park(i)
        self.result["success"]=self.stable(lambda:all(v["inside"] for v in checks().values()) and all(self.released(f"item{i}") for i in (0,1)))
        self.result["final_containment"]=checks()
    def pour(self):
        p=self.p
        def inside(ball,cup):
            b=self.pos(ball);c=self.env.get_body_pos(cup);r=p["ball_radius"]
            return bool(np.linalg.norm(b[:2]-c[:2])+r<p["cup_inner_radius"]-.002 and b[2]-r>=self.h+p["cup_floor"]-.001 and b[2]+r<self.h+p["cup_height"])
        self.result["initial_goal_false"]=not any(inside("ball"+s,"cup"+s) for s in ("L","R"))
        if not self.result["initial_goal_false"]:raise ValueError("invalid trial: content already in receiver")
        for i,s in enumerate(("L","R")):
            can,ball,cup="can"+s,"ball"+s,"cup"+s
            self.pick(i,can,self.pos(can)[2]+.015)
            xy=self.env.get_body_pos(cup)[:2]+p["can_offset_xy"]
            self.transport(i,can,xy,z=self.h+.29)
            sk=self.skills[i]
            bid=self.env.obj_bid[can];rot=self.env.d.xmat[bid].reshape(3,3)
            local=rot.T@(self.pos(ball)-self.pos(can))
            local[2]+=p.get("handle_height",p["can_height"]/2)  # floor-relative coordinate
            local[0]+=p.get("handle_offset",0.0)  # handle origin -> vessel center
            contained=bool(np.linalg.norm(local[:2])+p["ball_radius"]<p["can_inner_radius"]+.001 and
                           p["can_floor"]-.001<=local[2]-p["ball_radius"] and local[2]+p["ball_radius"]<p["can_height"]+.002)
            self.mark("before_tip",content_in_source=contained,ball=ball,local_ball=local.tolist())
            if not contained or inside(ball,cup):raise RuntimeError("content not in source immediately before tilting")
            sk.tip_pour_joint(axis=p["tilt_axis"],total_angle=p["tilt_radians"],n=90,hold=100,can_name=can)
            ok=inside(ball,cup);self.mark("poured",ball=ball,inside=ok)
            if not ok:raise RuntimeError(f"{ball} missed receiver; no result correction")
            # Scope ends with the cans held tilted. Contents must remain contained
            # through the second arm's motion and final stability window.
        self.result["success"]=self.stable(lambda:all(inside("ball"+s,"cup"+s) for s in ("L","R")))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("task",choices=("relay","packing","pour"))
    ap.add_argument("--config",type=Path,default=ROOT/"scenes/feasible_dual_v1.json")
    ap.add_argument("--output",type=Path,required=True);ap.add_argument("--seed",type=int,default=0)
    a=ap.parse_args();c=json.loads(a.config.read_text());Curriculum(c,a.task,a.output,a.seed).run()
if __name__=="__main__":main()

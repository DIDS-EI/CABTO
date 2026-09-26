"""Five CABTO MuJoCo reconstructions: initial geometry, visibility and pose-IK audit.

Candidate layouts are opt-in exported XML, never applied to legacy generators.
No task execution, object snapping during execution, model calls or success claims.
IK is a necessary pose test only, not collision-free path or controller validation.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import itertools
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
TASK_MODULES = {"cover": "scene_cover", "blocks": "scene_stack", "pour": "scene_pour",
                "handover": "scene_handover", "storage": "scene_storage"}
CANDIDATE_XY = {
    "pour": {"canL": (0.29, 0.20), "ballL": (0.29, 0.20),
             "canR": (0.29, -0.20), "ballR": (0.29, -0.20),
             "cupL": (0.22, 0.0), "cupR": (0.38, 0.0)},
    "storage": {"carton": (0.30, 0.0), "item_g1": (0.27, 0.19),
                "item_g2": (0.27, -0.19), "shelf": (0.59, 0.0)},
}


def task_bodies(xml):
    root = ET.fromstring(xml)
    bodies = []
    for body in root.findall("./worldbody/body"):
        name = body.get("name", "")
        if name in ("table", "plate") or name.startswith("link"):
            continue
        if body.find("geom") is not None and not body.findall("body"):
            bodies.append(name)
    return bodies


def set_home(m, d):
    home = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    for suffix in ("", "_L", "_R"):
        for i, value in enumerate(home, 1):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}{suffix}")
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}{suffix}")
            if jid >= 0:
                d.qpos[m.jnt_qposadr[jid]] = value
            if aid >= 0:
                d.ctrl[aid] = value
        for i in (1, 2):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"finger_joint{i}{suffix}")
            if jid >= 0:
                d.qpos[m.jnt_qposadr[jid]] = 0.04
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator8{suffix}")
        if aid >= 0:
            d.ctrl[aid] = 255.0
    mujoco.mj_forward(m, d)


def geom_corners(m, d, body):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    points = []
    for gid in range(m.ngeom):
        if m.geom_bodyid[gid] != bid:
            continue
        size = m.geom_size[gid]
        typ = m.geom_type[gid]
        if typ == mujoco.mjtGeom.mjGEOM_BOX:
            ext = size
        elif typ == mujoco.mjtGeom.mjGEOM_SPHERE:
            ext = np.repeat(size[0], 3)
        elif typ == mujoco.mjtGeom.mjGEOM_CYLINDER:
            ext = np.array([size[0], size[0], size[1]])
        elif typ == mujoco.mjtGeom.mjGEOM_CAPSULE:
            ext = np.array([size[0], size[0], size[0] + size[1]])
        else:
            continue
        rot = d.geom_xmat[gid].reshape(3, 3)
        points.extend(d.geom_xpos[gid] + np.asarray(sign) * ext @ rot.T
                      for sign in itertools.product((-1, 1), repeat=3))
    return np.asarray(points)


def camera_params(pos, target):
    z = np.asarray(pos) - target
    z = z / np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.concatenate([x, y])


def fit_task_cameras(xml, bodies):
    root = ET.fromstring(xml)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    set_home(m, d)
    points = np.vstack([geom_corners(m, d, b) for b in bodies])
    # Reserve headroom for lifted objects; this is framing, not a task guarantee.
    low, high = points.min(axis=0), points.max(axis=0)
    high[2] += 0.15
    target = (low + high) / 2
    radius = np.linalg.norm(high - low) / 2
    fovy = 44.0
    dist = radius / math.sin(math.radians(fovy / 2)) * 1.20
    for name, direction in (("overview", [0.70, -0.85, 0.90]), ("front", [1.0, 0.0, 0.65])):
        direction = np.asarray(direction, float)
        pos = target + dist * direction / np.linalg.norm(direction)
        cam = root.find(f"./worldbody/camera[@name='{name}']")
        if cam is not None:
            cam.set("pos", " ".join(map(str, pos)))
            cam.set("xyaxes", " ".join(map(str, camera_params(pos, target))))
            cam.set("fovy", str(fovy))
    return ET.tostring(root, encoding="unicode")


def candidate_xml(xml, task):
    root = ET.fromstring(xml)
    for body, xy in CANDIDATE_XY.get(task, {}).items():
        node = root.find(f"./worldbody/body[@name='{body}']")
        pos = list(map(float, node.get("pos").split()))
        node.set("pos", f"{xy[0]} {xy[1]} {pos[2]}")
    if task == "pour":
        # Place initial geometry without penetration: can bottom at table+2mm;
        # content sphere bottom at can inner floor+2mm. Initialization only.
        for can, ball in (("canL", "ballL"), ("canR", "ballR")):
            node = root.find(f"./worldbody/body[@name='{can}']")
            pos = list(map(float, node.get("pos").split()))
            base = node.find("geom[@type='cylinder']")
            base_pos = list(map(float, base.get("pos").split()))
            half_height = float(base.get("size").split()[1])
            pos[2] = 0.40 + 0.002 - (base_pos[2] - half_height)
            node.set("pos", " ".join(map(str, pos)))
            ball_node = root.find(f"./worldbody/body[@name='{ball}']")
            radius = float(ball_node.find("geom").get("size"))
            ball_node.set("pos", f"{pos[0]} {pos[1]} {pos[2]+base_pos[2]+half_height+radius+0.002}")
    out = ET.tostring(root, encoding="unicode")
    return fit_task_cameras(out, task_bodies(out))


def penetrations(m, d, bodies):
    rows = []
    for c in d.contact:
        b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[c.geom1]))
        b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[c.geom2]))
        if (b1 in bodies or b2 in bodies) and b1 != b2 and c.dist < -0.001:
            rows.append({"bodies": [b1, b2], "penetration_mm": float(-1000*c.dist)})
    return sorted(rows, key=lambda r: -r["penetration_mm"])


def visibility(m, d, bodies, camera="overview", size=640):
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    rot = d.cam_xmat[cid].reshape(3, 3)
    focal = size / 2 / math.tan(math.radians(m.cam_fovy[cid]/2))
    rows = {}
    for name in bodies:
        points = geom_corners(m, d, name)
        local = (points - d.cam_xpos[cid]) @ rot
        depth = -local[:, 2]
        uv = np.column_stack([local[:, 0] / depth*focal+size/2,
                              -local[:, 1] / depth*focal+size/2])
        lo, hi = uv.min(axis=0), uv.max(axis=0)
        rows[name] = {"bbox": [*lo.tolist(), *hi.tolist()],
                      "in_frame": bool(np.all(depth>0) and np.all(lo>=0) and np.all(hi<size)),
                      "note": "bounding projection only; occlusion not assessed"}
    return rows


def pose_ik(xml, task, bodies):
    from exp4_env import Exp4Env
    dual = task in ("pour", "handover", "storage")
    spec = [("_L", -math.pi/2), ("_R", math.pi/2)] if dual else [("", 0.0)]
    root = ET.fromstring(xml)
    free = [b.get("name") for b in root.findall("./worldbody/body") if b.find("freejoint") is not None]
    env = Exp4Env(xml, spec, free, render=False)
    set_home(env.m, env.d)
    rows = []
    for name in bodies:
        if name.startswith("ball"):
            continue
        corners = geom_corners(env.m, env.d, name)
        low, high = corners.min(axis=0), corners.max(axis=0)
        target = (low + high) / 2
        target[2] = high[2] + 0.04
        for arm in env.arms:
            q = arm.solve_ik(target, arm.grasp_quat, restarts=4, iters=70)
            for adr, v in zip(arm.arm_qadr, q):
                env.ik_data.qpos[adr] = v
            _, _, pe, re = arm._err_and_jac(env.ik_data, target, arm.grasp_quat, 0.5)
            rows.append({"object": name, "arm": arm.s or "single", "target": target.tolist(),
                         "position_error_mm": pe*1000, "rotation_error_rad": re,
                         "pose_ik_pass": bool(pe<0.01 and re<0.05),
                         "note": "limited-seed IK; failure is not an infeasibility proof; no path/collision test"})
    return rows


def audit(xml, task, variant, out, run_ik):
    bodies = task_bodies(xml)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    set_home(m, d)
    before = penetrations(m, d, bodies)
    positions = {b: d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,b)].copy() for b in bodies}
    image_paths = {}
    renderer = mujoco.Renderer(m, 640, 640)
    try:
        for cam in ("overview", "front"):
            renderer.update_scene(d, camera=cam)
            path = out / f"{task}_{variant}_{cam}.png"
            image = np.asarray(renderer.render()).copy()
            if image.max() == image.min():
                raise RuntimeError("uniform render; image-based audit invalid")
            Image.fromarray(image).save(path)
            image_paths[cam] = path.name
        frames = visibility(m, d, bodies)
        # Only simulator stepping: no grasp_assist, no object-state corrections.
        for _ in range(500):
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        drift = {b: float(np.linalg.norm(d.xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,b)][:2]-p[:2])*1000)
                 for b, p in positions.items()}
        after = penetrations(m, d, bodies)
        renderer.update_scene(d, camera="overview")
        Image.fromarray(np.asarray(renderer.render()).copy()).save(out / f"{task}_{variant}_settled.png")
    finally:
        renderer.close()
    record = {"task":task,"variant":variant,"xml_sha256":hashlib.sha256(xml.encode()).hexdigest(),
              "bodies":bodies,"initial_penetrations_over_1mm":before,
              "settled_penetrations_over_1mm":after,"settle_seconds":500*m.opt.timestep,
              "horizontal_drift_mm":drift,"visibility":frames,"images":image_paths,
              "task_execution_tested":False,"pose_ik":pose_ik(xml,task,bodies) if run_ik else []}
    print(task, variant, "initial overlaps",len(before),"max drift mm",round(max(drift.values()),2),
          "out of frame",[b for b,v in frames.items() if not v["in_frame"]],flush=True)
    return record


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cabto-root", type=Path, default=ROOT.parent/"CABTO"/"exp4_bt_tasks")
    ap.add_argument("--output",type=Path,default=ROOT/"outputs"/"cabto_reproduction_audit")
    ap.add_argument("--ik",action="store_true")
    args=ap.parse_args()
    sys.path.insert(0,str(args.cabto_root/"stage2_scripted"))
    args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for task,module in TASK_MODULES.items():
        legacy=importlib.import_module(module).build_scene_xml()
        for variant, xml in (("legacy",legacy),("candidate",candidate_xml(legacy,task))):
            (args.output/f"{task}_{variant}.xml").write_text(xml,encoding="utf-8")
            records.append(audit(xml,task,variant,args.output,args.ik))
    result={"kind":"preflight_not_task_success","mujoco":mujoco.__version__,"records":records,
            "caveats":["1s settling does not establish task feasibility", "pose IK is not path planning",
                       "frustum inclusion is not visibility under occlusion", "legacy assisted executors not run"]}
    (args.output/"scene_preflight.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    sheet=Image.new("RGB",(1280,5*400),(24,27,33)); draw=ImageDraw.Draw(sheet)
    for i,task in enumerate(TASK_MODULES):
        for j,variant in enumerate(("legacy","candidate")):
            image=Image.open(args.output/f"{task}_{variant}_overview.png"); image.thumbnail((620,360))
            sheet.paste(image,(j*640+(640-image.width)//2,i*400+28))
            draw.text((j*640+20,i*400+8),f"{task} | {variant} | initial scene only",fill=(230,230,240))
    sheet.save(args.output/"five_scenes_comparison.png")

if __name__ == "__main__":
    main()

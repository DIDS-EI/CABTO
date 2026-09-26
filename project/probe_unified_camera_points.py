"""Matched-state old/global/wrist Qwen pointing diagnostics, NOT robot commands.
Pixel error uses evaluation-only reference projection; millimeter plane error uses
true reference z ONLY to quantify a conditional geometric error, not deployable 3D.
No per-view ground truth selection or correction is fed to Qwen or robot control.
"""
from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image,ImageDraw
from core.cover_cabto_grounding import LocalModel,parse_json,save,digest
from core.cover_cabto_api import Perception
from core.unified_camera_rig import project,intersect_plane

PROBES={
"storage":[(0,"item1","the center of the small green hinged toy brick with two large round studs outside the open cardboard box"),
           (2,"item1","the center of the small green hinged toy brick with two large round studs outside the open cardboard box"),
           (0,"box_floor","the center of the flat interior floor of the open cardboard box, not the walls or flaps"),
           (4,"box_floor","the center of the flat interior floor of the open cardboard box, not the walls or flaps")],
"pour":[(4,"basin","the middle of the brown handle attached to the white basin, where the gripper should hold it"),
         (8,"basin_floor","the center of the interior floor of the white receiving basin, not its rim or wall")],
"handover":[(5,"grasp1","the upper edge of the green tea box under the approaching OPEN gripper, at that gripper's grasp location; not the end already held by the other gripper or a display box")]
}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--runs",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--tasks",nargs="+",default=["storage","pour","handover"]);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    client=LocalModel("mlx-community/Qwen2.5-VL-3B-Instruct-4bit");rows=[]
    for task in a.tasks:
        resultpath=a.runs/task/"result.json";result=json.loads(resultpath.read_text())
        for index,target_name,description in PROBES[task]:
            cap=result["captures"][index];gt=next(t for t in cap["evaluation_only_targets"] if t["name"]==target_name)
            for view in cap["views"]:
                name=view["camera"];out=a.output/task/f"frame{index:03d}_{target_name}_{name}.json"
                if out.is_file():record=json.loads(out.read_text());rows.append(record);continue
                prompt=(f"Point to {description}. Output the absolute pixel coordinate as JSON {{\"point\":[x,y]}}. "
                        "If that exact target is not visible or you cannot identify it, output only {\"visible\":false}. Do not point to a robot link.")
                record={"task":task,"capture_index":index,"event":cap["event"],"target":target_name,"camera":name,
                    "input_image":view["image"],"input_sha256":digest(view["image"]),"original_run":str(resultpath),"original_run_sha256":digest(resultpath),
                    "original_physical_state_time":cap["simulation_time"],"used_by_robot":False,"type":"diagnostic_point_not_executable_grasp",
                    "calibration":view["calibration"],"evaluation_only_reference":gt,"reference_semantics":gt["semantic"]}
                record["call"]=client.call(prompt,[Path(view["image"])],max_tokens=220)
                try:
                    raw=parse_json(record["call"]["raw"])
                    if raw.get("visible") is False:record["status"]="model_abstained"
                    else:
                        uv,_=Perception.parse_pixel(record["call"],720);record.update(point=uv.tolist(),status="point_returned_not_validated")
                        true_uv=project(view["calibration"],gt["xyz"])
                        diag=next(x for x in view["evaluation_only_visibility"] if x["target"]==target_name)
                        record["reference_visibility"]=diag
                        if true_uv is not None:
                            record["evaluation_only_reference_uv"]=true_uv.tolist()
                            record["pixel_error"]=float(np.linalg.norm(uv-true_uv))
                            if diag["in_frame"]:
                                pred=intersect_plane(view["calibration"],uv,gt["xyz"][2])
                                record["conditional_true_plane_xy_error_mm"]=float(np.linalg.norm(pred[:2]-gt["xyz"][:2])*1000)
                except Exception as exc:record.update(status="rejected_output",error=str(exc))
                image=Image.open(view["image"]).convert("RGB");d=ImageDraw.Draw(image)
                if "point" in record:
                    u,v=record["point"];d.line((u-13,v,u+13,v),fill=(255,150,68),width=3);d.line((u,v-13,u,v+13),fill=(255,150,68),width=3)
                if "evaluation_only_reference_uv" in record:
                    u,v=record["evaluation_only_reference_uv"];d.ellipse((u-7,v-7,u+7,v+7),outline=(82,204,236),width=2)
                d.rectangle((8,8,650,37),fill=(22,30,33));d.text((16,17),"DIAGNOSTIC ONLY | Orange: Qwen | Cyan: evaluation reference",fill=(240,221,185))
                overlay=out.with_suffix(".png");overlay.parent.mkdir(parents=True,exist_ok=True);image.save(overlay);record["overlay"]=str(overlay)
                save(out,record);rows.append(record)
                print("VIEW_POINT",task,index,target_name,name,record["status"],round(record.get("pixel_error",-1),2),flush=True)
    summary={"diagnostic_only":True,"used_for_control":False,"point_results":rows,
       "note":"Conditional true-plane errors use evaluation GT z; not deployment estimates. Pixel comparison is meaningful only when prompt feature matches reference and reference is visible. Body ray visibility alone does not prove precise feature visibility."}
    save(a.output/"results.json",summary)

if __name__=="__main__":main()

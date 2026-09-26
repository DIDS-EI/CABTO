"""Camera gallery evidence. Existing online Cover/Blocks records are read-only.
Dual scenes: NEW real Qwen calls on saved RGB keyframes, never used to drive the
already completed Oracle rollouts. No simulation, GT points, or success updates.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil
from PIL import Image,ImageDraw
from core.cover_cabto_grounding import LocalModel
from core.cover_cabto_api import Perception

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/remaining_cabto"
GALLERY=OUT/"camera_pointing"
MODEL="mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
NAMES={"bowl":"碗内放置点","board":"砧板放置点","pan":"锅内放置点","shrimp":"小虾抓取点","apple":"苹果抓取点","potato":"土豆抓取点","green":"绿色支撑积木","blue":"蓝色支撑积木","red":"红色待叠积木","yellow":"黄色待叠积木"}
# These are descriptions of visible image features, NOT robot grasp targets.
PROBES={
 "pour":[
  ("source_cup",0,"源杯区域 · 抓杯前","the red source cup on the table"),
  ("basin_handle",1,"接收盆把手 · 抓盆前","the brown handle attached to the white basin"),
  ("basin_interior",3,"盆内区域 · 倾倒前","the visible interior center of the white receiving basin")],
 "handover":[
  ("active_box",0,"活动茶盒 · 交出臂抓取前","the green tea box standing on the green rectangular source marker in the middle of the table, not the other display boxes"),
  ("receiver_region",2,"空中茶盒 · 接收前","the center of the green tea box held in the air near the middle of the image, immediately next to the right robot gripper"),
  ("held_box",5,"接收臂持盒 · 放置前","the center of the green tea box held by a robot gripper, not any tea box resting on the table")],
 "storage":[
  ("green_item",0,"绿色积木 · 装箱前","the small bright green block outside the cardboard box, near the front of the table"),
  ("second_item",3,"另一件积木 · 抓取前","the small yellow block with green studs outside the cardboard box near its rear edge"),
  ("box_interior",1,"纸箱内部 · 放入前","the visible interior center of the open cardboard box"),
  ("shelf",8,"货架承载面 · 搬箱前","the exposed dark brown top surface of the low shelf on the right side of the table")]
}

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return os.path.relpath(p,OUT)
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2))

def build(run_calls=False):
    GALLERY.mkdir(parents=True,exist_ok=True)
    manifest={"schema_version":1,"display_only":True,"original_execution_data_modified":False,"scenes":{}}
    for task,path in [("cover",ROOT/"outputs/cover_cabto_fix/final_visual/C3/perception/perception.json"),
                      ("blocks",OUT/"blocks/final_seed0/B3/perception/perception.json")]:
        records=json.loads(path.read_text());items=[]
        for i,row in enumerate(records):
            if "call" not in row:continue
            image=Path(row["call"]["image_files"][0]);overlay=image.with_name(image.stem+"_overlay.png")
            assert image.is_file() and overlay.is_file()
            coarse=row.get("coarse_uv",row.get("refinement",{}).get("coarse_uv"))
            name=row["object"]
            p=GALLERY/task/f"record_{i:03d}_{name}.json"
            evidence={"origin":"existing_closed_loop_call","used_in_control":True,"source_json":str(path),"source_sha256":digest(path),"record_index":i,"record":row}
            write(p,evidence)
            if task=="cover":
                local_image=p.parent/image.name;local_overlay=p.parent/overlay.name
                shutil.copy2(image,local_image);shutil.copy2(overlay,local_overlay)
                image,overlay=local_image,local_overlay
            items.append({"id":f"{task}-{i}","title":NAMES[name],"object":name,"input":rel(image),"overlay":rel(overlay),"record":rel(p),
                 "input_sha256":digest(image),"overlay_sha256":digest(overlay),"prompt":row["call"]["text"],"raw":row["call"]["raw"],"model":row["call"]["model_id"],
                 "camera":"front（标定俯视）","image_size":list(Image.open(image).size),"point":coarse,"refined_point":row.get("uv"),"xyz":row.get("xyz"),
                 "status":"来自已完成执行的实际感知调用","kind":"online","used_in_control":True,"coordinate_contract":"absolute_pixels",
                 "note":"黄色十字：Qwen 粗点；绿色十字：RGB 几何细化。已用于该次控制；不是真值标注。"})
        source_copy=GALLERY/task/"source_perception.json";shutil.copy2(path,source_copy)
        manifest["scenes"][task]={"items":items,"source_record":rel(source_copy),"online_calls":len(items),"cache_reads":sum("call" not in r for r in records),"diagnostic_calls":0}
    client=None
    for task,probes in PROBES.items():
        items=[]
        for name,step,title,description in probes:
            image=OUT/task/"final_seed0"/task.upper()/f"step{step:02d}"/"before.png"
            result=OUT/task/"final_seed0"/task.upper()/"result.json"
            trace=json.loads(result.read_text())["steps"][step]
            recordpath=GALLERY/task/f"{name}.json";overlay=GALLERY/task/f"{name}_overlay.png"
            if recordpath.exists():
                record=json.loads(recordpath.read_text());assert record["source_image_sha256"]==digest(image)
            else:
                if not run_calls:raise RuntimeError(f"Missing supplemental call {recordpath}; run once with --run-dual-probes")
                if client is None:client=LocalModel(MODEL)
                text=f"Point to {description}. Output the pixel coordinate."
                record={"origin":"new_posthoc_keyframe_Qwen_diagnostic","used_in_control":False,"source_image":str(image),"source_image_sha256":digest(image),
                   "source_episode_result":str(result),"source_episode_sha256":digest(result),"source_step":step,"action_model":trace["model"],
                   "original_execution_perception":"oracle_geometry","camera":"overview","coordinate_contract":"absolute_pixels",
                   "scope":"semantic region point on a recorded RGB frame, NOT calibrated robot grasp or placement command; no motion or GT evaluation"}
                try:
                    record["call"]=client.call(text,[image],max_tokens=220)
                    width,height=Image.open(image).size
                    point,field=Perception.parse_pixel(record["call"],max(width,height))
                    if not(0<=point[0]<width and 0<=point[1]<height):raise ValueError("Point outside actual image bounds")
                    record.update(parse_status="valid_2d_point",point=point.tolist(),field=field,image_size=[width,height])
                except Exception as exc:record.update(parse_status="rejected",error=str(exc))
                write(recordpath,record)
                print("POSTHOC_POINT",task,name,record["parse_status"],record.get("point"),flush=True)
            # Re-render only display overlays; leave saved inputs and rollouts intact.
            annotated=Image.open(image).convert("RGB");draw=ImageDraw.Draw(annotated)
            if record.get("point"):
                x,y=record["point"];draw.ellipse((x-13,y-13,x+13,y+13),outline=(18,24,29),width=5)
                draw.ellipse((x-12,y-12,x+12,y+12),outline=(242,162,78),width=3)
                draw.line((x-18,y,x+18,y),fill=(242,162,78),width=3);draw.line((x,y-18,x,y+18),fill=(242,162,78),width=3)
            draw.rectangle((8,8,min(annotated.width-8,400),34),fill=(26,31,34))
            draw.text((16,15),"QWEN KEYFRAME DIAGNOSTIC / NOT USED BY CONTROL",fill=(250,210,146))
            annotated.save(overlay)
            items.append({"id":f"{task}-{name}","title":title,"object":name,"input":rel(image),"overlay":rel(overlay),"record":rel(recordpath),
                "input_sha256":digest(image),"overlay_sha256":digest(overlay),"prompt":record.get("call",{}).get("text",""),"raw":record.get("call",{}).get("raw",""),"model":MODEL,
                "camera":"overview（原执行关键帧）","image_size":record.get("image_size",list(Image.open(image).size)),"point":record.get("point"),"refined_point":None,"xyz":None,
                "status":"新增图片打点测试 · 未用于原控制","kind":"posthoc","used_in_control":False,"parse_status":record["parse_status"],
                "action_name":trace["model"]["name"],"original_target_read_calls":[v for v in trace["trace"] if v["method"] in ("target_pose","grasp_pose","source_pose","receiver_pose","target_handles","target_tray_pose")],
                "note":"橙色准星：本次真实 Qwen 对旧相机图的语义打点；位置尚未做抓取精度/三维标定验收。原成功执行仍为 Oracle 定位，不能据此认定视觉闭环通过。"})
        manifest["scenes"][task]={"items":items,"online_calls":0,"cache_reads":0,"diagnostic_calls":len(items),"original_execution_perception":"oracle_geometry"}
    manifest["image_count"]=sum(len(v["items"]) for v in manifest["scenes"].values())
    manifest["online_call_images"]=sum(v["online_calls"] for v in manifest["scenes"].values())
    manifest["posthoc_call_images"]=sum(v["diagnostic_calls"] for v in manifest["scenes"].values())
    write(GALLERY/"manifest.json",manifest);print("GALLERY",manifest["image_count"],flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--run-dual-probes",action="store_true");a=ap.parse_args();build(a.run_dual_probes)

"""Render a reusable same-time wrist/global camera panel for the five-scene index."""
from pathlib import Path
import json,html,shutil,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT/"outputs/unified_cameras/selected_v3"
SOURCE=ROOT/"outputs/unified_cameras";OUT=ROOT/"outputs/remaining_cabto/unified_cameras"
TASKS=["storage","pour","handover","cover","blocks"]
CAM_LABELS={"overview":"原全景 · 仅对照","rig_global":"统一全局俯视","rig_wrist_left":"左腕相机 · hand_L","rig_wrist_right":"右腕相机 · hand_R","rig_wrist_single":"腕部相机 · hand"}

def esc(x):return html.escape(str(x))
def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.is_file() or digest(src)!=digest(dst):shutil.copy2(src,dst)
    return str(dst.relative_to(OUT.parent))

def build():
    OUT.mkdir(parents=True,exist_ok=True);data={"profile":read(ROOT/"project/scenes/camera_rig_unified_v3.json"),"tasks":{},"diagnostic_only":True}
    for task in TASKS:
        run=read(RUNS/task/"result.json");cams=run["camera_rig"]["cameras"];frames=[];mount_error=0;roundtrip=0
        copy(RUNS/task/"rig.json",OUT/task/"rig.json");copy(RUNS/task/"scene_with_cameras.xml",OUT/task/"scene_with_cameras.xml")
        copy(RUNS/task/"result.json",OUT/task/"result.json")
        for c in run["captures"]:
            frame={"index":c["index"],"event":c["event"],"time":c["simulation_time"],"action":None if c["action"] is None else c["action"]["name"],"views":[]}
            for view in c["views"]:
                src=Path(view["image"]);dst=OUT/task/"frames"/src.parent.name/src.name
                display=copy(src,dst);cal=view["calibration"]
                frame["views"].append({"camera":view["camera"],"label":CAM_LABELS[view["camera"]],"image":display,
                     "calibration":cal,"evaluation_only_visibility":view["evaluation_only_visibility"]})
                roundtrip=max(roundtrip,view["max_project_plane_roundtrip_m"])
                if view["camera"].startswith("rig_wrist"):
                    expected=next(m for m in run["camera_rig"]["wrist_mounts"] if m["name"]==view["camera"])["T_hand_camera"]
                    actual=cal["measured_T_body_camera"]
                    mount_error=max(mount_error,float(np.max(np.abs(np.array(actual["position"])-expected["position"]))),float(np.max(np.abs(np.array(actual["rotation"])-expected["rotation"])) ))
            frame["record"]=copy(Path(c["views"][0]["image"]).parent/"capture.json",OUT/task/"frames"/Path(c["views"][0]["image"]).parent.name/"capture.json")
            frames.append(frame)
        moves={}
        for cam in cams:
            positions=[np.array(next(v for v in f["views"] if v["camera"]==cam)["calibration"]["T_world_camera"]["position"]) for f in frames]
            moves[cam]=max(float(np.linalg.norm(p-positions[0])) for p in positions)
        data["tasks"][task]={"success":run["success"],"actions":len(run["steps"]),"all_transition_checks":all(s["effect_ok"] for s in run["steps"]),
            "camera_names":cams,"frames":frames,"physics_unchanged":run["camera_rig"]["physics_validation"]["matched"],"same_local_mount_max_error":mount_error,
            "max_roundtrip_m":roundtrip,"max_translation_from_initial_m":moves,"perception_used_for_motion":run["perception_used_for_motion"],
            "new_cameras_used_for_control":False,"source_hashes":run["source_hashes"],"sources_unchanged_during_run":run["sources_unchanged"]}
    pointpath=SOURCE/"pointing_v3/results.json";pointgroups={}
    if pointpath.is_file():
        for row in read(pointpath)["point_results"]:
            task=row["task"];group=f'{row["capture_index"]:03d}_{row["target"]}'
            record={**row,"display_input":copy(Path(row["input_image"]),OUT/"pointing"/task/f'{group}_{row["camera"]}_input.png'),
                    "display_overlay":copy(Path(row["overlay"]),OUT/"pointing"/task/f'{group}_{row["camera"]}_overlay.png')}
            pointgroups.setdefault(task,{}).setdefault(group,[]).append(record)
        copy(pointpath,OUT/"pointing_results.json")
    data["pointing"]=pointgroups
    data["mount_comparison"]=[]
    for version,base in [("v1",SOURCE/"final"),("v2",SOURCE/"mount_v2"),("v3",SOURCE/"mount_v3")]:
        p=base/"handover/frames/005_after_approach/rig_wrist_right.png"
        if p.is_file():data["mount_comparison"].append({"version":version,"image":copy(p,OUT/"mount_compare"/(version+".png"))})
    data["success_count"]=sum(r["success"] for r in data["tasks"].values());data["frame_count"]=sum(len(r["frames"]) for r in data["tasks"].values())
    data["image_count"]=sum(len(f["views"]) for r in data["tasks"].values() for f in r["frames"])
    copy(ROOT/"project/scenes/camera_rig_unified_v3.json",OUT/"camera_rig_unified_v3.json")
    if (SOURCE/"reproduce.txt").exists():copy(SOURCE/"reproduce.txt",OUT/"reproduce.txt")
    (OUT/"manifest.json").write_text(json.dumps(data,ensure_ascii=False,indent=2))
    return data


def load():return read(OUT/"manifest.json") if (OUT/"manifest.json").is_file() else None


def overview_html(data):
    if not data:return ""
    mounts="".join(f'<figure><img loading="lazy" src="{r["image"]}" alt="{r["version"]}腕部安装对照"><figcaption>{r["version"]} · '+{"v1":"手侧前置：盒面偏侧","v2":"镜头后移：腕壳明显遮挡","v3":"夹爪侧置：更易见盒侧面，最终采用"}[r["version"]]+"</figcaption></figure>" for r in data["mount_comparison"])
    return f'''<section id="unified-camera-rig"><div class="eyebrow">新增 / 统一相机方案</div><h2>每只手腕一台相机 ＋ 一台全局俯视</h2><p>五个场景共用同一安装配置。双臂是 <code>rig_wrist_left / rig_wrist_right / rig_global</code>；单臂是 <code>rig_wrist_single / rig_global</code>。原overview保留为视频与同帧对照，不冒充腕部相机。</p><div class="notice"><strong>这些是真正挂在MuJoCo手部body上的移动相机，不是裁剪全景。</strong>所有镜头采用固定局部外参，没有根据物体真值自动追踪。新图像仍是诊断输入；这次5场景的相机兼容性运行沿用Oracle控制，不计作新视觉闭环成功。</div><div class="scroll"><table><tr><th>统一项目</th><th>实际配置</th></tr><tr><td>腕部安装</td><td>hand局部 [0, −100, +25] mm；光轴指向局部 [0, 0, +160] mm，固定朝向</td></tr><tr><td>视场角 / 分辨率</td><td>腕部75°；全局60°；均720×720</td></tr><tr><td>全局安装</td><td>桌面中心沿世界X偏移80mm，桌面上方1.20m，垂直俯视；各场景同一规则</td></tr><tr><td>兼容性验收</td><td>{data['success_count']}/5原程序完成；{data['frame_count']}组同步关键帧、{data['image_count']}张图；物理参数未变</td></tr></table></div><p>腕部近景能放大积木与抓持区，但手持物体、另一条臂仍会遮挡；因此全局相机保留，不能仅依赖腕部。相机只建模光学位姿，尚未加真实镜头外壳的质量、碰撞体或电缆。</p><p><strong>打点还未达到可执行验收：</strong>本轮同帧同提示共28次Qwen诊断，18次返回点、10次选择不输出；返回点中仍有同色积木身份混淆。新视角放大了目标，但不能把“返回二维点”当作定位正确，更不能直接升级为可执行抓取点。</p><details><summary>为什么选第三版腕部安装？查看同阶段三版实拍</summary><div class="rig-mounts">{mounts}</div></details><div class="links"><a href="#storage-rig">先看Storage新视角 ↓</a><a href="unified_cameras/camera_rig_unified_v3.json">统一配置JSON</a><a href="unified_cameras/manifest.json">全部帧与动态标定</a><a href="unified_cameras/verification.json">相机验收</a><a href="unified_cameras/reproduce.txt">启用与复跑说明</a></div></section>'''


def scene_html(task,data):
    if not data:return ""
    r=data["tasks"][task];frames=r["frames"]
    selected=next((f for f in frames if f["event"] in ("after_align_xy","after_approach")),frames[0]);index=frames.index(selected)
    options="".join(f'<option value="{i}" '+("selected " if i==index else "")+f'>{f["index"]:02d} · {esc(f["event"])} · {f["time"]:.2f}s · {esc(f["action"] or "初始/结束")}</option>' for i,f in enumerate(frames))
    order=["rig_global"]+[n for n in r["camera_names"] if n.startswith("rig_wrist")]+["overview"]
    cards=""
    for cam in order:
        v=next(v for v in selected["views"] if v["camera"]==cam)
        cards+=f'<figure class="rig-view" data-camera="{cam}"><figcaption>{CAM_LABELS[cam]}</figcaption><a href="{v["image"]}" target="_blank" rel="noopener"><img loading="lazy" src="{v["image"]}" alt="{task} {CAM_LABELS[cam]}真实相机"></a><small class="rig-position">相机随本体运动；点击图片查看原图</small></figure>'
    encoded=esc(json.dumps(frames,ensure_ascii=False))
    paragraph={"storage":"优先检查积木抓取前近景和箱内槽位：全局提供箱体/积木布局，腕部在接近后放大目标；物体被夹住后仍可能遮住腕部视野。槽位本身无视觉标记，需由箱体位姿与明确的槽位局部偏移确定，不应把箱内任意点称为准确槽位。",
        "pour":"看抓盆把手时的腕部近景，以及对齐/倾倒阶段的全局。两条臂仍可能挡住盆内，腕部相机也不保证每阶段都看到盆底；旧的盆壁诊断点不能直接升级为接收点。",
        "handover":"看接收臂接近后的右腕图：侧置镜头较前两版更容易看到茶盒侧面，但指定接持区、双指接触和高度仍需单独标定。两只手贴近时遮挡仍存在。",
        "cover":"仅验证同一相机安装规则能适配单臂与旧动作程序；本次没有替换此前已通过的Cover视觉定位算法。",
        "blocks":"仅验证同一腕部/全局规则适配积木场景；本次新镜头不作为原Blocks视觉成绩的输入，旧俯视视觉试验保留。"}[task]
    htmlpart=f'''<div class="rig-panel" id="{task}-rig" data-rig="{task}" data-frames="{encoded}"><h3>新增统一腕部／全局相机 · 同时刻对照</h3><p>{paragraph}</p><label>选择实际执行阶段 <select class="rig-frame-select" aria-label="{task}相机阶段">{options}</select></label><p class="rig-time">同一仿真时刻 {selected['time']:.3f}s；各图之间没有推进仿真。新相机输出仅诊断，未用于控制。</p><div class="rig-view-grid">{cards}</div><p><a class="rig-capture-link" href="{selected['record']}">本帧标定/可见性记录</a> · <a href="unified_cameras/{task}/rig.json">安装外参</a> · <a href="unified_cameras/{task}/scene_with_cameras.xml">含真实腕部相机的MJCF</a></p></div>'''
    points=data.get("pointing",{}).get(task,{})
    if points:
        htmlpart+='<details class="rig-point-diagnostics"><summary>新旧视角真实Qwen打点对照（诊断点，不可直接执行）</summary><p>每组使用同一物理时刻、同一提示，输入仅含RGB。橙色为模型点，青色为仅评估用参考投影，未用于纠正预测。像素误差只对相同语义特征有效；参考遮挡或特征不一致时不能据此判断可抓取。</p>'
        for group,rows in points.items():
            first=rows[0];htmlpart+=f'<h4>{esc(first["target"])} · 阶段{first["capture_index"]:02d}</h4><div class="rig-view-grid">'
            for p in rows:
                error="未返回点" if "pixel_error" not in p else f'{p["pixel_error"]:.1f} px'
                vis=p.get("reference_visibility",{});quality="参考在视野且射线命中目标本体" if vis.get("ray_hits_target_body") else "参考可能遮挡／不在视野"
                if p["status"]=="model_abstained":quality="模型拒答；拒答本身不证明目标不可见"
                htmlpart+=f'<figure><figcaption>{CAM_LABELS[p["camera"]]}</figcaption><a href="{p["display_overlay"]}" target="_blank" rel="noopener"><img loading="lazy" src="{p["display_overlay"]}" alt="Qwen诊断点与评估参考"></a><small>{error} · {quality}</small><details><summary>原始提示与回复</summary><pre>{esc(p["call"]["text"])}\n\n{esc(p["call"]["raw"])}</pre></details></figure>'
            htmlpart+='</div>'
        htmlpart+='<p>表中误差不是闭环成功率。详细JSON中的“真实高度平面XY误差”使用评估用高度，只是条件几何误差，不能当作可部署的三维定位结果。</p><a href="unified_cameras/pointing_results.json">全部原始调用、像素误差与标定</a></details>'
    return htmlpart

STYLE='''.rig-view-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.rig-view-grid figure,.rig-mounts figure{margin:0;min-width:0;background:#132328;border:1px solid #35545b;border-radius:8px;padding:10px}.rig-view-grid img,.rig-mounts img{display:block;width:100%;height:auto;aspect-ratio:1;object-fit:contain}.rig-view-grid figcaption{font-size:14px;color:#9fe2c4;margin-bottom:8px}.rig-view-grid small{display:block;font-size:11px;color:#b6cac5;overflow-wrap:anywhere}.rig-panel{border-top:2px solid #578a7c;margin-top:32px;padding-top:20px}.rig-panel select{max-width:100%;background:#203940;color:#e7f0ed;border:1px solid #5b7b80;border-radius:6px;padding:9px;margin:7px 0}.rig-mounts{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.rig-mounts figcaption{font-size:12px}.rig-time{font-size:12px}@media(max-width:700px){.rig-view-grid,.rig-mounts{grid-template-columns:1fr}}'''
SCRIPT='''document.querySelectorAll('.rig-panel').forEach(panel=>{const frames=JSON.parse(panel.dataset.frames);panel.querySelector('select').addEventListener('change',e=>{const frame=frames[Number(e.target.value)];panel.querySelectorAll('.rig-view').forEach(card=>{const view=frame.views.find(v=>v.camera===card.dataset.camera);card.querySelector('img').src=view.image;card.querySelector('a').href=view.image;card.querySelector('.rig-position').textContent='世界坐标 [m]: '+view.calibration.T_world_camera.position.map(v=>v.toFixed(3)).join(', ');});panel.querySelector('.rig-time').textContent='同一仿真时刻 '+frame.time.toFixed(3)+'s；'+frame.event+'；新图像仅诊断，未用于控制。';panel.querySelector('.rig-capture-link').href=frame.record;});});'''

if __name__=="__main__":
    data=build();print(json.dumps({"success":data["success_count"],"frames":data["frame_count"],"images":data["image_count"],"profile":data["profile"]["version"]},indent=2))

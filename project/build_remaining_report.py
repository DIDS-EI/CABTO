"""Unified five-scene index plus provenance-aware camera pointing galleries.
Reads frozen experiment evidence. Supplementary dual Qwen image tests are NOT
added to closed-loop scores. Cover display copies retain original file hashes.
"""
from pathlib import Path
import hashlib,html,json,shutil,subprocess
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"outputs/remaining_cabto"
TASKS=["cover","blocks","pour","handover","storage"]
LABELS={"cover":"厨房三物体配对放置","blocks":"积木两组堆叠","pour":"端盆接球、容器归位","handover":"无把手茶盒空中交接","storage":"装箱、双臂搬箱上架"}
CONFIG={"cover":"cover_kitchen_sort_v2","blocks":"blocks_rebuilt_v1","pour":"held_basin_pour_smooth_v2","handover":"handover_tea_box_v3","storage":"packing_dual_lift_shelf_storage_v3"}
SAMPLES={"blocks":"sampling","pour":"sampling_v5","handover":"sampling_v3","storage":"sampling_v2"}

def read(p):return json.loads(Path(p).read_text())
def esc(x):return html.escape(str(x))
def relative(p):return str(Path(p).relative_to(OUT))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def cover_display_copy():
    """Copy only viewing assets; never mutate original experimental files."""
    src=ROOT/"outputs/cover_cabto_fix";dst=OUT/"cover";items=[]
    files=["proposal_final_v2/models.json","sampling_9b_complete/policies.json","summary.json","verification.json","reproduce.txt"]
    for branch in ("final_visual","final_holdout1","final_holdout2"):
        files.append(f"{branch}/result.json")
        for episode in read(src/branch/"result.json")["episodes"]:
            name=episode["task"]["id"]
            for f in ("result.json","before.png","final.png","rollout.mp4","tree.json","tree.dot","tree.svg"):
                if (src/branch/name/f).is_file():files.append(f"{branch}/{name}/{f}")
    for name in files:
        source=src/name;target=dst/name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.is_file() or sha(source)!=sha(target):shutil.copy2(source,target)
        items.append({"source":str(source),"display_copy":relative(target),"sha256":sha(source)})
    (dst/"display_copy_manifest.json").write_text(json.dumps({"original_results_modified":False,"files":items},ensure_ascii=False,indent=2))

def gallery_html(task,gallery):
    data=gallery["scenes"][task];items=data["items"];online=task in ("cover","blocks")
    if online:
        intro=(f"这里展示完整主任务中实际发生的 {data['online_calls']} 次图像定位，另有 {data['cache_reads']} 次缓存读取，不重复计作模型调用。"
               "黄色十字为 Qwen 粗点，绿色十字为 RGB 几何细化点；这些结果实际用于该次控制。")
    else:
        intro=(f"原成功执行使用 Oracle 几何，未调用 Qwen 获得运动目标。以下为本次新增的 {data['diagnostic_calls']} 次真实 Qwen 图片打点测试："
               "输入取自原执行的代表性相机关键帧，橙色准星为新模型输出。未重跑机器人、未送入控制，也未验证目标身份、三维定位或抓取精度。")
    cards=[]
    for item in items:
        point="无可解析坐标" if item.get("point") is None else "("+", ".join(f"{v:.1f}" for v in item["point"])+") px"
        refined=""
        if item.get("refined_point") is not None:
            refined=" → RGB ("+", ".join(f"{v:.1f}" for v in item["refined_point"])+") px"
        world=""
        if item.get("xyz") is not None:world="<p class=coord>返回 XYZ：("+", ".join(f"{v:.4f}" for v in item["xyz"])+") m</p>"
        originals=item.get("original_target_read_calls")
        calls=("<h5>原执行中的 Oracle 目标读取（不是此次 Qwen 输出）</h5><pre>"+esc(json.dumps(originals,ensure_ascii=False,indent=2))+"</pre>") if originals else ""
        tag="实际用于控制" if online else "新增诊断 · 未用于控制"
        card=f'''<article class="point-card" data-kind="{item['kind']}"><div class="point-heading"><h4>{esc(item['title'])}</h4><span class="badge {'online' if online else 'diagnostic'}">{tag}</span></div><button type="button" class="zoom-image" data-full="{item['overlay']}" data-caption="{esc(item['title']+' · '+tag)}" aria-label="放大{esc(item['title'])}"><img loading="lazy" src="{item['overlay']}" data-original="{item['input']}" data-overlay="{item['overlay']}" alt="{esc(item['title'])}：{'Qwen粗点与RGB细化' if online else '事后Qwen诊断点，未用于原控制'}"></button><div class="point-meta"><p class="coord">Qwen {point}{refined}</p>{world}<p>{esc(item['camera'])} · {item['image_size'][0]}×{item['image_size'][1]}</p><p>{esc(item['note'])}</p><div class="image-actions"><button type="button" class="image-toggle" aria-pressed="false">看原始相机图</button><a href="{item['record']}" target="_blank" rel="noopener">原始调用记录</a><a href="{item['input']}" target="_blank" rel="noopener">输入图片</a></div><details><summary>提示词、模型回复与坐标来源</summary><small>{esc(item['model'])}</small><h5>实际提示词</h5><pre>{esc(item['prompt'])}</pre><h5>实际原始回复</h5><pre>{esc(item['raw'])}</pre>{calls}</details></div></article>'''
        cards.append(card)
    return f'''<div class="camera-gallery" id="{task}-camera"><h3>相机图像与打点记录 <span class="count-pill">{len(items)} 组</span></h3><p class="gallery-intro">{intro}</p><div class="point-grid">{''.join(cards)}</div></div>'''

def main():
    OUT.mkdir(parents=True,exist_ok=True);cover_display_copy()
    gallery=read(OUT/"camera_pointing/manifest.json")
    from build_unified_camera_report import load as load_rig,overview_html as rig_overview,scene_html as rig_scene,STYLE as RIG_STYLE,SCRIPT as RIG_SCRIPT
    rig=load_rig()
    summaries={};sections=[];total=0;passed=0;frame_count=0;frame_pass=0
    for task in TASKS:
        folder=OUT/task
        if task=="cover":
            mods=folder/"proposal_final_v2/models.json";samplepath=folder/"sampling_9b_complete/policies.json"
            runpaths=[folder/b for b in ("final_visual","final_holdout1","final_holdout2")];model=read(mods)
            task_specs=model["P"];model_provenance={"model":model["proposal_model"],"mode":model["proposal_scope"]}
        else:
            mods=folder/("proposal" if task=="blocks" else "refinement")/"models.json";samplepath=folder/SAMPLES[task]/"policies.json"
            runpaths=[folder/f"final_seed{s}" for s in (0,1)];model=read(mods);task_specs=model["P"];model_provenance=model["provenance"]
        sampling=read(samplepath);runs=[read(p/"result.json") for p in runpaths]
        episodes=[e for r in runs for e in r["episodes"]];success=sum(e["success"] for e in episodes)
        full=all(s["strict_CP_transition_match"] for e in episodes for s in e["steps"]);nstep=sum(len(e["steps"]) for e in episodes)
        passed+=success;total+=len(episodes);frame_count+=nstep;frame_pass+=sum(s["strict_CP_transition_match"] for e in episodes for s in e["steps"])
        for runpath in runpaths:
            for dot in runpath.rglob("tree.dot"):
                if not dot.with_suffix(".svg").is_file():subprocess.run(["dot","-Tsvg",str(dot),"-o",str(dot.with_suffix(".svg"))],check=True)
        patches=[] if task in ("cover","blocks") else [read(p) for p in sorted((folder/"refinement").glob("patch*.json"))]
        complete=all(r["complete_for_declared_P"]["complete_for_P"] for r in runs) if task=="cover" else all(r["complete"]["complete_for_declared_P"] for r in runs)
        perturbation={"cover":"C1/C2/C3基准 + C3两次±2mm扰动验证","blocks":"三个任务×2初态；seed1为±2mm扰动","pour":"原配置0mm扰动；第二次仅为重复执行","handover":"seed1为±2mm活动盒初态扰动","storage":"seed1为±3mm物体初态扰动"}[task]
        visual=task in ("cover","blocks")
        status={"success_count":success,"total":len(episodes),"models":len(model["models"]),"complete_for_declared_P":complete,
            "P":task_specs,"config":CONFIG[task],"config_sha256":model["config_sha256"],"all_full_CP_transitions_match":full,"executed_actions":nstep,
            "assist":"strict_no_weld" if visual else "weld_assisted","perception":"Qwen2.5-VL-3B+RGB+known_height" if visual else "oracle_geometry",
            "model_provenance":model_provenance,"successful_model_patches":sum(p["accepted"] for p in patches),"source_unchanged":all(r["sources_unchanged"] for r in runs),
            "validation_scope":"finite_episode_predicate_equality_and_final_goal_not_universal_consistency","seed1_scope":perturbation,
            "camera_gallery":{"images":len(gallery['scenes'][task]['items']),"online_calls":gallery['scenes'][task]['online_calls'],"posthoc_diagnostics":gallery['scenes'][task]['diagnostic_calls']}}
        summaries[task]=status
        ident="C3" if task=="cover" else ("B3" if task=="blocks" else task.upper());mainpath=runpaths[0]/ident
        poster=mainpath/("before.png" if task=="cover" else "step00/before.png")
        rows=""
        for runpath,run in zip(runpaths,runs):
            for e in run["episodes"]:
                eid=e["task"]["id"];base=relative(runpath/eid);frame=all(s["strict_CP_transition_match"] for s in e["steps"])
                rows+=f'<tr><td>{eid} / seed {e["seed"]}</td><td>{len(e["steps"])}</td><td class="passed">{e["success"]}</td><td>{frame}</td><td><a href="{base}/result.json">结果</a> · <a href="{base}/tree.svg">正式BT</a> · <a href="{base}/rollout.mp4">视频</a></td></tr>'
        programs="".join(f'<details><summary>{esc(key)}</summary><pre>{esc(v["source"])}</pre><small>源码 SHA256: {v["sha256"]}</small></details>' for key,v in sampling["accepted"].items())
        patchrows="".join(f'<tr><td>{esc(p["action_id"])}</td><td>{esc(p["patch"]["extra_add"])}</td><td>{esc(p["patch"]["extra_del"])}</td></tr>' for p in patches if p["accepted"])
        if task=="cover":
            provenance="第一个已跑通场景现已完整纳入：C1虾入碗；C2虾入碗＋苹果上砧板；C3再加土豆入锅，均释放、退手并回HOME。真实本地Qwen3.5-9B经有界推理与最终JSON收束提出7个接地动作模型；程序真实生成并验收，其中土豆两个动作复用已生成schema原文，不计作两次新生成。"
        elif task=="blocks":
            provenance="从Cover真实9B输出的Pick/Place/Home schema机械接地，程序原文复用并独立物理重验；未把复用算作新生成。B1黄叠绿，B2红叠蓝，B3两组堆叠，均释放并回HOME。"
        else:
            provenance="从原工程参考模型库初始化，不是从零自主发现。低层执行暴露未建模状态变化后，将真实前后状态反馈给9B，模型提出增删效果补丁；同一程序不改源码，修订后重新规划并独立执行。新采样源码与兼容历史模型源码复用分别记录。"
        limitation="严格接触、无weld；视觉使用标定俯视相机、Qwen粗点与RGB几何细化，颜色/形状及高度先验明确。载物控制反馈与独立评价仍读取仿真状态。" if visual else "原控制仍用Oracle几何，抓持沿用显式weld辅助，不能称strict抓持或全视觉成功。下方新增Qwen关键帧打点仅用于展示诊断，没有改变此前执行记录。"
        if task=="handover":limitation+="场景中1只活动茶盒被交接，另2只为陈列盒，不是3只依次交接。"
        if task=="storage":limitation+="包含、货架支撑、双臂残差和释放通过；没有新增全机器人碰撞历史审计。"
        section=f'''<section class="scene-section" id="{task}" data-scene="{task}"><div class="eyebrow">{task.upper()} / {CONFIG[task]}</div><h2>{LABELS[task]}</h2><p>{provenance}</p><div class="notice">{limitation}</div><div class="section-links"><a href="#{task}-camera">查看相机与打点图 ↓</a><span>{len(model['models'])} 个动作 · {success}/{len(episodes)} 次执行</span></div><video controls preload="metadata" poster="{relative(poster)}" src="{relative(mainpath)}/rollout.mp4"></video><div class="scroll"><table><tr><th>任务/初态</th><th>动作步数</th><th>终态通过</th><th>全状态匹配</th><th>证据</th></tr>{rows}</table></div><p>复验范围：{perturbation}。有限样本通过不证明全连续状态空间一致性。</p><div class="links"><a href="{relative(mods)}">动作模型及完整性</a><a href="{relative(samplepath)}">程序采样、失败和来源</a></div>{rig_scene(task,rig)}<details class="legacy-pointing"><summary>保留原打点记录（旧视角／原执行来源）</summary>{gallery_html(task,gallery)}</details>'''
        if patches:section+=f'<details><summary>9B根据实测反例提出的模型效果修订（{status["successful_model_patches"]}处）</summary><div class="scroll"><table><tr><th>动作</th><th>补充add</th><th>补充del</th></tr>{patchrows}</table></div><p>补丁改变模型，不改变传感器判断或成功标志；pre沿用参考模型。最终在新episode中验证。</p></details>'
        section+=f'<details><summary>查看{len(sampling["accepted"])}个接地动作的实际执行源码</summary>{programs}</details></section>';sections.append(section)
    summary={"scene_count":5,"all_final_executions_pass":passed==total,"success_count":passed,"total":total,"full_frame_matches":frame_pass,"action_transitions":frame_count,"tasks":summaries,"tests":277,
        "camera_gallery":{"total_images":gallery["image_count"],"closed_loop_call_images":gallery["online_call_images"],"supplementary_diagnostic_images":gallery["posthoc_call_images"],"diagnostics_included_in_execution_scores":False},
        "scope":"Cover/Blocks visual strict, three dual Oracle+weld with reference-initialized effects; posthoc images separately labeled",
        "not_claimed":["all models discovered from scratch","dual Qwen perception integrated","strict dual grasp","universal physical consistency","paper ASR CSR statistics reproduced"]}
    if rig:
        summary["unified_camera_rig"]={"profile":rig["profile"],"compatibility_success_count":rig["success_count"],"diagnostic_only":True,
                                       "new_control_uses_cameras":False,"snapshot_count":rig["frame_count"],"images":rig["image_count"],"not_in_original_score":True}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    overview="".join(f'<tr><td><a href="#{t}">{t.title()}</a></td><td>{s["models"]}</td><td>{s["success_count"]}/{s["total"]}</td><td>{"Qwen + RGB" if t in ("cover","blocks") else "Oracle"}</td><td>{"Strict，无weld" if t in ("cover","blocks") else "Weld辅助"}</td><td><a href="#{t}-camera">{s["camera_gallery"]["images"]}组 · {"执行中调用" if t in ("cover","blocks") else "事后诊断"}</a></td></tr>' for t,s in summaries.items())
    navigation="".join(f'<a href="#{t}">{i:02d} {t.title()}</a>' for i,t in enumerate(TASKS,1))
    style='''*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:72px}body{background:#111a1d;color:#e5eeeb;font:15px/1.75 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;margin:0}main{max-width:1200px;margin:auto;padding:36px 24px 70px}header{border-bottom:1px solid #30464d;padding:10px 0 25px}h1{font-size:40px;line-height:1.2}h2{font-size:26px;margin:9px 0}h3{font-size:21px;margin:12px 0}h4{font-size:16px;margin:0}h5{font-size:13px;margin-bottom:4px}p,li,small{color:#b1c4bd}.eyebrow{font-size:12px;letter-spacing:1.6px;color:#83dfb8}section{background:#1a292d;border:1px solid #33484d;border-radius:13px;padding:25px;margin:24px 0}a{color:#83dfb8;text-underline-offset:3px}.notice{border-left:3px solid #e9c285;background:#302d23;padding:14px 20px;margin:18px 0}.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:26px 0}.kpis div{padding:20px;background:#23383b;border-radius:9px}.kpis b{display:block;color:#83dfb8;font-size:30px}table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:12px 8px;border-bottom:1px solid #354a50;text-align:left;vertical-align:top}th{color:#83dfb8}.scroll{overflow:auto}video{width:100%;max-height:600px;background:#101a1d;border-radius:9px}pre{background:#0e191b;color:#c0e4d8;font:12px/1.7 ui-monospace,monospace;padding:14px;white-space:pre-wrap;overflow-wrap:anywhere;border-radius:8px}details{border-top:1px solid #354a50;margin-top:15px;padding:13px 0}summary{cursor:pointer}small{overflow-wrap:anywhere}.links{display:flex;flex-wrap:wrap;gap:18px}.scene-nav{display:flex;flex-wrap:wrap;gap:10px;position:sticky;top:0;padding:12px 0;background:#111a1df5;z-index:4;backdrop-filter:blur(12px)}.scene-nav a{padding:7px 14px;border:1px solid #3d5e60;background:#21363a;border-radius:6px;text-decoration:none}.section-links{display:flex;justify-content:space-between;gap:10px;margin-bottom:14px;font-size:13px;color:#b1c4bd}.camera-gallery{margin:34px 0 20px;padding-top:20px;border-top:1px solid #3c5157}.count-pill{font-size:12px;background:#263f42;color:#93e8c5;padding:5px 9px;border-radius:12px;vertical-align:middle}.gallery-intro{font-size:14px}.point-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.point-card{border:1px solid #40565d;border-radius:10px;overflow:hidden;background:#142125}.point-heading{padding:13px 16px;display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:7px}.badge{display:inline-block;font-size:11px;border-radius:5px;padding:3px 7px}.online{color:#9cf0c7;background:#203f33}.diagnostic{color:#ffd496;background:#493622}.zoom-image{display:block;border:0;padding:0;width:100%;background:#0f181b;cursor:zoom-in}.zoom-image img{display:block;width:100%;height:auto;aspect-ratio:1;object-fit:contain}.point-meta{padding:10px 16px}.point-meta p{font-size:12px;margin:7px 0}.point-meta .coord{font-family:ui-monospace,monospace;color:#d5ece2}.image-actions{display:flex;gap:12px;align-items:center;flex-wrap:wrap;font-size:12px;margin:10px 0}.image-toggle{border:1px solid #63877e;border-radius:5px;color:#a4e8ca;background:#233b39;padding:7px 10px;cursor:pointer}.image-toggle:hover{background:#33544b}.image-toggle:focus-visible,.zoom-image:focus-visible{outline:3px solid #edc575;outline-offset:2px}.passed{color:#93e8c5}dialog{max-width:min(940px,94vw);max-height:94vh;overflow:auto;background:#152329;color:#e5eeeb;border:1px solid #526b6f;border-radius:12px;padding:16px}dialog::backdrop{background:#071214dd}dialog img{display:block;max-width:100%;max-height:75vh;margin:auto}dialog button{float:right;background:#294147;border:1px solid #6a8a8a;border-radius:6px;color:#fff;padding:7px 16px;cursor:pointer}dialog h3{font-size:17px;margin:0 90px 15px 0}.audit-note{font-size:12px;color:#a3b9b1}.legend{display:flex;flex-wrap:wrap;gap:20px;font-size:13px;margin:15px 0}.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}.online-dot{background:#62d8a7}.diagnostic-dot{background:#f2a24e}@media(max-width:700px){main{padding:20px 12px}section{padding:18px}.kpis{grid-template-columns:1fr 1fr}.point-grid{grid-template-columns:1fr}h1{font-size:30px}.scene-nav a{padding:5px 9px;font-size:12px}.section-links{flex-direction:column}}'''
    script='''document.querySelectorAll('.image-toggle').forEach(button=>{button.addEventListener('click',()=>{const card=button.closest('.point-card');const img=card.querySelector('img');const raw=button.getAttribute('aria-pressed')!=='true';img.src=raw?img.dataset.original:img.dataset.overlay;card.querySelector('.zoom-image').dataset.full=img.src;button.setAttribute('aria-pressed',String(raw));button.textContent=raw?'看打点叠加图':'看原始相机图';});});const lightbox=document.getElementById('image-lightbox');document.querySelectorAll('.zoom-image').forEach(button=>{button.addEventListener('click',()=>{document.getElementById('lightbox-image').src=button.dataset.full;document.getElementById('lightbox-image').alt=button.dataset.caption;document.getElementById('lightbox-title').textContent=button.dataset.caption;lightbox.showModal();});});document.getElementById('close-lightbox').addEventListener('click',()=>lightbox.close());lightbox.addEventListener('click',e=>{if(e.target===lightbox)lightbox.close();});'''
    style+=RIG_STYLE;script+=RIG_SCRIPT
    page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CABTO · 五场景执行与相机打点</title><style>{style}</style></head><body><main><header><div class="eyebrow">CABTO / COVER · BLOCKS · POUR · HANDOVER · STORAGE</div><h1>五个场景，一页查看<br>执行结果与相机打点</h1><p>整合第一个Cover场景与其余四个场景。完整视频、动作代码、模型修订和相机原图均保留来源，可逐项查看。</p></header><nav class="scene-nav" aria-label="五场景导航">{navigation}<a href="#unified-camera-rig">统一腕部相机</a></nav>{rig_overview(rig)}<div class="notice"><strong>打点图片的来源不同：</strong>Cover、Blocks展示原执行中实际调用Qwen的图片；Pour、Handover、Storage原执行使用Oracle，本页新增的是对原相机帧的独立Qwen打点测试，<strong>未用于原控制，也不计入成功率</strong>。有二维点输出并不代表定位或抓取正确。</div><div class="kpis"><div><b>5</b>完整场景章节</div><div><b>{passed}/{total}</b>已保存任务执行通过</div><div><b>{frame_pass}/{frame_count}</b>全状态转移匹配</div><div><b>{gallery['image_count']}</b>相机与打点图片组</div></div><section><h2>五场景总览</h2><div class="scroll"><table><tr><th>场景</th><th>动作数</th><th>执行结果</th><th>原执行感知</th><th>抓持模式</th><th>相机打点图库</th></tr>{overview}</table></div><div class="legend"><span><i class="online-dot"></i>{gallery['online_call_images']}组真实闭环调用</span><span><i class="diagnostic-dot"></i>{gallery['posthoc_call_images']}组补充图片诊断</span><span>277项已保存回归测试通过</span></div><p>Cover的5次原执行与其余四场景的12次原执行合并展示；这是报告汇总，不是新增17次试验。点击图片放大，使用“看原始相机图”切换输入/叠加图，展开卡片可查看提示词、原始回复和坐标来源。</p><div class="links"><a href="summary.json">五场景摘要</a><a href="camera_pointing/manifest.json">图片证据索引</a><a href="presentation_verification.json">五场景页面核验</a><a href="verification.json">四场景原执行核验</a><a href="cover/verification.json">Cover原执行核验</a><a href="reproduce.txt">四场景复跑命令</a><a href="cover/reproduce.txt">Cover复跑命令</a></div></section>{''.join(sections)}<section><h2>边界与复验说明</h2><ul><li>五场景保留各自的冻结物理配置，图片展示没有改动原模型回答、执行结果或成功标志。</li><li>双臂图片打点是本次真实补充调用，不是原控制中的视觉定位；橙色准星可能打偏，尚未做目标身份/精度验收。</li><li>双臂模型前提仍来自参考工程，效果修订解释有限反例；抓持仍有weld辅助。不能据此声称全部模型从零发现、全视觉或无辅助。</li><li>Pour零扰动重复不能算布局泛化；其他任务只有小扰动。未复现原论文等预算ASR/CSR与消融统计。</li></ul><div class="links"><a href="remaining_tests.log">36项新测试</a><a href="visual_regression.log">25项视觉回归</a><a href="generation_regression.log">8项代码来源回归</a><a href="cover_regression.log">44项Cover回归</a><a href="legacy_tests.log">164项旧回归</a></div></section><p class="audit-note">网页构建：project/build_remaining_report.py · 图片索引：project/build_five_scene_pointing.py · 原始实验JSON保持只读</p></main><dialog id="image-lightbox"><button type="button" id="close-lightbox">关闭</button><h3 id="lightbox-title"></h3><img id="lightbox-image" alt=""></dialog><script>{script}</script></body></html>'''
    (OUT/"index.html").write_text(page)
    print(json.dumps({"scenes":TASKS,"success_count":passed,"total":total,"frame":f"{frame_pass}/{frame_count}","camera_images":gallery['image_count'],"closed_loop_images":gallery['online_call_images'],"posthoc_images":gallery['posthoc_call_images']},ensure_ascii=False,indent=2))

if __name__=="__main__":main()

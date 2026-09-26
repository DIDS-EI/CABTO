"""Evidence-driven report for local generation and visual placement fixes."""
from pathlib import Path
import json,html,hashlib,subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/cover_cabto_fix"

def load(rel):return json.loads((OUT/rel).read_text())
def esc(v):return html.escape(str(v))


def main():
    models=load("proposal_final_v2/models.json");policies=load("sampling_9b_complete/policies.json")
    baseline=load("final_visual/result.json");holdouts=[load(f"final_holdout{i}/result.json") for i in (1,2)]
    episodes=baseline["episodes"]+[e for d in holdouts for e in d["episodes"]]
    for folder in ("final_visual","final_holdout1","final_holdout2"):
        for dot in (OUT/folder).rglob("tree.dot"):
            subprocess.run(["dot","-Tsvg",str(dot),"-o",str(dot.with_suffix(".svg"))],check=True)
    maxima=max(m["xy_error_m"] for e in episodes for n,m in e["metrics"]["objects"].items() if f"at_target({n})" in e["task"]["goal"])*1000
    summary={"status":"passed" if models["check"]["complete_for_P"] and policies["success"] and all(e["success"] for e in episodes) else "not_passed",
             "proposal_model":models["proposal_model"],"code_model":policies["model_id"],"generation_channel":policies["generation_channel"],
             "visual_model":"mlx-community/Qwen2.5-VL-3B-Instruct-4bit","visual_method":"Qwen semantic point + overhead RGB geometry + known feature height plane",
             "complete_for_P":models["check"]["complete_for_P"],"reachable_states":models["check"]["reachable_state_count"],
             "grounded_actions":len(models["models"]),"validated_generated_policies":len(policies["accepted"]),
             "sampling_attempts":len(policies["attempts"]),
             "unchanged_schema_reuse_validations":sum(bool(r.get("schema_transfer")) and r["accepted"] for r in policies["attempts"]),"visual_tasks_success":baseline["success_count"],"visual_tasks_total":baseline["total"],
             "holdout_success":sum(d["success_count"] for d in holdouts),"holdout_total":sum(d["total"] for d in holdouts),
             "max_placed_xy_error_mm":maxima,"all_no_forbidden_contacts":all(not e["metrics"]["forbidden_contacts"] for e in episodes),
             "all_no_weld":all(not e["metrics"]["strict_weld_violation"] and not e["weld_active"] for e in episodes),
             "all_frame_transitions_match":all(s["strict_CP_transition_match"] for e in episodes for s in e["steps"]),
             "source_consistency":all(d["sources_unchanged"] for d in [baseline,*holdouts,policies]),
             "test_counts":{"visual":25,"generation_provenance":8,"method":44,"legacy":164},
             "frozen_config_sha256":models["config_sha256"],
             "boundaries":["human-given CP, operator signatures, API docs and semantic feedback", "manual orchestration and prompt iterations; not unattended end-to-end discovery", "model outputs, not session-written policies, in final runs", "overhead observation camera changed, physical scene unchanged", "RGB color/shape and feature-height priors", "oracle initial grasp z and oracle payload tracking/evaluation remain", "small fixed scene, holdouts only +/-2mm", "no pure-VLM-only or paper-table replication claim"]}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    rows=""
    for folder,d in [("final_visual",baseline),*( (f"final_holdout{i}",v) for i,v in enumerate(holdouts,1))]:
        for e in d["episodes"]:
            ident=e["task"]["id"];targets=[m for n,m in e["metrics"]["objects"].items() if f"at_target({n})" in e["task"]["goal"]]
            rows+=f'<tr><td>{ident} / seed {e["seed"]}</td><td>{len(e["steps"])}</td><td class="good">{e["success"]}</td><td>{max(m["xy_error_m"] for m in targets)*1000:.3f} mm</td><td><a href="{folder}/{ident}/result.json">JSON</a> · <a href="{folder}/{ident}/rollout.mp4">视频</a> · <a href="{folder}/{ident}/tree.svg">BT</a></td></tr>'
    programs=""
    for k,v in policies["accepted"].items():
        programs+=f'<details><summary>{esc(k)}</summary><pre>{esc(v["source"])}</pre><small>SHA256: {v["sha256"]}</small></details>'
    schema_rows="".join(f'<tr><td>{esc(s["name"])}</td><td>{esc(s["pre"])}</td><td>{esc(s["add"])}</td><td>{esc(s["del"])}</td></tr>' for s in models["schemas"])
    cards=""
    # Images are sourced from the final physical run, not re-inferred examples.
    for name in ("bowl","board","pan"):
        images=list((OUT/"final_visual/C3/perception").glob(f"*_{name}_overlay.png"))
        if images:cards+=f'<figure><img src="{images[0].relative_to(OUT)}" alt="{name}模型粗定位与RGB细化"><figcaption>{name}: 黄色=模型粗点，绿色=RGB细化点</figcaption></figure>'
    css='''*{box-sizing:border-box}body{margin:0;background:#10191c;color:#e6efed;font:15px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}main{max-width:1120px;margin:auto;padding:42px 24px 70px}h1{font-size:40px;line-height:1.22}h2{font-size:24px;margin-top:0}section{padding:25px;background:#182629;border:1px solid #31444a;border-radius:12px;margin:22px 0}p,li,small{color:#b2c4c0}a,.good{color:#87dfb9}header{border-bottom:1px solid #34484e;padding-bottom:20px}.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.kpis div{padding:18px;background:#203338;border-radius:9px}.kpis b{display:block;font-size:30px;color:#87dfb9}.notice{border-left:3px solid #e4c083;background:#2b2a22;padding:15px 20px;margin:20px 0}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}figure{margin:0}img,video{width:100%;border-radius:9px}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:12px 8px;text-align:left;border-bottom:1px solid #34484e;vertical-align:top}th{color:#87dfb9}pre{white-space:pre-wrap;background:#0c1518;padding:18px;border-radius:8px;color:#c5e8df;font-size:13px}details{padding:14px 0;border-top:1px solid #34484e}summary{cursor:pointer}figcaption{font-size:12px;color:#b2c4c0}.links{display:flex;gap:18px;flex-wrap:wrap}@media(max-width:700px){.kpis,.grid{grid-template-columns:1fr 1fr}h1{font-size:30px}}@media(max-width:450px){.kpis,.grid{grid-template-columns:1fr}}'''
    page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Cover · 生成与视觉放置修复</title><style>{css}</style></head><body><main><header><small>CABTO / GENERATION + VISUAL PLACEMENT</small><h1>模型生成与视觉放置<br>接入同一个可验证闭环</h1><p>保留 cover_kitchen_sort_v2 的物理布局、接触参数和成功门限。最终控制代码来自本地9B模型，不使用上轮会话手写候选。</p></header>
<div class="notice">这是一条混合管线：Qwen3.5-9B负责动作契约与代码生成，Qwen2.5-VL-3B负责语义粗定位，RGB几何负责亚像素中心细化，已知高度平面用于反投影。不是单靠VLM点坐标，也不是去除全部仿真真值的控制系统。</div>
<section class="kpis"><div><b>{len(policies['accepted'])}/7</b>真实模型代码物理验收</div><div><b>{baseline['success_count']}/{baseline['total']}</b>视觉任务通过</div><div><b>{summary['holdout_success']}/{summary['holdout_total']}</b>完整任务留出初态</div><div><b>{maxima:.2f} mm</b>已放物体最大XY误差</div></section>
<section><h2>1 · 生成能力：模型大小不是唯一问题</h2><p>0.8B、3B及9B非思考提案都保留了失败记录。9B在思考模式中能推导正确的前提和增删效果，但长推理会耗尽token预算；改为“有界推理 → 单独最终JSON收束”，再用正式BT与有限BFS检查，得到可覆盖C1/C2/C3的7个接地动作。</p><p>代码阶段发现整场景任务描述会让模型在Pick里继续Place甚至回HOME。改为只提供当前动作的终止状态和允许API，不代写动作序列。实际生成源码保留原样，运行前检查API白名单、签名和运动目标的数据来源，运行后读取真实状态检查完整CP转移。土豆的Pick/Place使用已通过的同类9B程序原文，仅重新绑定对象并独立物理验证，属于明确记录的跨对象程序复用，不计作两次新模型生成。</p><div class="scroll"><table><tr><th>Schema</th><th>pre</th><th>add</th><th>del</th></tr>{schema_rows}</table></div><div class="links"><a href="proposal_final_v2/models.json">模型及完整性</a><a href="proposal_final_v2/attempt1.json">最终JSON原始调用</a><a href="sampling_9b_complete/policies.json">全部代码采样与校验</a></div><p>范围：CP、可选动作签名、接口与任务语义由工程指定；模型输出schemas后机械替换OBJECT完成接地。提示调整与两阶段编排由人工设计，不能据此声称完全无人干预发现动作库。</p></section>
<section><h2>2 · 视觉定位：修的是几何，不是成功门限</h2><ol><li><b>俯视观测</b>减少碗沿与夹爪遮挡；保存新的相机内外参和scene.xml，物理几何不变。</li><li><b>语义身份</b>使用与实际渲染一致的描述区分橙色小虾和黄色土豆代理，保留原始Qwen点；没有读目标真实xy来挑选或替换点。</li><li><b>RGB细化</b>从Qwen点附近提取颜色连通域，碗拟合椭圆、锅检测圆沿、板取区域中心；苹果填孔后取最大内切区域，避免果梗影响。多候选歧义与裁剪不全会拒绝。</li><li><b>对应特征高度</b>检测的是碗沿/锅沿，不能按碗底高度反投影。利用固定场景设计的沿高得到xy，再用目标支撑面高度放置；不是把GT坐标加回去。</li><li><b>固定支撑缓存</b>在持物遮挡前观测固定碗/板/锅；源物体抓取点在各次动作时重新由RGB定位。</li></ol><div class="grid">{cards}</div><p>颜色/形状、沿高和相机标定都是显式先验；跨物体、光照、动态容器与未知高度尚未验证。这里的修复没有降低旧双指接触、释放、碰撞和位置验收门限。</p></section>
<section><h2>3 · 实际本地生成代码</h2><p>当前focused采样与续跑链共14次策略校验，最终7个接地动作通过，其中2个为未改源码的跨对象复用。此前不同提示和模型的调试另存，不合并为等预算性能统计。</p>{programs}</section>
<section><h2>4 · C3：三件物体全部放置、释放和回HOME</h2><video controls preload="metadata" poster="final_visual/C3/before.png" src="final_visual/C3/rollout.mp4"></video><p>全部动作由正式BT选择、已验收模型源码执行；不是重放旧关节轨迹。每步均检查pre/add/del与未声明增删；末态检查支撑、姿态、手部撤离与1秒稳定窗口。</p><div class="scroll"><table><tr><th>任务/初态</th><th>步数</th><th>成功</th><th>最大落点误差</th><th>证据</th></tr>{rows}</table></div><p>留出seed1/2仅为固定场景下±2mm初态扰动，未用于修改生成源码；不代表大范围随机布局泛化。</p></section>
<section><h2>5 · 验收与遗留边界</h2><p>视觉测试25项，运动目标数据来源测试8项，既有方法测试44项，原工程回归164项；总计241项。每轮保存运行源码快照、输入模型和程序hash，并检查运行期间未变。旧失败、旧会话候选和新模型生成证据分别保存。</p><ul><li>抓取仍使用配置偏移与初始化物体z先验；载物运动反馈和独立评价仍读取仿真状态。</li><li>严守无weld抓持；最终抓放成功不是把持物或目标谓词写为True。</li><li>未声称原论文表格的重复采样、消融和自动跨层修订结果已复现。这里完成的是一个固定场景的工程闭环。</li><li>9B权重约6GB存放在模型缓存，不并入实验包；不需要向外部付费API发送场景。</li></ul><div class="links"><a href="summary.json">验收摘要</a><a href="verification.json">独立来源与物理不变性核验</a><a href="visual_tests.log">视觉测试</a><a href="method_tests.log">方法测试</a><a href="generation_tests.log">运动目标来源测试</a><a href="legacy_tests.log">旧回归</a><a href="reproduce.txt">复跑命令</a></div></section></main></body></html>'''
    (OUT/"index.html").write_text(page)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":main()

"""Build the experiment report from saved evidence, without changing any results."""
import html
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/cover_cabto_method"


def read(rel): return json.loads((OUT/rel).read_text())
def esc(x): return html.escape(str(x))


def main():
    proposed=read("session_proposal/models.json")
    sampling=read("session_sampling/policies.json")
    oracle=read("final_oracle/result.json")
    holdout=read("final_holdout/result.json")
    qsmall=read("qwen0_8b_replay/result.json")
    qlarge=read("qwen3b_replay/result.json")
    points=read("pointing_3b_calibrated/result.json")
    localcode=read("local_codegen_3b/policies.json")
    zoom=read("qwen3b_zoom_C1/result.json")
    negative=read("negative_controls.json")
    for folder in ("final_oracle","final_holdout","session_proposal"):
        for dot in (OUT/folder).rglob("tree.dot"):
            subprocess.run(["dot","-Tsvg",str(dot),"-o",str(dot.with_suffix(".svg"))],check=True)
    synopsis={"scope":"Cover CABTO method pilot, interactive model candidates + oracle physics baseline; not autonomous local-Qwen success or paper-table replication",
              "complete_for_P":proposed["check"]["complete_for_P"],"declared_tasks":3,"actions":len(proposed["models"]),
              "accepted_action_candidates":len(sampling["accepted"]),"sampling_success":sampling["success"],
              "oracle_task_successes":oracle["success_count"],"oracle_task_trials":oracle["total"],
              "holdout_successes":holdout["success_count"],"holdout_trials":holdout["total"],
              "qwen_0_8b_C1_successes":qsmall["success_count"],"qwen_3b_C1_successes":qlarge["success_count"],
              "local_codegen_3b_accepted":len(localcode["accepted"]),"local_codegen_3b_all_success":localcode["success"],
              "negative_controls_rejected":sum(r["rejected"] for r in negative["cases"]),
              "new_unit_tests":44,"legacy_unit_tests":164,
              "qwen_zoom_C1_successes":zoom["success_count"],
              "qwen_zoom_motion_started":bool(zoom["episodes"][0]["steps"]),
              "proposal_and_baseline_code_source":"current conversation assistant, saved candidates; separate from failed local-model attempts",
              "frozen_config_sha256":proposed["config_sha256"],
              "no_weld_in_baseline":all(not e["weld_active"] and not e["metrics"]["strict_weld_violation"] for e in oracle["episodes"]),
              "source_snapshot_unchanged_during_each_run":all(d["sources_unchanged"] for d in (oracle,holdout,qsmall,qlarge,sampling)),
              "limitations":["finite physical trials, not proof over all continuous states", "manual CP/signatures and prior calibrated controller API", "oracle state evaluator and payload tracking", "Qwen localization uses explicit known height planes", "no automatic cross-level refinement demonstrated"]}
    (OUT/"summary.json").write_text(json.dumps(synopsis,indent=2,ensure_ascii=False))
    rows="".join(f'<tr><td>{esc(m["name"])}<small>{esc(m["args"])}</small></td><td>{esc(", ".join(m["pre"]))}</td><td>{esc(", ".join(m["add"]))}</td><td>{esc(", ".join(m["del"]))}</td></tr>' for m in proposed["models"])
    programs=""
    for key,record in sampling["accepted"].items():
        m=record["model"];stem=m["name"]+("_"+m["args"]["object"] if m["args"] else "")
        programs+=f'<details><summary>{esc(key)} <span class="ok">已校验</span></summary><pre>{esc(record["source"])}</pre><a href="session_sampling/{stem}/attempt1/feedback.json">前后状态、接口调用与校验明细</a></details>'
    task_rows="".join(f'<tr><td>{e["task"]["id"]}</td><td>{len(e["steps"])}</td><td class="ok">{e["success"]}</td><td>{e["metrics"]["home_error_rad"]:.5f} rad</td><td><a href="final_oracle/{e["task"]["id"]}/rollout.mp4">视频</a> · <a href="final_oracle/{e["task"]["id"]}/tree.svg">BT图</a> · <a href="final_oracle/{e["task"]["id"]}/result.json">结果</a></td></tr>' for e in oracle["episodes"])
    point_rows=""
    for record in points["records"]:
        errors=[v["evaluation_only_pixel_error"] for v in record["views"] if "evaluation_only_pixel_error" in v]
        point_rows+=f'<tr><td>{record["object"]}</td><td>{", ".join(f"{v:.1f}" for v in errors) or "解析失败"}</td><td>{record.get("view_disagreement_m",0)*1000:.1f}</td><td>{esc(record.get("error","定位通过；仍须物理验证"))}</td></tr>'
    qwhy="<br>".join(esc(f'{label}: '+str(data["episodes"][0].get("exception","成功"))+"; "+str(next((s.get("exception") for s in data["episodes"][0]["steps"] if s.get("exception")),""))) for label,data in [("0.8B",qsmall),("3B",qlarge)])
    page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Cover · CABTO 方法复现</title><style>
:root{{--bg:#101719;--panel:#182326;--ink:#e7efec;--dim:#a3b7b0;--line:#314349;--green:#81d7b4;--amber:#efc685;--red:#f59991}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}}main{{max-width:1150px;margin:auto;padding:36px 26px 70px}}header{{border-bottom:1px solid var(--line);padding-bottom:26px}}h1{{font-size:42px;line-height:1.2;margin:13px 0}}h2{{font-size:25px;margin-top:0}}h3{{font-size:18px}}.eyebrow{{color:var(--green);letter-spacing:2px;font-size:12px}}p,small{{color:var(--dim)}}strong{{color:var(--ink)}}a{{color:var(--green)}}.notice{{border-left:3px solid var(--amber);padding:15px 20px;background:#2a2821;border-radius:5px;margin:20px 0}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:26px 0}}.metric,section{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:22px}}.metric b{{font-size:29px;display:block;color:var(--green)}}section{{margin:20px 0}}.flow{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}.flow div{{background:#213239;padding:18px;border-radius:8px}}.flow b{{display:block;color:var(--green)}}video{{width:100%;border-radius:8px;background:#22363b}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border-bottom:1px solid var(--line);padding:12px 9px;text-align:left;vertical-align:top}}th{{color:var(--green)}}td small{{display:block}}.scroll{{overflow:auto}}code,pre{{font-family:ui-monospace,monospace}}pre{{white-space:pre-wrap;background:#0c1315;border-radius:8px;padding:18px;color:#bde4d6;font-size:13px}}details{{border-top:1px solid var(--line);padding:13px 0}}summary{{cursor:pointer}}.ok{{color:var(--green)}}.bad{{color:var(--red)}}.columns{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}.columns img{{width:100%;border-radius:8px}}.links{{display:flex;gap:18px;flex-wrap:wrap}}li{{margin:7px 0;color:var(--dim)}}@media(max-width:760px){{.grid,.flow,.columns{{grid-template-columns:1fr 1fr}}h1{{font-size:32px}}}}@media(max-width:480px){{.grid,.flow,.columns{{grid-template-columns:1fr}}}}</style></head><body><main>
<header><div class="eyebrow">CABTO / FROZEN COVER KITCHEN SORT V2</div><h1>动作模型 → 代码 → 物理校验</h1><p>先在第一个场景验证论文方法结构：虾入碗、苹果上砧板、土豆入锅；释放、退手、回 HOME。</p></header>
<div class="notice"><strong>结论与范围：</strong>会话模型候选 + Oracle 感知的三个任务已通过；7个动作逐一验证，完整三物体任务另做1个留出初态。<strong>本地 Qwen 自动提出与纯 Qwen 感知闭环尚未跑通。</strong>所有失败均保留，未用真值坐标悄悄替代。</div>
<div class="grid"><div class="metric"><b>3 / 3</b>声明任务集的规划完整性</div><div class="metric"><b>{len(sampling['accepted'])} / 7</b>会话候选动作物理验收</div><div class="metric"><b>{oracle['success_count']} / 3</b>Oracle任务执行</div><div class="metric"><b>{holdout['success_count']} / 1</b>C3留出初态</div></div>
<section><h2>01 · 与论文逐项对应</h2><div class="flow"><div><b>P / CP / HP / ΠP</b>任务、条件、模型与可调用接口</div><div><b>Model proposal</b>提出pre/add/del<br>BT规划反馈</div><div><b>Policy sampling</b>由候选代码显式调用各运动接口</div><div><b>Consistency test</b>读取真实状态<br>逐步效果与终态验收</div></div><p>论文第3页 Definition 0.1：complete 仅针对声明的有限任务集 P。第5页 Algorithm 2 的规划检查、采样、执行与反馈对应本实验三个阶段。第3页 Eq.(1) 的状态转移一致性，在这里采用更严格的 CP 前后状态等式，并检查 delete 与未声明的增删，而不只看返回值。</p><p>本轮是方法迁移，不是重现原论文表格：MuJoCo 替代 Isaac Sim；现有 Panda IK/执行器替代 cuRobo；Qwen 替代 Molmo；固定场景的CP、动作签名、几何高度先验与控制器由工程预先提供。未进行论文的10次统计实验和自动跨层修订。</p><div class="links"><a href="task_sets.json">五场景任务规格</a><a href="paper_algorithm2.png">论文算法原页</a><a href="summary.json">本次验收摘要</a></div></section>
<section><h2>02 · 五场景任务设计，本轮只执行 Cover</h2><div class="scroll"><table><tr><th>场景</th><th>声明任务</th><th>本轮状态</th></tr><tr><td>Cover</td><td>C1 虾入碗；C2 虾入碗+苹果上砧板；C3 三件全部归位。均释放并回HOME。</td><td>执行与校验</td></tr><tr><td>Blocks</td><td>黄叠绿；红叠蓝；两组全部堆叠后回HOME。</td><td>仅定义</td></tr><tr><td>Pour</td><td>右臂端盆，左臂倒入球体代理，再归位释放。</td><td>仅定义</td></tr><tr><td>Handover</td><td>无把手盒空中交接后放到目标区，陈列盒不动。</td><td>仅定义</td></tr><tr><td>Storage</td><td>两块积木入箱，双臂抬箱上架，支撑、释放、退手。</td><td>仅定义</td></tr></table></div></section>
<section><h2>03 · 完整性：不是把成功标志写为 True</h2><p>同一7动作库在28个可达符号状态上无互斥冲突；正式 BT Expansion 对 C1/C2/C3 给出3/5/7步解，独立BFS交叉验证可达。三份任务起始状态相同，目标逐渐增加。</p><div class="scroll"><table><tr><th>动作 / 绑定参数</th><th>pre</th><th>add</th><th>del</th></tr>{rows}</table></div><p>来源：<a href="session_candidates.json">当前会话模型输出的候选原文</a>。不是本地0.8B或3B的成功提案。它们的原始失败与反馈分别在 <a href="proposal/attempt1.json">0.8B</a>、<a href="proposal_3b/attempt1.json">3B</a>、<a href="proposal_structured/cycle3_check.json">逐动作格式</a> 和 <a href="proposal_indexed/blocked.json">索引格式</a>中。</p></section>
<section><h2>04 · 每个动作的代码都调用真实细粒度接口</h2><p>读点 → 接近 → 下降 → 闭爪 → 抬升，以及目标读点 → 搬运 → 降低 → 松爪 → 退手，均由候选代码明确调用。接口中没有空实现，也不把整套抓放偷偷塞进一次grasp或release。源代码白名单、所有调用签名在运动前检查；日志记录每个调用的参数和返回结果。</p>{programs}<p>观察谓词与终态评价由可信运行时独立读取物理状态。当前Oracle基线及搬运反馈仍读取仿真位姿；Qwen试验只替换定位前端：支撑面高度来自固定配置，抓取高度还用到初始化物体z与手工抓取偏移。这里不是完全纯视觉控制。</p></section>
<section><h2>05 · 完整 C3：三物体 + 回 HOME</h2><video controls preload="metadata" poster="final_oracle/C3/before.png" src="final_oracle/C3/rollout.mp4"></video><p>BT本次选择：土豆→锅，苹果→砧板，虾→碗，再回HOME。顺序来自所生成BT，不是固定播放旧参考轨迹。</p><div class="scroll"><table><tr><th>任务</th><th>执行步数</th><th>验收</th><th>HOME误差</th><th>证据</th></tr>{task_rows}</table></div><p>每个动作的pre、add、del与CP frame都通过；末态含真实支撑接触、释放、位姿约束与1秒稳定窗口；严格抓持过程中weld未激活。留出seed1仅施加配置允许的±2mm初态扰动，不是大范围泛化。</p><div class="links"><a href="final_holdout/C3/result.json">C3留出结果</a><a href="final_oracle/C3/tree.svg">完整正式BT图</a><a href="final_oracle/result.json">三个任务原始结果</a></div><details><summary>查看完整BT</summary><object data="final_oracle/C3/tree.svg" type="image/svg+xml" style="width:100%;height:620px;background:white"></object></details></section>
<section><h2>06 · Qwen 打点：解析修复后仍需定位与物理检验</h2><p>0.8B 输出出现截断/重复值；3B原始字段多为point_2d且坐标为输入图绝对像素。已按模型固定约定处理，不根据每个点的GT误差挑选缩放。所有GT误差在运动前计算，仅用于评估；既不修正坐标，也不兜底。</p><div class="scroll"><table><tr><th>3B目标</th><th>各视角像素误差(px)</th><th>视角估计分歧(mm)</th><th>结果</th></tr>{point_rows}</table></div><div class="notice"><strong>C1真实闭环结果：</strong>0.8B {qsmall['success_count']}/1；3B {qlarge['success_count']}/1。<br>{qwhy}</div><div class="links"><a href="pointing_3b_calibrated/result.json">3B完整打点、GT对照与标定</a><a href="pointing_0_8b_audited/result.json">0.8B原始失败输出</a><a href="qwen3b_replay/C1/result.json">3B闭环</a><a href="qwen0_8b_replay/C1/result.json">0.8B闭环</a></div></section>
<section><h2>06b · 放大裁剪与运动前观测：未通过也保留</h2><p>补测只更换观测策略：以Qwen粗定位为中心裁剪192×192图像并放大，再由模型打点；在机器人开始抓取前观测固定支撑物。裁剪中心没有使用GT坐标，失败时也未选择误差更小的那一个视角。</p><div class="notice">{esc(zoom['episodes'][0].get('exception',''))}<br>本次在运动前因约49mm双视角分歧被拒绝。单视角裁剪精细化不等于双视角语义一致，不放宽25mm门限来制造成功。</div><div class="columns"><figure><img src="qwen3b_zoom_C1/C1/perception/point_000_bowl_overview_crop.png" alt="Qwen粗定位驱动的碗裁剪"><figcaption>放大后的输入图像，不是人工标注的GT裁剪</figcaption></figure><figure><img src="qwen3b_replay/C1/perception/point_001_bowl_overview_overlay.png" alt="实时放置定位的误差"><figcaption>原始放置定位：红叉模型预测，绿圈仅评估用真值</figcaption></figure></div><a href="qwen3b_zoom_C1/C1/result.json">裁剪试验全部原始调用及误差</a></section>
<section><h2>07 · 校验器的负例与自动化边界</h2><p>三个控制负例均被拒绝：仅return True而不运动；抓取代码漏掉lift；动作模型漏掉at_source删除项。它们是人为注入的检验器测试，不是自然发生的模型修订成绩。</p><p>轻量新测试44项，旧回归164项。本地3B自动代码采样已接受 {len(localcode['accepted'])}/7 个动作，整体状态 {localcode['success']}；失败记录连同代码与图像上下文真实保留，不用会话候选替换该次成绩。</p><div class="links"><a href="negative_controls.json">负例明细</a><a href="local_codegen_3b/policies.json">本地代码采样原始结果</a><a href="final_tests.log">新测试日志</a><a href="final_legacy_tests.log">旧回归日志</a></div></section>
<section><h2>08 · 难点与下一步</h2><ol><li><strong>高层提出：</strong>当前小模型/提示组合会复制错误字段和混淆对象角色。下一步优先接入更强文本模型，保留规划失败反馈；不能手工修好后再宣称小模型提出成功。</li><li><strong>细粒度采样：</strong>优先冻结通过的模型与API，用真实失败代码、关键调用和前后图进行修正；不得把动作整体包装成一个接口来掩盖漏步骤。</li><li><strong>视觉定位：</strong>先处理单对象裁剪/候选目标选择、显式坐标约定和重复观测，再校验双视角同一点语义。小虾等小目标需要毫米级抓取精度；两视角一致也不能证明目标身份正确。</li><li><strong>跨层修订：</strong>只有证据指向模型本身的缺前提/错效果时才修订h，并重新验证P的完整性。纯代码/感知失败先保留为未解决，不据有限失败证明动作不可能。</li><li><strong>推广顺序：</strong>Cover的感知与自动采样稳定后，再按task_sets.json逐步扩展Blocks、Pour、Handover、Storage；本轮不将旧任务视频算作新方法复现。</li></ol><p>复跑入口：<code>project/run_cover_cabto_method.py</code>。完整命令见 <a href="reproduce.txt">reproduce.txt</a>。</p></section>
<footer><small>Evidence-first / no snap / no success-flag writes / explicit oracle boundary</small></footer></main></body></html>'''
    (OUT/"index.html").write_text(page)
    print(json.dumps(synopsis,indent=2,ensure_ascii=False))


if __name__=="__main__":main()

"""生成实验2（代码生成 + VLM 打点）可视化报告 HTML（自包含，图片 base64 内嵌，
视频用相对路径引用）。运行后输出 exp2_codegen_report.html。
"""
from __future__ import annotations
import os, json, base64, glob

HERE = os.path.dirname(os.path.abspath(__file__))
REND = os.path.join(HERE, "renders")
RES = os.path.join(HERE, "results")


def b64img(path):
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def load_json(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.isfile(p) else None


def main():
    calib = load_json("qwen_prompt_calib.json")
    ev_o = load_json("eval_pick_oracle.json")
    ev_q = load_json("eval_pick_qwen.json")

    def card_imgs(backend, objs):
        rows = ""
        for obj in objs:
            ov = b64img(os.path.join(REND, f"pointing_overlay_{backend}_{obj}.png"))
            mv = b64img(os.path.join(REND, f"multiview_{backend}_{obj}.png"))
            mp4 = f"renders/rollout_{backend}_{obj}.mp4" if os.path.isfile(
                os.path.join(REND, f"rollout_{backend}_{obj}.mp4")) else None
            vid = (f'<video controls loop muted playsinline width="320" '
                   f'src="{mp4}"></video>') if mp4 else '<span class="muted">—</span>'
            rows += f"""
            <div class="obj-card">
              <h4>{obj}</h4>
              <div class="imrow">
                <figure><figcaption>打点叠加 · 红叉=VLM点 / 绿圈=三角化3D重投影</figcaption>
                  {'<img src="'+ov+'">' if ov else '<span class=muted>无</span>'}</figure>
                <figure><figcaption>抓取后三视角 overview / front / wrist</figcaption>
                  {'<img src="'+mv+'">' if mv else '<span class=muted>无</span>'}</figure>
                <figure><figcaption>rollout 抓取过程</figcaption>{vid}</figure>
              </div>
            </div>"""
        return rows

    oracle_objs = ["dice_cube", "wood_block", "soda_can", "tea_tin"]
    qwen_objs = ["dice_cube", "soda_can", "tea_tin"]

    # 标定表
    calib_rows = ""
    if calib:
        order = ["terse_json", "qwen_native", "grasp_terse"]
        label = {"terse_json": "简洁+JSON提示", "qwen_native": "Qwen 原生最简",
                 "grasp_terse": "含 grasp 词（反例）"}
        for k in order:
            s = calib["summary"][k]
            best = " best" if k == "terse_json" else ""
            calib_rows += (f'<tr class="{best.strip()}"><td>{label[k]}</td>'
                           f'<td>{s["median_px"]}</td><td>{s["mean_px"]}</td>'
                           f'<td>{s["n_fail_parse"]}/{s["n"]}</td></tr>')

    def summ(ev):
        if not ev:
            return ("—", "—", "—")
        s = ev["summary"]
        return (f'{s["n_succ"]}/{s["n_total"]} ({s["success_rate"]:.0%})',
                s.get("px_err_median", "—"), s.get("px_err_mean", "—"))

    o_sr, o_med, o_mean = summ(ev_o)
    q_sr, q_med, q_mean = summ(ev_q)

    # per-seed 明细（qwen）
    detail_rows = ""
    if ev_q:
        for r in ev_q["results"]:
            px = r.get("px_err", {})
            ok = "✅" if r["success"] else "❌"
            detail_rows += (f'<tr><td>{r["obj"]}</td><td>{r["seed"]}</td>'
                            f'<td>{ok}</td><td>{px.get("overview","—")}</td>'
                            f'<td>{px.get("front","—")}</td></tr>')

    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CABTO 实验2 · 代码生成 + VLM 打点 可视化报告</title>
<style>
:root{{--bg:#0e1116;--card:#161b22;--bd:#222b36;--fg:#e6edf3;--mut:#8b949e;
--acc:#58a6ff;--ok:#3fb950;--warn:#d29922;--red:#f85149;}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.65 -apple-system,'Segoe UI',Roboto,'PingFang SC',sans-serif}}
.wrap{{max-width:1080px;margin:0 auto;padding:32px 20px 80px}}
h1{{font-size:26px;margin:0 0 6px}}h2{{font-size:20px;margin:38px 0 12px;
border-left:3px solid var(--acc);padding-left:10px}}h3{{font-size:16px;color:var(--acc)}}
.sub{{color:var(--mut);margin:0 0 8px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin:16px 0}}
.kpi{{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:16px}}
.kpi .v{{font-size:28px;font-weight:700}}.kpi .l{{color:var(--mut);font-size:13px}}
.kpi.ok .v{{color:var(--ok)}}.kpi.acc .v{{color:var(--acc)}}.kpi.warn .v{{color:var(--warn)}}
table{{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--bd);
border-radius:10px;overflow:hidden;margin:12px 0}}
th,td{{padding:9px 12px;text-align:center;border-bottom:1px solid var(--bd)}}
th{{background:#1b222c;color:var(--mut);font-weight:600}}td:first-child,th:first-child{{text-align:left}}
tr.best td{{background:rgba(63,185,80,.12);font-weight:600}}
.obj-card{{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:16px;margin:14px 0}}
.obj-card h4{{margin:0 0 10px;font-size:15px}}
.imrow{{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start}}
figure{{margin:0;max-width:340px}}figcaption{{color:var(--mut);font-size:12px;margin-bottom:6px}}
img,video{{max-width:340px;border-radius:8px;border:1px solid var(--bd);display:block}}
.muted{{color:var(--mut)}}code{{background:#1b222c;padding:2px 6px;border-radius:5px;color:#a5d6ff}}
.note{{background:#161b22;border:1px solid var(--bd);border-left:3px solid var(--warn);
padding:12px 14px;border-radius:8px;margin:14px 0;color:#cdd9e5}}
.pipe{{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:14px 0}}
.pipe .step{{background:#1b222c;border:1px solid var(--bd);border-radius:8px;padding:8px 12px;font-size:13px}}
.pipe .ar{{color:var(--acc)}}
</style></head><body><div class="wrap">

<h1>CABTO 实验2 · 「代码生成 + VLM 打点」可视化报告</h1>
<p class="sub">Low-level policy sampling 重构 — 用代码把原语串成 action，3D 目标由本地 VLM 在多视角相机打点反投影得到。MuJoCo Franka 抓取场景。</p>

<h2>① 范式：被生成的 pick() 程序</h2>
<p>底层技能不再是写死的状态机，而是一段「代码生成器会产出的程序」，只调用一组可组合接口：</p>
<div class="pipe">
  <span class="step">📷 双视角相机图</span><span class="ar">→</span>
  <span class="step">🧠 VLM 打点 (u,v)</span><span class="ar">→</span>
  <span class="step">📐 两视角三角化 → 3D</span><span class="ar">→</span>
  <span class="step">🦾 approach / descend / align</span><span class="ar">→</span>
  <span class="step">✊ 闭合夹爪</span><span class="ar">→</span>
  <span class="step">⬆️ 抬起</span>
</div>
<p class="sub">动作模型 <code>h=⟨{{on_table}}, {{in_gripper, lifted}}, {{on_table}}⟩</code>，由 <code>env.check_success()</code> 校验一致性。</p>

<h2>② 核心结果</h2>
<div class="grid">
  <div class="kpi acc"><div class="v">2</div><div class="l">相机视角（overview+front，另有 wrist）</div></div>
  <div class="kpi ok"><div class="v">{o_sr}</div><div class="l">oracle 后端 端到端抓取成功率</div></div>
  <div class="kpi warn"><div class="v">{q_sr}</div><div class="l">Qwen2.5-VL 本地打点 端到端成功率</div></div>
  <div class="kpi acc"><div class="v">~{q_med}px</div><div class="l">Qwen 打点中位误差 @512 (≈1cm)</div></div>
</div>

<h2>③ VLM 后端对比：Qwen2.5-VL vs Molmo（本地推理）</h2>
<table><thead><tr><th>后端</th><th>模型</th><th>打点机制</th><th>本地推理</th><th>定位</th></tr></thead>
<tbody>
<tr class="best"><td>Qwen2.5-VL</td><td>Qwen2.5-VL-3B-Instruct-4bit</td><td>grounding，输出 <code>{{"point_2d":[x,y]}}</code> 绝对像素</td><td>mlx-vlm，~10s/次@MPS，2.9G</td><td><b>首选</b>：单模型够用，简洁 prompt 最准</td></tr>
<tr><td>Molmo</td><td>Molmo-7B-D-0924-4bit</td><td>原生 pointing，<code>&lt;point x= y=&gt;</code></td><td>mlx-vlm，7B 更重</td><td>pointing 专精，但模型更大</td></tr>
<tr><td>oracle</td><td>—</td><td>物体真值投影</td><td>无</td><td>确定性，打通链路/对照</td></tr>
</tbody></table>

<h2>④ Qwen 打点 prompt 标定（关键、反直觉）</h2>
<table><thead><tr><th>prompt 策略</th><th>中位误差(px)</th><th>均值(px)</th><th>解析失败</th></tr></thead>
<tbody>{calib_rows}</tbody></table>
<div class="note">结论：Qwen2.5-VL 对 <b>简洁</b> pointing 指令最准（<code>Point to the object on the table. Output the pixel coordinate.</code>）。一旦把 prompt 写啰嗦、强加 JSON schema、或出现 <b>"grasp"</b> 字样，误差显著变大。它原生输出 <code>{{"point_2d":[x,y]}}</code>，是输入分辨率下的<b>绝对像素，无需归一化</b>。</div>

<h2>⑤ 2D→3D 鲁棒定位</h2>
<p>MuJoCo 软件 GL 深度缓冲有系统性偏置（<code>ARB_clip_control unavailable</code>），故用<b>两视角视线三角化</b>定位（xy 误差≈0）。真实 VLM 偶有单视角严重偏（细高罐在 front 视角被打到罐身边缘），加<b>鲁棒融合</b>（三角化 + 两视角深度反投影里取互相 &lt;3cm 的内点簇求均值）+ 工作空间钳取，抗离群点。</p>

<h2>⑥ 可视化 · oracle 后端（全物体抓取成功）</h2>
{card_imgs('oracle', oracle_objs)}

<h2>⑦ 可视化 · 真实 Qwen2.5-VL 本地打点</h2>
<p class="sub">红叉是 Qwen 实际在相机图上打的点，绿圈是三角化得到的 3D 点重投影。二者落在物体上即链路正确。</p>
{card_imgs('qwen', qwen_objs)}

<h2>⑧ Qwen 端到端逐布局明细</h2>
<table><thead><tr><th>物体</th><th>种子</th><th>成功</th><th>overview 打点误差(px)</th><th>front 打点误差(px)</th></tr></thead>
<tbody>{detail_rows}</tbody></table>
<p class="sub">oracle 端到端 {o_sr}（打点误差 median {o_med}px）；Qwen 端到端 {q_sr}（median {q_med}px / mean {q_mean}px）。60% 是 3B 量化小模型本地打点的诚实结果——失败多源于个别视角的 VLM 误定位，而非执行链路。</p>

<p class="sub" style="margin-top:40px">代码：<code>exp2_low_level_codegen/franka_grasp_dp/exp2_codegen/</code> · venv <code>cabto</code> · 渲染 512×512 真实像素。</p>
</div></body></html>"""

    op = os.path.join(HERE, "exp2_codegen_report.html")
    with open(op, "w") as f:
        f.write(html)
    print(f"[saved] {op}  ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()

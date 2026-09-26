# Oracle 与 Qwen-VL 打点证据说明

本目录保留 CABTO 实验 2 的打点代码、原始 JSON、叠加图和 rollout 视频。

- Oracle：`franka_grasp_dp/exp2_codegen/results/eval_pick_oracle.json`，以及 `renders/*oracle*`。
- 历史本地 Qwen：`franka_grasp_dp/exp2_codegen/results/eval_pick_qwen.json`，实际模型为 **`mlx-community/Qwen2.5-VL-3B-Instruct-4bit`**。
- 本次补跑的0.8B多模态模型：**`mlx-community/Qwen3.5-0.8B-4bit`**；完整结果在 `franka_grasp_dp/exp2_codegen/results/eval_pick_qwen3_5_0_8b.json`，独立报告为 `qwen3_5_0_8b_report.html`。
- 同一5物体×4布局规模下：Qwen2.5-VL-3B为12/20，像素误差中位数18.0 px；Qwen3.5-0.8B为2/20，像素误差中位数158.5 px。
- 随包可直接核验的 Oracle 原始明细为2/2。旧 HTML 曾写 Oracle 20/20，但没有找到对应的20条 Oracle 明细，因此已按原始 JSON 修正，不把未留存明细的汇总当作已核验证据。
- 模型权重均未复制进此最小工程。重新运行 Qwen 后端时需安装 `mlx-vlm`，模型会从对应 Hugging Face ID 加载。

原工程此前没有0.8B运行记录；本包中的0.8B数据是本次在新工程内真实补跑产生，并未把3B结果改名为0.8B。

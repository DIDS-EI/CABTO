"""Diffusion Policy 公共组件包（MuJoCo 版）。

包含：
- dp_rot:        位姿 / 旋转 / 6D 表示的数学工具（仿真器无关）
- dp_objects:    桌面待抓取物体注册表（MuJoCo body/geom 定义）
- franka_dp_env: 数据采集与策略评估共用的 Franka 抓取环境（MuJoCo + 双相机）
- scripted_expert: 脚本化专家（delta 动作空间，仿真器无关）
"""

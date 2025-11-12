# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os

if os.environ.get("SIM_MODE", "Standalone") == "Standalone":
#     from ._extension.extension import *
# else:
    from isaacsim import SimulationApp

    class SimulationAppSingleton:
        _instance: SimulationApp = None
        
        @classmethod
        def get_instance(cls, config: dict = None) -> SimulationApp:
            if cls._instance is None:
                if config is None:
                    config = {"headless": False}
                cls._instance = SimulationApp(config)
            return cls._instance

    # 导出get_instance方法作为模块级别的函数
    def get_simulation_app():
        return SimulationAppSingleton.get_instance()

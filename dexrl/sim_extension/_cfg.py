# from jax._src.api import F
from re import T
from dexrl.utils.configclass import configclass

@configclass
class ExtSimCfg:

    
    # franka
    # ext_scenario_name = "FrankaExtScenario" 
    # ext_scenario_name = "RLFrankaExtScenario" 

    # cabto
    # ext_scenario_name = "FrankaFollowCubeExtScenario" # FrankaFollowCubeExtScenario FrankaGenerationExtScenario

    # ext_scenario_name = "MultiFrankaExtScenario" 
    # ext_scenario_name = "MultiFrankaCleanExtScenario" 
    ext_scenario_name = "MultiFrankaHandOverExtScenario"
    

    
    device = "cpu"
    # device = "cuda:0"
    default_show_ui = False
    enable_camera_view = True


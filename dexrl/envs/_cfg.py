from dexrl.utils.configclass import configclass

@configclass
class EnvCfg:
    # device = "cpu"
    device = "cuda:0"
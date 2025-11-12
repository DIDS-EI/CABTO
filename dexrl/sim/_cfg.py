from dexrl.utils.configclass import configclass

@configclass
class SimCfg:
    headless: bool = False
    device: str = "cuda:0"

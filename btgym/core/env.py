from btgym.utils.logger import set_logger_entry

from btgym.core.controller import Controller
from btgym.core.simulator import Simulator
from btgym.core.ui import UI
# Don't use GPU dynamics and use flatcache for performance boost
# gm.USE_GPU_DYNAMICS = True
# gm.ENABLE_FLATCACHE = True



class Env:
    """
    Demonstrates how to use the action primitives to pick and place an object in an empty scene.

    It loads Rs_int with a robot, and the robot picks and places a bottle of cologne.
    """

    def __init__(self,enable_ui=True):
        # Load the config
        self.sim = Simulator()
        self.controller = Controller(self.sim)
        self.enable_ui = enable_ui
        if enable_ui:
            self.ui = UI(self.controller,self.sim)

    def run(self):
        while True:
            self.step()

    def step(self):
        self.sim.step()
        self.controller.step()
        if self:
            self.ui.step()




if __name__ == "__main__":
    set_logger_entry(__file__)
    env = Env()
    env.run()


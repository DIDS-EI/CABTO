import time
import pygame
import pygame_gui
from btgym.utils.logger import log
# import matplotlib
# matplotlib.use('Agg')

from btgym.utils.path import ROOT_PATH

valid_task_list = open(f"{ROOT_PATH}/assets/tasks_valid.txt", "r").read().splitlines()
valid_task_list_with_index = []
for i in range(len(valid_task_list)):
    valid_task_list_with_index.append(f"{i}: {valid_task_list[i]}")

class UI:
    def __init__(self, controller=None, simulator=None):
        self.controller = controller
        self.simulator = simulator

        pygame.init()

        window_size = (1000, 600)
        self.screen = pygame.display.set_mode(window_size)
        self.manager = pygame_gui.UIManager(window_size)

        self.clock = pygame.time.Clock()
        self.last_step_time = time.time()
        self.last_log_time = time.time()
        self.step_interval = 1/60

        # create task list
        self.task_list_box = pygame_gui.elements.UISelectionList(
            relative_rect=pygame.Rect((50, 50), (300, 400)),
            item_list=valid_task_list_with_index,
            manager=self.manager,
            allow_multi_select=False
        )

        # create status display area
        self.status_text = pygame_gui.elements.UITextBox(
            relative_rect=pygame.Rect((400, 50), (500, 500)),
            html_text="Select a task",
            manager=self.manager
        )

        # create control button
        self.start_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((50, 470), (100, 50)),
            text='Start Task',
            manager=self.manager
        )

    def init_trav_map(self):
        trav_map = self.simulator.get_trav_map()
        self.trav_map = trav_map


    def step(self):
        current_time = time.time()
        if current_time - self.last_step_time < self.step_interval:
            return
        self.last_step_time = current_time

        self.handle_events()
        # self.render()
        self.update_info()
        # self.log_step_info()

    def update_info(self):
        self.status_text.set_text(f"{self.controller.goal_status_str}")

    def log_step_info(self):
        current_time = time.time()
        if current_time - self.last_log_time < self.step_log_interval:
            return
        self.last_log_time = current_time
        # log(f"UI info: {self.info}")


    def handle_events(self):
        time_delta = self.clock.tick(60)/1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.USEREVENT:
                if event.user_type == pygame_gui.UI_SELECTION_LIST_NEW_SELECTION:
                    if event.ui_element == self.task_list_box:
                        selected_task = event.text.split(":")[1].strip()
                        self.simulator.load_behavior_task_by_name(selected_task)
                        self.controller.do_task(selected_task)

            self.manager.process_events(event)

        self.manager.update(time_delta)
    
        self.screen.fill((255, 255, 255))
        self.manager.draw_ui(self.screen)

        pygame.display.update()

    # def on_submit(self):
    #     text = self.text_entry.text
    #     log(f"UI: Button Submit {text}")

    #     try:
    #         task_index = int(text)
    #         self.simulator.load_behavior_task_by_index(task_index)
    #     except:
    #         if text == '':
    #             task_index = 0
    #         else:
    #             self.simulator.load_behavior_task_by_name(text)
    #     # self.simulator.load_task_by_index(task_index)
    #     self.controller.reset()
    #     self.controller.do_task(text)


if __name__ == "__main__":
    ui = UI()
    while True:
        ui.step()

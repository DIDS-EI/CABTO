import sys
from btgym.utils.path import ROOT_PATH
from btgym.dataclass.cfg import cfg
from btgym.utils.patches import apply_all_patches

apply_all_patches()

sys.path.append(f'{ROOT_PATH}/simulator')

def get_activity_list():
    with open(f'{ROOT_PATH}/assets/task_names.txt', 'r') as f:
        task_names = f.read().splitlines()
    return task_names


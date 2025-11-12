# from btgym.behavior_tree.scene.scene import Scene
# from btgym.behavior_tree.behavior_tree.btml.btmlCompiler import load

import os
from btgym.behavior_tree.utils.draw import render_dot_tree
from btgym.behavior_tree.behavior_trees.BehaviorTree import BehaviorTree
from pathlib import Path
if __name__ == '__main__':
    TASK_NAME = 'OT'

    # create robot
    DIR = Path(__file__).parent
    btml_path = os.path.join(DIR, 'Default.ptml')
    behavior_lib_path = os.path.join(DIR, 'exec_lib')

    bt = BehaviorTree(btml_path, behavior_lib_path)


    render_dot_tree(bt.root,name="llm_test")
    # build and tick
    # scene.BT = ptree.trees.BehaviourTree(scene.BT)
    # todo: tick this behavior_tree
    print(bt)
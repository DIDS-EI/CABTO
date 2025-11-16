from btgym.behavior_tree.base_nodes import Action
from btgym.behavior_tree import Status
from btgym.behavior_tree.behavior_trees import BehaviorTree

class OGAction(Action):
    can_be_expanded = True
    num_args = 1

    CanGrasp = {"apple", 'wine'}
    CanWalkTo = {"table"}

    AllObject = CanGrasp | CanWalkTo

    def __init__(self, *args):
        super().__init__(*args)
        self.args = args
        self.info = self.get_info(*args)

    @classmethod
    def get_info(cls, *arg):
        raise NotImplementedError

    def change_condition_set(self, agent):
        agent.condition_set |= self.info["add"]
        agent.condition_set -= self.info["del_set"]

    @property
    def action_class_name(self):
        return self.__class__.__name__

    def update(self) -> Status:
        return Status.RUNNING
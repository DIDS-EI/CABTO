import copy

from tabulate import tabulate
import numpy as np
import random

# from OptimalBTExpansionAlgorithm import generate_random_state,state_transition
# from OptimalBTExpansionAlgorithm import Action,OptBTExpAlgorithm
# from BTExpansionAlgorithm import BTExpAlgorithm # 调用最优行为树扩展算法


import time
np.random.seed(1)
random.seed(1)


#定义行动类，行动包括前提、增加和删除影响
class Action:
    def __init__(self,name='anonymous action',pre=set(),add=set(),del_set=set(),cost=1,vaild_num=0,vild_args=set()):
        self.pre=copy.deepcopy(pre)
        self.add=copy.deepcopy(add)
        self.del_set=copy.deepcopy(del_set)
        self.name=name
        self.cost=cost
        self.vaild_num=vaild_num
        self.vild_args = vild_args

    def __str__(self):
        return self.name
    # 从状态随机生成一个行动
    def generate_from_state(self,state,num):
        for i in range(0,num):
            if i in state:
                if random.random() >0.5:
                    self.pre.add(i)
                    if random.random() >0.5:
                        self.del_set.add(i)
                    continue
            if random.random() > 0.5:
                self.add.add(i)
                continue
            if random.random() >0.5:
                self.del_set.add(i)

    # def generate_from_state_local(self,literals_num_set):
    #     # pre_num = random.randint(0, min(pre_max, len(state)))
    #     # self.pre = set(np.random.choice(list(state), pre_num, replace=False))
    #     #
    #     # add_set = literals_num_set - self.pre
    #     # add_num = random.randint(0, len(add_set))
    #     # self.add = set(np.random.choice(list(add_set), add_num, replace=False))
    #     #
    #     # del_set = literals_num_set - self.add
    #     # del_num = random.randint(0, len(del_set))
    #     # self.del_set = set(np.random.choice(list(del_set), del_num, replace=False))
    #
    #     pre_num = random.randint(0, len(state))
    #     self.pre = set(random.sample(state, pre_num))
    #
    #     add_set = literals_num_set - self.pre
    #     add_num = random.randint(0, len(add_set))
    #     self.add = set(random.sample(add_set, add_num))
    #
    #     del_set = literals_num_set - self.add
    #     del_num = random.randint(0, len(del_set))
    #     self.del_set = set(random.sample(del_set, del_num))


    def generate_from_state_local(self,state,literals_num_set,all_obj_set=set(),obj_num=0,obj=None):
        # pre_num = random.randint(0, min(pre_max, len(state)))
        # self.pre = set(np.random.choice(list(state), pre_num, replace=False))
        #
        # add_set = literals_num_set - self.pre
        # add_num = random.randint(0, len(add_set))
        # self.add = set(np.random.choice(list(add_set), add_num, replace=False))
        #
        # del_set = literals_num_set - self.add
        # del_num = random.randint(0, len(del_set))
        # self.del_set = set(np.random.choice(list(del_set), del_num, replace=False))

        pre_num = random.randint(0, len(state))
        self.pre = set(random.sample(state, pre_num))

        add_set = literals_num_set - self.pre
        add_num = random.randint(0, len(add_set))
        self.add = set(random.sample(add_set, add_num))

        del_set = literals_num_set - self.add
        del_num = random.randint(0, len(del_set))
        self.del_set = set(random.sample(del_set, del_num))

        if all_obj_set!=set():
            self.vaild_num = random.randint(1, obj_num-1)
            self.vild_args = (set(random.sample(all_obj_set, self.vaild_num)))
            if obj!=None:
                self.vild_args.add(obj)
                self.vaild_num = len(self.vild_args)

    def update(self,name,pre,del_set,add):
        self.name = name
        self.pre = pre
        self.del_set = del_set
        self.add = add
        return self


    def print_action(self):
        print (self.pre)
        print(self.add)
        print(self.del_set)



#生成随机状态
def generate_random_state(num):
    result = set()
    for i in range(0,num):
        if random.random()>0.5:
            result.add(i)
    return result

#从状态和行动生成后继状态
def state_transition(state,action):
    if not action.pre <= state:
        print ('error: action not applicable')
        return state
    new_state=(state | action.add) - action.del_set
    return new_state





def print_action_data_table(goal,start,actions):
    data = []
    for a in actions:
        data.append([a.name ,a.pre ,a.add ,a.del_set ,a.cost])
    data.append(["Goal" ,goal ," " ,"Start" ,start])
    print(tabulate(data, headers=["Name", "Pre", "Add" ,"Del" ,"Cost"], tablefmt="fancy_grid"))  # grid plain simple github fancy_grid



# def BTTest(bt_algo_opt=True,seed=1,literals_num=10,depth=10,iters=10,total_count=1):
#
#     if bt_algo_opt:
#         print("============= OptBT Test ==============")
#     else:
#         print("============= XiaoCai BT Test ==============")
#     random.seed(seed)
#     # 设置生成规划问题集的超参数：文字数、解深度、迭代次数
#     literals_num=literals_num
#     depth = depth
#     iters= iters
#     total_tree_size = []
#     total_action_num = []
#     total_state_num = []
#     total_steps_num=[]
#     total_cost=[]
#     total_tick=[]
#     #fail_count=0
#     #danger_count=0
#     success_count =0
#     failure_count = 0
#     planning_time_total = 0.0
#
#     error = False
#
#     # 实验1000次
#     for count in range (total_count):
#
#         action_num = 1
#
#         # 生成一个规划问题，包括随机的状态和行动，以及目标状态
#         states = []
#         actions = []
#         start = generate_random_state(literals_num)
#         state = copy.deepcopy(start)
#         states.append(state)
#         #print (state)
#         # for k in range(10):
#         for i in range (0,depth):
#             a = Action()
#             a.generate_from_state(state,literals_num)
#             a.cost = random.randint(1, 100)
#             if not a in actions:
#                 a.name = "a"+str(action_num)
#                 action_num+=1
#                 actions.append(a)
#             state = state_transition(state,a)
#             if state in states:
#                 pass
#             else:
#                 states.append(state)
#                 #print(state)
#
#         goal = states[-1]
#         state = copy.deepcopy(start)
#         for i in range (0,iters):
#             a = Action()
#             a.generate_from_state(state,literals_num)
#             if not a in actions:
#                 a.name = "a"+str(action_num)
#                 action_num+=1
#                 actions.append(a)
#             state = state_transition(state,a)
#             if state in states:
#                 pass
#             else:
#                 states.append(state)
#             state = random.sample(states,1)[0]
#
#         # 选择测试本文算法btalgorithm，或对比算法weakalgorithm
#
#         if bt_algo_opt:
#             # if count==874:
#             #     algo = OptBTExpAlgorithm(verbose=False)
#             # else:
#             algo = OptBTExpAlgorithm(verbose=False)
#         else:
#             algo = BTExpAlgorithm(verbose=False)
#         algo.clear()
#
#         #algo = Weakalgorithm()
#         start_time = time.time()
#         if count ==  0 : #874:
#             print_action_data_table(goal, start, list(actions))
#         # print_action_data_table(goal, start, list(actions))
#         if algo.run_algorithm_test(start, goal, actions):#运行算法，规划后行为树为algo.bt
#             total_tree_size.append( algo.bt.count_size()-1)
#             # if count==0:
#             #     algo.print_solution()
#             # algo.print_solution()  # 打印行为树
#         else:
#             print ("error")
#         end_time = time.time()
#         planning_time_total += (end_time-start_time)
#
#         #开始从初始状态运行行为树，测试
#         state=start
#         steps=0
#         current_cost = 0
#         current_tick_time=0
#         val, obj, cost, tick_time = algo.bt.cost_tick(state,0,0)#tick行为树，obj为所运行的行动
#
#         current_tick_time+=tick_time
#         current_cost += cost
#         while val !='success' and val !='failure':#运行直到行为树成功或失败
#             print(state, obj)
#             state = state_transition(state,obj)
#             val, obj,cost, tick_time = algo.bt.cost_tick(state,0,0)
#
#             current_cost += cost
#             current_tick_time += tick_time
#             if(val == 'failure'):
#                 print("bt fails at step",steps)
#                 error = True
#                 break
#             steps+=1
#             if(steps>=500):#至多运行500步
#                 break
#         if not goal <= state:#错误解，目标条件不在执行后状态满足
#             #print ("wrong solution",steps)
#             failure_count+=1
#             error = True
#         else:#正确解，满足目标条件
#             #print ("right solution",steps)
#             success_count+=1
#             total_steps_num.append(steps)
#         if error:
#             print_action_data_table(goal, start, list(actions))
#             algo.print_solution()
#             break
#
#         print("step:",steps)
#         algo.clear()
#         total_action_num.append(len(actions))
#         total_state_num.append(len(states))
#         total_cost.append(current_cost)
#         total_tick.append(current_tick_time)
#
#     print("success:",success_count,"failure:",failure_count)#算法成功和失败次数
#     print("Total Tree Size: mean=",np.mean(total_tree_size), "std=",np.std(total_tree_size, ddof=1))#1000次测试树大小
#     print("Total Steps Num: mean=",np.mean(total_steps_num),"std=",np.std(total_steps_num,ddof=1))
#     print("Average Number of States:",np.mean(total_state_num))#1000次问题的平均状态数
#     print("Average Number of Actions",np.mean(total_action_num))#1000次问题的平均行动数
#     print("Planning Time Total:",planning_time_total,planning_time_total/total_count)
#     print("Average Number of Ticks", np.mean(total_tick),"std=",np.std(total_tick,ddof=1))
#     print("Average Cost of Execution:", np.mean(total_cost),"std=",np.std(total_cost,ddof=1))
#     # print(total_steps_num) 第21个
#     if bt_algo_opt:
#         print("============= End OptBT Test ==============")
#     else:
#         print("============= End XiaoCai BT Test ==============")

    # xiao cai
    # success: 1000 failure: 0
    # Total Tree Size: mean= 35.303 std= 29.71336526001515
    # Total Steps Num: mean= 1.898 std= 0.970844240101644
    # Average number of states: 20.678
    # Average number of actions 20.0
    # Planning Time Total: 0.6280641555786133 0.0006280641555786133

    # our start
    # success: 1000 failure: 0
    # Total Tree Size: mean= 17.945 std= 12.841997192488865
    # Total Steps Num: mean= 1.785 std= 0.8120556843187752
    # Average number of states: 20.678
    # Average number of actions 20.0
    # Planning Time Total: 1.4748523235321045 0.0014748523235321046

    # our
    # success: 1000 failure: 0
    # Total Tree Size: mean= 48.764 std= 20.503626574406358
    # Total Steps Num: mean= 1.785 std= 0.8120556843187752
    # Average number of states: 20.678
    # Average number of actions 20.0
    # Planning Time Total: 3.3271877765655518 0.0033271877765655516


def get_act_start_goal(seed=1,literals_num=10,depth=10,iters=10,total_count=1000):
        act_list=[]
        start_list=[]
        goal_list=[]

        for count in range(total_count):
            # 生成一个规划问题，包括随机的状态和行动，以及目标状态
            action_num=1
            states = []
            actions = []
            start = generate_random_state(literals_num)
            state = copy.deepcopy(start)
            states.append(state)
            # print (state)
            for k in range(int(iters/5)):
                state = copy.deepcopy(start)
                for i in range(0, depth):
                    a = Action()
                    a.generate_from_state(state, literals_num)
                    a.cost = random.randint(1, 100)
                    if not a in actions:
                        a.name = "a" + str(action_num)
                        action_num += 1
                        actions.append(a)
                    state = state_transition(state, a)
                    if state in states:
                        pass
                    else:
                        states.append(state)
                        # print(state)

            goal = states[-1]
            state = copy.deepcopy(start)
            for i in range(0, int(iters/5)):
                a = Action()
                a.generate_from_state(state, literals_num)
                if not a in actions:
                    a.name = "a" + str(action_num)
                    action_num += 1
                    actions.append(a)
                state = state_transition(state, a)
                if state in states:
                    pass
                else:
                    states.append(state)
                state = random.sample(states, 1)[0]

            act_list.append(actions)
            start_list.append(start)
            goal_list.append(goal)
            # print("action:",len(actions))
        return act_list, start_list, goal_list


def cal_tree_cond_tick(start,goal,bt):
    # 开始从初始状态运行行为树，测试
    # print("start:",start)
    # print("goal:", goal)


    state = start
    error=False
    current_cost = 0
    current_cond_tick_time = 0
    # val, obj, cost, tick_time = algo.bt.cost_tick(state,0,0)#tick行为树，obj为所运行的行动
    val, obj, cost, tick_time, cond_times = bt.cost_tick_cond(state, 0, 0, 0)
    # print(state, val, obj, cost)
    current_cond_tick_time += cond_times
    current_cost += cost
    while val != 'success' and val != 'failure':  # 运行直到行为树成功或失败
        state = state_transition(state, obj)
        # val, obj,cost, tick_time = algo.bt.cost_tick(state,0,0)
        val, obj, cost, tick_time, cond_times = bt.cost_tick_cond(state, 0, 0, 0)
        # print(val,obj)
        current_cost += cost
        current_cond_tick_time += cond_times
        if (val == 'failure'):
            error = True
            break
    if not goal <= state:  # 错误解，目标条件不在执行后状态满足
        error = True
    if error:
        print("Merge Error")
    return error,current_cost,current_cond_tick_time



def conflict(c):
    have_at = False
    for str in c:
        if 'Not' not in str and 'RobotNear' in str:
            if have_at:
                return True
            have_at = True

    Holding = False
    HoldingNothing = False
    for str in c:
        if 'Not ' not in str and 'Holding(Nothing)' in str: # 注意 'Not ' in 'Nothing'
            HoldingNothing = True
        if 'Not' not in str and 'Holding(Nothing)' not in str and 'Holding' in str:
            if Holding:
                return True
            Holding = True
        if HoldingNothing and Holding:
            return True
    return False

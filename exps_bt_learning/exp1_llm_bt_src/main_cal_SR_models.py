import os
DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from exps_bt_learning.tools import parse_bddl
from exps_bt_learning.llm_generate_lib_func import llm_generate_behavior_lib
from exps_bt_learning.validate_bt_fun import validate_bt_fun

task2name = {
    "task1":"PlaceApple",
    "task2":"ActivateLights",
    "task3":"PutInDrawer",
    "task4":"HomeRearrangement",
    "task5":"MealPreparation",
    "task6":"aaa_demo0_draw6"
}
task2objects = {
    "task1":['apple','coffeetable'],
    "task2":['light1','light2'],
    "task3":['pen','cabinet'],
    "task4":['apple','coffeetable','pen','cabinet'],
    "task5":['oven','chickenleg','apple','coffeetable'],
    # "task6":['cake','microwave',"yard_table","oven"]
}
task2start_state = {
    "task1":{'IsHandEmpty()'},
    "task2":{'IsHandEmpty()','ToggledOff(light1)','ToggledOn(light2)'},
    "task3":{'IsHandEmpty()','IsClose(cabinet)'},
    "task4":{'IsHandEmpty()','IsOpen(cabinet)','In(pen,cabinet)'},
    "task5":{'IsHandEmpty()','IsOpen(oven)','ToggledOff(oven)'},
    # "task6":{'IsHandEmpty()','IsOpen(microwave)','ToggledOff(oven)'}
}
task2goal_str = {
    "task1":'On(apple,coffeetable)',
    "task2":'ToggledOn(light1) & ToggledOff(light2)',
    "task3":'In(pen,cabinet)',
    "task4":'On(pen,coffeetable) & IsClose(cabinet) & On(apple,coffeetable)', #In(apple,cabinet) & Closed(cabinet) & 
    "task5":'IsClose(oven) & ToggledOn(oven) & On(apple,coffeetable) & In(chickenleg,oven)', #& On(apple,coffee_table) & On(chicken_leg,coffee_table)
    # "task6":'On(cake,yard_table) & IsClose(microwave) & ToggledOn(oven)'
}

total_try_times = 10

# model = "gpt-4o"
model = "gpt-3.5-turbo"
# model = "gpt-4o-mini"
latex_data = {}
model_ls = ["gpt-3.5-turbo"]#,"gpt-4o-mini","gpt-4o"
for model in model_ls:
    latex_data[model] = {}

    for task_id in range(1,6):

        # 1. set task
        task_name = f"task{task_id}"

        bddl_file = os.path.join(DIR,f"tasks/{task_name}/problem0.bddl")
        behavior_lib_path = os.path.join(DIR,f"tasks/{task_name}/exec_lib")  # os.path.join(DIR,"../exec_lib")
        output_dir = os.path.join(DIR,f"tasks/{task_name}/bt.btml")


        # objects, start_state, goal = parse_bddl(bddl_file)
        # goal_str = ' '.join(goal) # convert goal to string
        objects, start_state, _ = parse_bddl(bddl_file)
        
        # start_state = task2start_state[task_name]
        # objects = set(task2objects[task_name])
        # goal_str = task2goal_str[task_name]
        
        objects.update(task2objects[task_name])
        start_state.update(task2start_state[task_name])
        goal_str = task2goal_str[task_name]
        print("objects:",objects)
        print("start_state:",start_state)
        print("goal_str:",goal_str)

        # 2. run experiment
        
        success_times = 0

        # create result directory and result csv
        result_dir = os.path.join(DIR,"results")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        dataframe_path = os.path.join(result_dir,f"exp1_{task_name}_success_rate_{total_try_times}_{model}.csv")
        table_data = []

        for i in range(total_try_times):
            print(f"try {i+1} times")
            # 1. generate behavior lib
            llm_generate_behavior_lib(bddl_file=bddl_file,goal_str=goal_str,objects=objects,start_state=start_state,\
                behavior_lib_path=behavior_lib_path,model=model)
            # 2. validate behavior lib 
            print("Validate behavior lib...")
            try:
                error,bt,expanded_num,act_num,record_act_ls,ptml_string = validate_bt_fun(behavior_lib_path=behavior_lib_path, goal_str=goal_str,cur_cond_set=start_state,output_dir=output_dir)
                if error == 0:
                    success_times += 1
                # break # success then break loop
            except Exception as e:
                error=True
                act_num=-1
                expanded_num=-1
                print(f"error: {e}")
                
            # output generated action lib number and condition lib number
            action_lib_num = len(os.listdir(os.path.join(behavior_lib_path,'Action')))
            condition_lib_num = len(os.listdir(os.path.join(behavior_lib_path,'Condition')))
            print(f"action lib num: {action_lib_num}")
            print(f"condition lib num: {condition_lib_num}")
            # save each result to table
            # table columns: task name, try times, success times, action lib number, condition lib number, success or not
            # table rows: each try
            table_data.append([task_name,i+1,action_lib_num,condition_lib_num,expanded_num,act_num,not error])
        
        # output success rate
        # output success rate as percentage
        # output success rate/total try times
        print(f"success rate: {success_times/total_try_times*100:.2f}%") 
        print(f"success rate/total try times: {success_times}/{total_try_times}")
        
        # last column: average value of action_lib_num,condition_lib_num,expanded_num,act_num
        # average value only calculate success times
        # if all are 0, then all are 0
        if success_times == 0:
            action_lib_num_avg = 0
            condition_lib_num_avg = 0
            expanded_num_avg = 0
            act_num_avg = 0
        else:
            action_lib_num_avg = sum([row[2] for row in table_data if row[6]]) / success_times
            condition_lib_num_avg = sum([row[3] for row in table_data if row[6]]) / success_times
            expanded_num_avg = sum([row[4] for row in table_data if row[6]]) / success_times
            act_num_avg = sum([row[5] for row in table_data if row[6]]) / success_times
        table_data.append([task_name,-1,action_lib_num_avg,condition_lib_num_avg,expanded_num_avg,act_num_avg,not error])
        
        # save table data to csv file, english title
        df = pd.DataFrame(table_data, columns=['task_name', 'try_times', 'action_lib_num', 'condition_lib_num','expanded_num','act_num','success_or_not'])
        df.to_csv(dataframe_path, index=False)
        
        # record the result of this model and task
        latex_data[model][task_name] = [action_lib_num_avg,condition_lib_num_avg,expanded_num_avg,act_num_avg,f"{success_times}/{total_try_times}"]

task_names2type = {"task1":"Pick \& Place","task2":"ToggleOn \& ToggleOff","task3":"Open \& PutIn",
                   "task4":"Home Rearrangement","task5":"Meal Preparation"}
rows = ""
for task_name in ["task1","task2","task3","task4","task5"]:
    type_name = task_names2type[task_name]
    len_model_ls = len(model_ls)
    for j,model in enumerate(model_ls):
        if j == 0:
            rows += f"{type_name} & "
        rows += " & ".join(str(x) for x in latex_data[model][task_name])
        if j == len_model_ls-1:
            rows += r" \\"+ "\n"
        else:
            rows += " & "
            
# save rows to txt file
with open(os.path.join(DIR,"results","exp1_SR_models.txt"),"w") as f:
    f.write(rows)
    
print("========================================")
print(rows)
print("========================================")
print("done")


    
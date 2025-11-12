# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for file I/O with pickle."""

import os
import pickle
from typing import Any


def load_pickle(filename: str) -> Any:
    """Loads an input PKL file safely.

    Args:
        filename: The path to pickled file.

    Raises:
        FileNotFoundError: When the specified file does not exist.

    Returns:
        The data read from the input file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def dump_pickle(filename: str, data: Any):
    """Saves data into a pickle file safely.

    Note:
        The function creates any missing directory along the file's path.

    Args:
        filename: The path to save the file at.
        data: The data to save.
    """
    # check ending
    if not filename.endswith("pkl"):
        filename += ".pkl"
    # create directory
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    # save data
    with open(filename, "wb") as f:
        pickle.dump(data, f)

# def save_pkl_data(data, floder_name,filename):
#     """
#     保存数据到pkl文件
#     """
#     # 保存 joint_pos_list
#     floder_name = f"{global_config.root_path}/outputs/pkl/bt_data_{max_transition_data_num}"
#     if not os.path.exists(floder_name):
#         os.makedirs(floder_name)
    
#     with open(f"{floder_name}/demo_data_20.pkl", "wb") as f:
#         pickle.dump(transition_data_list, f)
#     print(f"saved transition_data_list to {floder_name}/demo_data_20.pkl")
#     # 输出成功率
#     success_rate = transition_data_num / try_num
#     print(f"success_rate: {success_rate}")
    
    
#     for transition_data in transition_data_list:
#         if transition_data["task_stage"] == 0:
#             with open(f"{floder_name}/act_0.pkl", "wb") as f:
#                 pickle.dump(transition_data, f)
#         elif transition_data["task_stage"] == 1:
#             with open(f"{floder_name}/act_1.pkl", "wb") as f:
#                 pickle.dump(transition_data, f)
        
#         transition_data["actions"] = transition_data["task_stage"]
#         with open(f"{floder_name}/cond_0.pkl", "wb") as f:
#             pickle.dump(transition_data, f)
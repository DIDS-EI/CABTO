#!/usr/bin/env bash

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#==
# Configurations
#==

# Exits if error occurs
set -e

# Set tab-spaces
tabs 4

#==
# Helper functions
#==


current_folder=$(cd "$(dirname "$0")";pwd)
ini_path=$current_folder/../global_config.ini
root_path=$(cd "$(dirname "$0")";cd ..;pwd)

echo current folder: $current_folder
echo ini_path: $ini_path

# 定义读取INI文件的函数
read_ini() {
    local section=$1
    local key=$2
    local file=$3

    # 使用 awk 读取并去除空格
    awk -F '=' -v section="\\[$section\\]" -v key="$key" '
    BEGIN { RS=""; FS="\n"; ORS="" }
    $0 ~ section {
        for (i=1; i<=NF; i++) {
            if ($i ~ key) {
                split($i, a, "=")
                # 去除值两端的空格
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", a[2])
                print a[2]
                exit
            }
        }
    }
    ' "$file"
}


isaacsim_path=$(read_ini simulation isaacsim_path $ini_path)
isaac_path=$isaacsim_path
echo isaacsim_path: $isaacsim_path
python_exe=${CONDA_PREFIX}/bin/python
conda_env_name=$(conda info --envs | grep '*' | awk '{print $1}' | tail -1)
echo conda_env_name: $conda_env_name

# setup anaconda environment for Isaac Lab
setup_conda_env() {
    # get environment name from input
    local env_name=$1
    # check conda is installed
    if ! command -v conda &> /dev/null
    then
        echo "[ERROR] Conda could not be found. Please install conda and try again."
        exit 1
    fi

    # check if the environment exists
    # if { conda env list | grep -w ${env_name}; } >/dev/null 2>&1; then
    #     echo -e "[INFO] Conda environment named '${env_name}' already exists."
    # else
    #     echo -e "[INFO] Creating conda environment named '${env_name}'..."
    #     conda create -y --name ${env_name} python=3.10
    # fi

    # cache current paths for later
    cache_pythonpath=$PYTHONPATH
    cache_ld_library_path=$LD_LIBRARY_PATH
    # clear any existing files
    rm -f ${CONDA_PREFIX}/etc/conda/activate.d/setenv.sh
    rm -f ${CONDA_PREFIX}/etc/conda/deactivate.d/unsetenv.sh
    # activate the environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate ${env_name}
    # setup directories to load Isaac Sim variables
    mkdir -p ${CONDA_PREFIX}/etc/conda/activate.d
    mkdir -p ${CONDA_PREFIX}/etc/conda/deactivate.d

    # check if we have _isaac_sim directory -> if so that means binaries were installed.
    # we need to setup conda variables to load the binaries
    local isaacsim_setup_conda_env_script="${isaacsim_path}/setup_conda_env.sh"

    if [ -f "${isaacsim_setup_conda_env_script}" ]; then
        # add variables to environment during activation
        printf '%s\n' \
            '# for Isaac Sim' \
            'source '${isaacsim_setup_conda_env_script}'' \
            '' >> ${CONDA_PREFIX}/etc/conda/activate.d/setenv.sh
    fi

    # reactivate the environment to load the variables
    # needed because deactivate complains about Isaac Lab alias since it otherwise doesn't exist
    conda activate ${env_name}

    # remove variables from environment during deactivation
    printf '%s\n' '#!/usr/bin/env bash' '' \
        '' \
        '# restore paths' \
        'export PYTHONPATH='${cache_pythonpath}'' \
        'export LD_LIBRARY_PATH='${cache_ld_library_path}'' \
        '' \
        '# for Isaac Sim' \
        'unset RESOURCE_NAME' \
        '' > ${CONDA_PREFIX}/etc/conda/deactivate.d/unsetenv.sh

    # check if we have _isaac_sim directory -> if so that means binaries were installed.
    if [ -f "${isaacsim_setup_conda_env_script}" ]; then
        # add variables to environment during activation
        printf '%s\n' \
            '# for Isaac Sim' \
            'unset CARB_APP_PATH' \
            'unset EXP_PATH' \
            'unset ISAAC_PATH' \
            '' >> ${CONDA_PREFIX}/etc/conda/deactivate.d/unsetenv.sh
    fi

    # install some extra dependencies
    # echo -e "[INFO] Installing extra dependencies (this might take a few minutes)..."
    # conda install -c conda-forge -y importlib_metadata &> /dev/null

    # deactivate the environment
    conda deactivate
}

setup_conda_env ${conda_env_name}


echo "configuring VSCode environment, so that you can jump to the python code in isaacsim"

python ${current_folder}/setup_vscode.py

mkdir -p ${root_path}/sim_extension/dexrl.sim_extension/dexrl
ln -sf ${root_path}/dexrl/sim_extension  ${root_path}/sim_extension/dexrl.sim_extension/dexrl/
ln -sf ${root_path}/dexrl/sim  ${root_path}/sim_extension/dexrl.sim_extension/dexrl/

echo "installation completed!!!"

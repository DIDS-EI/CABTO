#!/bin/bash

current_folder=$(cd "$(dirname "$0")";pwd)
ini_path=$current_folder/global_config.ini

echo current folder: $current_folder
echo ini_path: $ini_path

# 定义读取INI文件的函数
read_ini() {
    local section=$1
    local key=$2
    local file=$3

    # 使用awk读取指定节和键的值
    awk -F '=' -v section="\\[$section\\]" -v key="$key" '
    BEGIN { RS=""; FS="\n"; ORS="" }
    $0 ~ section {
        for (i=1; i<=NF; i++) {
            if ($i ~ key) {
                split($i, a, "=")
                print a[2]
                exit
            }
        }
    }
    ' "$file"
}

isaacsim_path=$(read_ini simulation isaacsim_path $ini_path)

echo ISAACSIM_PATH: $isaacsim_path
cd $isaacsim_path

# 初始化变量
scenario="DexScenario"
use_extension=false

# 解析参数
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -e)
            use_extension=true
            shift
            ;;
        *)
            echo "use default scenario: $scenario"
            shift
            ;;
    esac
done

if [ "$use_extension" = true ]; then
    echo "Starting Isaac Sim with extension..."

    export SIM_MODE="Extension"
    export XDG_DATA_HOME=/home/cys/.local/share
    ./isaac-sim.sh --/isaac/startup/ros_bridge_extension= --/rtx/ecoMode/enabled=True \
    --ext-folder $current_folder/sim_extension \
    --enable dexrl.sim_extension \
    # --/physics/suppressReadback=True \
    # --/physics/cudaDevice=0
else
    echo "Starting original Isaac Sim..."
    export XDG_DATA_HOME=/home/cys/.local/share
    ./isaac-sim.sh --/isaac/startup/ros_bridge_extension= --/rtx/ecoMode/enabled=True
fi


import os
import configparser

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
assets_path = os.path.join(root_path, "assets")
config_ini = configparser.ConfigParser()
config_ini.read(os.path.join(root_path, "global_config.ini"))

isaacsim_path = config_ini["simulation"]["isaacsim_path"]
data_path = config_ini["simulation"]["data_path"]
model_path = config_ini["simulation"]["model_path"]

isaacsim_data_path = os.path.join(data_path, "Assets/IsaacSim/Assets/Isaac/4.2/Isaac/")


left_arm_ip = config_ini["real"]["left_arm_ip"]
right_arm_ip = config_ini["real"]["right_arm_ip"]
left_hand_port = config_ini["real"]["left_hand_port"]
right_hand_port = config_ini["real"]["right_hand_port"]

top_camera_id = config_ini["real"].get("top_camera_id")
left_camera_id = config_ini["real"].get("left_camera_id")
right_camera_id = config_ini["real"].get("right_camera_id")

# api_key = config_ini["api"]["api_key"]
# base_url = config_ini["api"]["base_url"]

if __name__ == "__main__":
    print(left_arm_ip, right_arm_ip, left_hand_port, right_hand_port)

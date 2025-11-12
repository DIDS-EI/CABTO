import os
import configparser

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
assets_path = os.path.join(root_path, "assets")
config_ini = configparser.ConfigParser()
config_ini.read(os.path.join(root_path, "task_stage"))

isaacsim_path = config_ini["simulation"]["isaacsim_path"]
data_path = config_ini["simulation"]["data_path"]
model_path = config_ini["simulation"]["model_path"]

api_key = config_ini["api"]["api_key"]
base_url = config_ini["api"]["base_url"]

if __name__ == "__main__":
    print(isaacsim_path, data_path, model_path)

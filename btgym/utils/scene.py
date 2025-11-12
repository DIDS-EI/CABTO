import json
from btgym.dataclass.cfg import cfg
from omnigibson.macros import gm


def load_scene_json(scene_name):
    with open(f"{gm.DATASET_PATH}/scenes/{scene_name}/json/{scene_name}_best.json", "r") as f:
        content = json.load(f)
    return content


def get_scene_info(scene_name):
    scene_json = load_scene_json(scene_name)
    object_info = scene_json['objects_info']['init_info']

    room_set = set()
    object_set = set()
    for objname, info_dict in object_info.items():
        for room in info_dict['args']["in_rooms"]:
            room_set.add(room)
        object_set.add(objname)

    return {
        'rooms': room_set,
        'objects': object_set
    }


if __name__ == "__main__":
    scene_info = get_scene_info("Rs_int")
    print(scene_info)

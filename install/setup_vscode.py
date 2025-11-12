# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script sets up the vs-code settings for the Isaac Lab project.

This script merges the python.analysis.extraPaths from the "{ISAACSIM_DIR}/.vscode/settings.json" file into
the ".vscode/settings.json" file.

This is necessary because Isaac Sim 2022.2.1 onwards does not add the necessary python packages to the python path
when the "setup_python_env.sh" is run as part of the vs-code launch configuration.
"""

import re
import sys
import os
import pathlib
import shutil

ROOT_PATH = pathlib.Path(__file__).parents[1]
print(f"ROOT_PATH: {ROOT_PATH}")
"""Path to the Isaac Lab directory."""

# create the .global folder
os.makedirs(os.path.join(ROOT_PATH, '.vscode'), exist_ok=True)

import configparser
config = configparser.ConfigParser()
config.read(os.path.join(ROOT_PATH, 'global_config.ini'))

isaacsim_dir = config['simulation']['isaacsim_path']


# try:
#     import isaacsim  # noqa: F401

#     isaacsim_dir = os.environ.get("ISAAC_PATH", "")
# except ModuleNotFoundError or ImportError:
#     isaacsim_dir = os.path.join(ROOT_DIR, "_isaac_sim")
# except EOFError:
#     print("Unable to trigger EULA acceptance. This is likely due to the script being run in a non-interactive shell.")
#     print("Please run the script in an interactive shell to accept the EULA.")
#     print("Skipping the setup of the VSCode settings...")
#     sys.exit(0)

# check if the isaac-sim directory exists
if not os.path.exists(isaacsim_dir):
    raise FileNotFoundError(
        f"Could not find the isaac-sim directory: {isaacsim_dir}. There are two possible reasons for this:"
        f"\n\t1. The Isaac Sim directory does not exist as a symlink at: {os.path.join(ROOT_PATH, '_isaac_sim')}"
        "\n\t2. The script could not import the 'isaacsim' package. This could be due to the 'isaacsim' package not "
        "being installed in the Python environment.\n"
        "\nPlease make sure that the Isaac Sim directory exists or that the 'isaacsim' package is installed."
    )

ISAACSIM_DIR = isaacsim_dir
"""Path to the isaac-sim directory."""


def overwrite_python_analysis_extra_paths(dexrl_settings: str) -> str:
    """Overwrite the python.analysis.extraPaths in the Isaac Lab settings file.

    The extraPaths are replaced with the path names from the isaac-sim settings file that exists in the
    "{ISAACSIM_DIR}/.vscode/settings.json" file.

    If the isaac-sim settings file does not exist, the extraPaths are not overwritten.

    Args:
        dexrl_settings: The settings string to use as template.

    Returns:
        The settings string with overwritten python analysis extra paths.
    """
    # isaac-sim settings
    isaacsim_vscode_filename = os.path.join(ISAACSIM_DIR, ".vscode", "settings.json")

    # we use the isaac-sim settings file to get the python.analysis.extraPaths for kit extensions
    # if this file does not exist, we will not add any extra paths
    if os.path.exists(isaacsim_vscode_filename):
        # read the path names from the isaac-sim settings file
        with open(isaacsim_vscode_filename) as f:
            vscode_settings = f.read()
        # extract the path names
        # search for the python.analysis.extraPaths section and extract the contents
        settings = re.search(
            r"\"python.analysis.extraPaths\": \[.*?\]", vscode_settings, flags=re.MULTILINE | re.DOTALL
        )
        settings = settings.group(0)
        settings = settings.split('"python.analysis.extraPaths": [')[-1]
        settings = settings.split("]")[0]

        # read the path names from the isaac-sim settings file
        path_names = settings.split(",")
        path_names = [path_name.strip().strip('"') for path_name in path_names]
        path_names = [path_name for path_name in path_names if len(path_name) > 0]

        # change the path names to be relative to the Isaac Lab directory
        rel_path = os.path.relpath(ISAACSIM_DIR, ROOT_PATH)
        path_names = ['"${workspaceFolder}/' + rel_path + "/" + path_name + '"' for path_name in path_names]
    else:
        path_names = []
        print(
            f"[WARN] Could not find Isaac Sim VSCode settings: {isaacsim_vscode_filename}."
            "\n\tThis will result in missing 'python.analysis.extraPaths' in the VSCode"
            "\n\tsettings, which limits the functionality of the Python language server."
            "\n\tHowever, it does not affect the functionality of the Isaac Lab project."
            "\n\tWe are working on a fix for this issue with the Isaac Sim team."
        )

    # add the path names that are in the external directory
    path_names.extend(['"${workspaceFolder}/"'])

    # add the path names that are in the Isaac Lab extensions directory
    # dexrl_extensions = os.listdir(os.path.join(ROOT_PATH, "sim_extension/dexrl.sim_extension"))
    # path_names.extend(['"${workspaceFolder}/sim_extension/' + ext + '"' for ext in dexrl_extensions])
    # path_names.extend(['"${workspaceFolder}/sim_extension/dexrl.sim_extension/"'])

    # combine them into a single string
    path_names = ",\n\t\t".expandtabs(4).join(path_names)
    # deal with the path separator being different on Windows and Unix
    path_names = path_names.replace("\\", "/")

    # replace the path names in the Isaac Lab settings file with the path names parsed
    dexrl_settings = re.sub(
        r"\"python.analysis.extraPaths\": \[.*?\]",
        '"python.analysis.extraPaths": [\n\t\t'.expandtabs(4) + path_names + "\n\t]".expandtabs(4),
        dexrl_settings,
        flags=re.DOTALL,
    )
    # return the Isaac Lab settings string
    return dexrl_settings


def main():
    # read the Isaac Lab template settings file
    isaacsim_vscode_filename = os.path.join(ISAACSIM_DIR, ".vscode", "settings.json")

    with open(isaacsim_vscode_filename) as f:
        isaacsim_template_settings = f.read()

    # overwrite the python.analysis.extraPaths in the Isaac Lab settings file with the path names
    dexrl_settings = overwrite_python_analysis_extra_paths(isaacsim_template_settings)

    # write the Isaac Lab settings file
    dexrl_vscode_filename = os.path.join(ROOT_PATH, ".vscode", "settings.json")
    with open(dexrl_vscode_filename, "w") as f:
        f.write(dexrl_settings)


if __name__ == "__main__":
    main()

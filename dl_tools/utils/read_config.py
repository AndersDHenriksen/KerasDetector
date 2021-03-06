import os
import json
from munch import Munch
from pathlib import Path
from datetime import datetime
from shutil import copytree, ignore_patterns

copy_code_to_model_dir = True


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Munch(config_dict)

    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    config.load_model = None
    experiment_folder = Path(config.experiment_folder)
    do_load_exp = "_run" in config.exp_name
    if do_load_exp:
        exp_name = list(experiment_folder.glob('*' + config.exp_name))
        if len(exp_name) and len(list((exp_name[0] / "checkpoint").glob("latest_epoch*"))):
            config.exp_name = exp_name[-1].stem
            model_chkpoints = list((exp_name[0] / "checkpoint").glob("latest_epoch*"))
            config.load_model = sorted(model_chkpoints, key=lambda p: int(p.stem[13:18]))[-1]
            config.model_epoch = int(config.load_model.stem[13:18])
        else:
            do_load_exp = False
    if not do_load_exp:
        run_n = len(list(experiment_folder.glob('*' + config.exp_name + '*')))
        config.exp_name = "{} - {}_run{}".format(datetime.now().strftime('%Y-%m-%d %H-%M-%S'), config.exp_name, run_n)
        config.model_epoch = 0

    config.data_folder = Path(config.data_folder)
    config.log_dir = str(experiment_folder / config.exp_name / "log") + os.sep
    config.checkpoint_dir = str(experiment_folder / config.exp_name / "checkpoint") + os.sep

    [Path(dir_path).mkdir(parents=True, exist_ok=True) for dir_path in [config.log_dir, config.checkpoint_dir]]

    if copy_code_to_model_dir:
        code_dir = str(experiment_folder / config.exp_name / "code") + os.sep
        copytree(src=__file__[:__file__.find('dl_tools')], dst=code_dir, ignore=ignore_patterns('__pycache__', '.*'),
                 dirs_exist_ok=True)

    return config

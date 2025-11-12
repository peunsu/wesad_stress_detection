""" util function.py """

import json
import argparse
from pathlib import Path
from datetime import datetime


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    return config_dict


def save_config(config):
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y-%H-%M")
    filename = Path(config['result_dir']) / f'training_config_{timestampStr}.txt'
    config_to_save = json.dumps(config)
    with open(filename, "w", encoding='utf-8') as f:
        f.write(config_to_save)

def process_config(json_file):
    config = get_config_from_json(json_file)

    # create directories to save experiment results and trained models
    if config['load_dir'] == "default":
        save_dir = Path("experiments/local-results") / config['exp_name'] / f"batch-{config['batch_size']}"
    else:
        save_dir = Path(config['load_dir'])
    # specify the saving folder name for this experiment
    if config['TRAIN_sigma'] == 1:
        save_name = f'{config["exp_name"]}-{config["l_win"]}-{config["l_seq"]}-{config["code_size"]}-trainSigma'
    else:
        save_name = f'{config["exp_name"]}-{config["l_win"]}-{config["l_seq"]}-{config["code_size"]}-fixedSigma-{config["sigma"]}'
    config['summary_dir'] = str(save_dir / save_name / "summary/")
    config['result_dir'] = str(save_dir / save_name / "result/")
    config['checkpoint_dir'] = str(save_dir / save_name / "checkpoint/")
    config['checkpoint_dir_lstm'] = str(save_dir / save_name / "checkpoint/lstm/")

    return config


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            Path(dir_).mkdir(parents=True, exist_ok=True)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args

import os
import random
import datetime
import yaml
import torch
import numpy as np

def load_config(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)

    return config

def set_env():
    os.environ['HF_TOKEN'] = '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'

def set_seed(seed:int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    print('SEED SET TO:', seed)
    

def make_today_path(dir):
    current_time = datetime.datetime.now().strftime("%T%m%d-%H%M%S")
    return os.path.join(dir, current_time)
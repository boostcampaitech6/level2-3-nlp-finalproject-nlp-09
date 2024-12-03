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
    os.environ['HF_TOKEN'] = 'your key'

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
    

def make_today_path(dir, model_name):
    # model name is huggingface name
    model_name = model_name.replace('/', '-')
    file_path = model_name + '-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(dir, file_path)
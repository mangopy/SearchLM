import os
from tuning import train
from inference import inference, entropy, formate, reward
import torch
import numpy as np
import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multiple gpus
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# set_seed(42)

if __name__ == '__main__':
    procedure = os.environ['PROCEDURE']
    if procedure == 'align':
        os.environ['http_proxy'] = "http://agent.baidu.com:8891"
        os.environ['https_proxy'] = "http://agent.baidu.com:8891"
        train(_type='align')
    elif procedure == 'sft':
        os.environ['http_proxy'] = "http://agent.baidu.com:8891"
        os.environ['https_proxy'] = "http://agent.baidu.com:8891"
        train(_type='sft')
    elif procedure == 'formate':
        formate()
    elif procedure == 'inference':
        inference()
    elif procedure == 'entropy':
        entropy()
    elif procedure == 'reward':
        reward()
    else:
        raise "Not implemented procedure ..."
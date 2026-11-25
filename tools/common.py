import argparse
import random
import numpy as np
import torch
from config import get_cfg_defaults

def options(description):
    parser=argparse.ArgumentParser(description=description)
    parser.add_argument('--config_file',required=True)
    parser.add_argument('opts',nargs=argparse.REMAINDER)
    args=parser.parse_args(); cfg=get_cfg_defaults()
    cfg.merge_from_file(args.config_file); cfg.merge_from_list(args.opts)
    if cfg.MODEL.DIST_TRAIN: raise ValueError('This submission provides the verified single-GPU training entry point.')
    if cfg.MODEL.METRIC_LOSS_TYPE!='triplet': raise ValueError('The supported training objective uses triplet and ID losses.')
    cfg.freeze(); return cfg

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=True

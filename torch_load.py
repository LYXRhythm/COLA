import argparse
import os, sys

import numpy as np

import resnet

sys.path.append('./')

import os.path as osp
import torch

import random
from collate import SimCLRCollateFunction
from Trainer import trainer_coda
from data_clus import office_load_idx
import clustering
from network import Model
from utils.utils import *
from train_udar_oh_coda import get_args

args = get_args()
model = torch.load("./pth/c2r_model_best.pt")
#print(model)
#model.eval()
dset_loaders = office_load_idx(args)
if __name__ == "__main__":
    map_train_data = trainer_coda.train_target(args,dset_loaders['source_tr'],dset_loaders['target'],model)
    print(map_train_data)
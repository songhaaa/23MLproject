from __future__ import print_function
import argparse
import errno

import os
import shutil
import time
import random
import copy
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchsummary import summary
from torch.autograd import Variable
import os,sys,inspect
from library.train_loop import TrainLoop
from library.optmization import Optmization
from library.optmization import ema_model, WeightEMA
from library.model import Conv_EEG
import math
from library.utils import *
import yaml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import f1_score, accuracy_score
from copy import deepcopy

parser = argparse.ArgumentParser(description='PyTorch SSL Training')
# Optimization options
parser.add_argument('--dataset', default='SEED-IV', type=str, metavar='N',
                    help='dataset name')
parser.add_argument('--method', default='PARSE', type=str, metavar='N',
                    help='method name')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run') ## epoch 수 ##
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=16, type=int, metavar='N',
                    help='train batchsize') # 8 -> 16 으로 변경
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate') # lr 0.017121492346507953로 수정
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
parser.add_argument('--gpu', default='3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--n-labeled', type=int, default=20,
                        help='Number of labeled data')
parser.add_argument('--out', default='result',
                        help='Directory to output the result')
parser.add_argument('--alpha', default=0.84, type=float) # 0.75 -> 0.8386694223628687로 변경
parser.add_argument('--T', default=0.5, type=float,
                    help='pseudo label temperature')  # 1 -> 1.821842823921708로 변경
parser.add_argument('--w-da', default=1.0, type=float,
                    help='data distribution weight')
parser.add_argument('--weak-aug', default=0.2, type=float,
                    help='weak aumentation')
parser.add_argument('--strong-aug', default=0.8, type=float,
                    help='strong aumentation')
parser.add_argument('--threshold', default=0.85, type=float,
                    help='pseudo label threshold') # 0.95 -> 0.852134269079369로 변경
parser.add_argument('--lambda-u', default=100, type=float) # 75 -> 100으로 변경
parser.add_argument('--ema-decay', default=0.64, type=float) # 0.999 -> 0.6371038517884978로 변경
parser.add_argument('--init-weight', default=20, type=float) # 0 -> 20으로 변경
parser.add_argument('--end-weight',  default=30, type=float)
# 새로 추가한 argument
parser.add_argument('--num_classes', type=int, default=4)
parser.add_argument('--lambda_cls', type=float, default=1)
parser.add_argument('--lambda_ot', type=float, default=1)
parser.add_argument('--paradigm', default='withinses', type=str,
                    help='choose the experimental paradigm as follows: withinses(default), sub2sub')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

def load_config(name):
    with open(os.path.join(sys.path[0], name)) as file:
        config = yaml.safe_load(file)
    return config

def augmentation(input, std):

    # input_shape = np.shape(input)
    # noise = torch.normal(mean=0.5, std=std, size =input_shape)
    # noise = noise.to(device)
    if std == 0.8:

    elif std == 0.2:

    else:
        raise Exception('Std Error')


    return

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'SEED-IV':
        # dataset (Differential Entropy) (x)
        train_de = '/home/user/bci2/SEED_IV/feat/train/{}_{}_X.npy'  # Subject_No, Session_No
        test_de = '/home/user/bci2/SEED_IV/feat/test/{}_{}_X.npy'  # Subject_No, Session_No
        # dataset (y)
        train_label = '/home/user/bci2/SEED_IV/feat/train/{}_{}_y.npy'
        test_label = '/home/user/bci2/SEED_IV/feat/test/{}_{}_y.npy'

    else:
        raise Exception('Datasets Name Error')

    config = load_config('dataset_params.yaml')

    dataset_dict = config[args.dataset]
    # for subject_num in (range(1, dataset_dict['Subject_No'] + 1)):
    #     for session_num in range(1, dataset_dict['Session_No'] + 1):
    for subject_num in (range(1, 3)):
        for session_num in range(1, 3):
            X_train = np.load(train_de.format(subject_num, session_num))
            X_test  = np.load(test_de.format(subject_num, session_num))

            print("X_train:", X_train)
            print("X_test:", X_test)

            print("X_train size", np.shape(X_train))
            #
            # Y_train = np.load(train_label.format(subject_num, session_num))
            # Y_test = np.load(test_label.format(subject_num, session_num))
            #
            # print("Y_train:", Y_train)
            # print("Y_test:", Y_test)


            # input_s = augmentation(X_train, args.strong_aug)  # strong aug
            # input_w = augmentation(X_train, args.weak_aug)  # weak aug
            #
            # output = (input_s, input_w)
            # print("output:", output)
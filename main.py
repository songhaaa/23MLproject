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
parser.add_argument('--lr', '--learning-rate', default=0.0017121492346507953, type=float,
                    metavar='LR', help='initial learning rate') # lr 0.017121492346507953로 수정
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--n-labeled', type=int, default=25,
                        help='Number of labeled data')
parser.add_argument('--out', default='result',
                        help='Directory to output the result')
parser.add_argument('--alpha', default=0.8386694223628687, type=float) # 0.75 -> 0.8386694223628687로 변경
parser.add_argument('--T', default=1.821842823921708, type=float,
                    help='pseudo label temperature')  # 1 -> 1.821842823921708로 변경
parser.add_argument('--w-da', default=1.0, type=float,
                    help='data distribution weight')
parser.add_argument('--weak-aug', default=0.2, type=float,
                    help='weak aumentation')
parser.add_argument('--strong-aug', default=0.8, type=float,
                    help='strong aumentation')
parser.add_argument('--threshold', default=0.852134269079369, type=float,
                    help='pseudo label threshold') # 0.95 -> 0.852134269079369로 변경
parser.add_argument('--lambda-u', default=100, type=float) # 75 -> 100으로 변경
parser.add_argument('--ema-decay', default=0.6371038517884978, type=float) # 0.999 -> 0.6371038517884978로 변경
parser.add_argument('--init-weight', default=20, type=float) # 0 -> 20으로 변경
parser.add_argument('--end-weight',  default=30, type=float)
# 새로 추가한 argument
parser.add_argument('--num_classes', type=int, default=4)
parser.add_argument('--lambda_cls', type=float, default=1)
parser.add_argument('--lambda_ot', type=float, default=1)


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


def load_config(name):
    with open(os.path.join(sys.path[0], name)) as file:
        config = yaml.safe_load(file)

    return config

config = load_config('dataset_params.yaml')


class Model(nn.Module):
    def __init__(self, Conv_EEG):
        super(Model, self).__init__()
        self.model = Conv_EEG(args.dataset, args.method)


    def augmentation(self, input, std):

        input_shape =input.size()
        noise = torch.normal(mean=0.5, std=std, size =input_shape)
        noise = noise.to(device)

        return input + noise

    def forward(self, input, compute_model=True):

        if compute_model==False:
            input_s  = self.augmentation(input, args.strong_aug)
            input_w  = self.augmentation(input, args.weak_aug)

            output = (input_s, input_w)

        else:
            if args.method == 'PARSE':
                output_c, output_d = self.model(input) # 모듈 수정
                output = (output_c, output_d)
            else:
                output = self.model(input)
        return output



def ssl_process(Net, X_train, X_test, Y_train, Y_test, n_labeled_per_class, random_seed,subject_num, session_num):


    data_train,  data_test  = np.expand_dims(X_train, axis=1), np.expand_dims(X_test, axis=1)
    label_train, label_test = Y_train, Y_test

    '''the choice of labeled and unlabeled samples'''

    train_labeled_idxs, train_unlabeled_idxs  = train_split(label_train, n_labeled_per_class, random_seed, config[args.dataset]['Class_No'])

    X_labeled,   Y_labeled   = data_train[train_labeled_idxs],   label_train[train_labeled_idxs]
    X_unlabeled, Y_unlabeled = data_train[train_unlabeled_idxs], label_train[train_unlabeled_idxs]* (-1)

    batch_size = args.batch_size

    unlabeled_ratio = math.ceil(len(Y_unlabeled)/ len(Y_labeled))
    max_iterations  = math.floor(len(Y_unlabeled)/batch_size)
    X_labeled, Y_labeled = np.tile(X_labeled, (unlabeled_ratio,1,1)), np.tile(Y_labeled,(unlabeled_ratio,1))

    train_dataset_labeled   = load_dataset_to_device(X_labeled,   Y_labeled,   batch_size=batch_size,   class_flag=True,  shuffle_flag=True)
    train_dataset_unlabeled = load_dataset_to_device(X_unlabeled, Y_unlabeled, batch_size=batch_size,   class_flag=False, shuffle_flag=True)
    test_dataset            = load_dataset_to_device(data_test,   label_test,  batch_size=batch_size,   class_flag=True,  shuffle_flag=False)

    result_acc, result_loss = train(Net, train_dataset_labeled, train_dataset_unlabeled, test_dataset, max_iterations, subject_num, session_num)
    result_acc = np.squeeze(result_acc) if result_acc.ndim == 2 else result_acc
    result_loss = np.squeeze(result_loss) if result_loss.ndim == 2 else result_loss

    return result_acc, result_loss #shape = (30,)




def train(Net, train_dataset_labeled, train_dataset_unlabeled, test_dataset, max_iterations, subject_num, session_num):

    training_params = {'method': args.method, 'batch_size': args.batch_size, 'alpha': args.alpha,
                        'threshold': args.threshold, 'T': args.T,
                        'w_da': args.w_da, 'lambda_u': args.lambda_u}

    test_metric = np.zeros((args.epochs, 1))
    loss_metric = np.zeros((args.epochs, 1))
    train_loss_epoch =np.zeros((args.epochs, 1))
    pbar_train = tqdm(range(args.epochs),
                     total=len(range(args.epochs)),  ## 전체 진행수
                     ncols=100,  ## 진행률 출력 폭 조절
                     ascii=' >=',  ## 바 모양, 첫 번째 문자는 공백이어야 작동
                     position= 1, ## 이중루프일때
                     leave=False  ## True 반복문 완료시 진행률 출력 남김. False 남기지 않음.
                     )


    for epoch in pbar_train:
        start = time.time()
        train_loss_batch = []
        train_acc_batch = []

        pbar_train.set_description(f"subject {subject_num} | session {session_num} | Train")


        if args.method == 'FixMatch':
            ema_Net = ema_model(Model(Conv_EEG).to(device))
            a_optimizer = optim.Adam(Net.parameters(), args.lr)
            ema_optimizer= WeightEMA(Net, ema_Net, alpha=args.ema_decay, lr=args.lr)

        else:
            pass

        Net.train()

        labeled_train_iter    = iter(train_dataset_labeled)
        unlabeled_train_iter  = iter(train_dataset_unlabeled)

        for batch_idx in range(max_iterations):
            try:
                inputs_x, targets_x = next(labeled_train_iter)
            except:
                labeled_train_iter  = iter(train_dataset_labeled)
                inputs_x, targets_x = next(labeled_train_iter)


            try:
                inputs_u, _ = next(unlabeled_train_iter)
            except:
                unlabeled_train_iter  = iter(train_dataset_unlabeled)
                inputs_u, _ = next(unlabeled_train_iter)


            optimizer = optim.Adam(Net.parameters(), args.lr)
            inputs_x, targets_x, inputs_u = inputs_x.to(device), targets_x.to(device, non_blocking=True), inputs_u.to(device)

            optmization_params = {'lr': args.lr, 'current_epoch': epoch, 'total_epochs': args.epochs, 'current_batch': batch_idx, 'max_iterations': max_iterations,
                                 'init_w': args.init_weight, 'end_w': args.end_weight}


            '''
            Training options for various methods
            '''

            if args.method   == 'MixMatch':
                unsupervised_weight =  Optmization(optmization_params).linear_rampup()
                loss = TrainLoop(training_params).train_step_mix(inputs_x, targets_x, inputs_u, Net, optimizer, unsupervised_weight)

            elif args.method == 'FixMatch':

                loss = TrainLoop(training_params).train_step_fix(inputs_x, targets_x, inputs_u, Net, a_optimizer, ema_optimizer)

            elif args.method == 'AdaMatch':
                unsupervised_weight = Optmization(optmization_params).ada_weight()
                reduced_lr = Optmization(optmization_params).decayed_learning_rate()
                optimizer = optim.Adam(Net.parameters(), reduced_lr)
                loss = TrainLoop(training_params).train_step_ada(inputs_x, targets_x, inputs_u, Net, optimizer, unsupervised_weight)

            elif args.method == 'PARSE':
                unsupervised_weight = Optmization(optmization_params).ada_weight()

                loss = TrainLoop(training_params).train_step_parse(inputs_x, targets_x, inputs_u, Net, optimizer, unsupervised_weight, args) # args 추가

            else:
                raise Exception('Methods Name Error')

            train_loss_batch.append(loss)


            # Create a string for the train loss
        mean_train_loss = np.mean(train_loss_batch)


        Net.eval()

        with torch.no_grad():
            test_loss_batch = []
            test_ytrue_batch = []
            test_ypred_batch = []

            for image_batch, label_batch in test_dataset:
                image_batch = image_batch.to(device)
                label_batch = label_batch.to(device)

                loss, y_true, y_pred  = TrainLoop(training_params).eval_step(image_batch, label_batch, Net)
                test_loss_batch.append(loss)

                test_ytrue_batch.append(y_true)
                test_ypred_batch.append(y_pred)


            test_ypred_epoch = np.array(test_ypred_batch).flatten()
            test_ytrue_epoch = np.array(test_ytrue_batch).flatten()

        if args.dataset == 'AMIGOS':
            metric = f1_score(test_ytrue_epoch, test_ypred_epoch, average='macro')
        else:
            metric = accuracy_score(test_ytrue_epoch, test_ypred_epoch)

        test_metric[epoch] = metric
        loss_metric[epoch] = mean_train_loss
        pbar_train.set_postfix({'Train Loss': mean_train_loss , 'Test Acc': metric })
    pbar_train.close()

    return test_metric, loss_metric #shape= (30,1)



def net_init(model):
    '''load and initialize the model'''
    Net = Model(model).to(device)
    Net.apply(WeightInit)
    Net.apply(WeightClipper)

    return Net



if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # Set the GPUs 2 and 3 to use

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''set random seeds for torch and numpy libraies'''
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)


    ''' data and label address loader for each dataset '''

    if args.dataset =='SEED-IV':
        #dataset (Differential Entropy) (x)
        train_de    = '/home/user/bci2/SEED_IV/feat/train/{}_{}_X.npy'  # Subject_No, Session_No
        test_de     = '/home/user/bci2/SEED_IV/feat/test/{}_{}_X.npy'  # Subject_No, Session_No
        #dataset (y)
        train_label = '/home/user/bci2/SEED_IV/feat/train/{}_{}_y.npy'
        test_label  = '/home/user/bci2/SEED_IV/feat/test/{}_{}_y.npy'

    else:
        raise Exception('Datasets Name Error')



    '''A set of random seeds for later use of choosing labeled and unlabeled data from training set'''
    random_seed_arr = np.array([100, 42, 19, 57, 598])

    pbar_seed = tqdm(range(len(random_seed_arr)),
                total=len(range(len(random_seed_arr))),  ## 전체 진행수
                desc='SEED iteration',  ## 진행률 앞쪽 출력 문장
                ncols=100,  ## 진행률 출력 폭 조절
                ascii=' #',  ## 바 모양, 첫 번째 문자는 공백이어야 작동
                leave=True,  ## True 반복문 완료시 진행률 출력 남김. False 남기지 않음.
                )
    for seed in pbar_seed:
        pbar_seed.set_description(f'Current seed "{seed+1}"')

        random_seed = random_seed_arr[seed]

        n_labeled_per_class = args.n_labeled # number of labeled samples need to be chosen for each emotion class

        '''create result directory'''
        # directory = './{}_result/ssl_method_{}/run_{}/'.format(args.dataset, args.method, seed+1)
        directory = './{}_result/ssl_method_{}/param_tuning/try_1/run_{}/'.format(args.dataset, args.method, seed+1)

        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                pass

        dataset_dict = config[args.dataset]


        '''
        Experiment setup for all four pulbic datasets
        '''
        if args.dataset == 'SEED-IV':
            acc_array = np.zeros((dataset_dict['Subject_No'], dataset_dict['Session_No'], args.epochs))
            loss_array = np.zeros((dataset_dict['Subject_No'], dataset_dict['Session_No'], args.epochs))

            for subject_num in (range(1, dataset_dict['Subject_No']+1)):
                for session_num in range(1, dataset_dict['Session_No']+1):

                    Net = net_init(Conv_EEG) #initializing the network
                    # Net_s = net_init(Conv_EEG_Song)

                    X_train = np.load(train_de.format(subject_num, session_num))
                    X_test  = np.load(test_de.format(subject_num, session_num))


                    X = np.vstack((X_train, X_test))
                    X = np.reshape(X, (-1,310)) # (trial , channel, frequency band ) -> (trial , ch x freq_b ): ch = 62, freq_b = 5

                    scaler = MinMaxScaler()
                    X = scaler.fit_transform(X)

                    X_train = X[0: X_train.shape[0]]
                    X_test  = X[X_train.shape[0]:]

                    Y_train = np.load(train_label.format(subject_num, session_num))
                    Y_test  = np.load(test_label.format(subject_num, session_num))


                    #split dataset as unlabeled / labeled , and  label guessing with ssl
                    acc_array[subject_num-1, session_num-1], loss_array[subject_num-1, session_num-1] = ssl_process(Net, X_train, X_test, Y_train, Y_test, n_labeled_per_class, random_seed, subject_num, session_num)

                    torch.cuda.empty_cache()

                    # print(np.shape(acc_array)) # (15, 3, 30)
                    # print("loss_array_shape:", np.shape(loss_array))  # (15, 3, epoch)
                    new_acc_array = []
                    new_loss_array = []
                    for i in range(0, len(acc_array)):
                        nacc = []
                        nloss = []
                        for j in range(0, len(acc_array[i])):
                            nacc.append(sum(acc_array[i][j]) / len(acc_array[i][j]))
                            nloss.append(sum(loss_array[i][j]) / len(loss_array[i][j]))
                        new_acc_array.append(nacc)
                        new_loss_array.append(nloss)
                      
                    np.savetxt(os.path.join(directory, 'new_acc_labeled_{}.txt').format(n_labeled_per_class), new_acc_array, delimiter=",")
                    np.savetxt(os.path.join(directory, 'new_loss_labeled_{}.txt').format(n_labeled_per_class),new_loss_array, delimiter=",")


        else:
            raise Exception('Datasets Name Error')



pbar_seed.close()

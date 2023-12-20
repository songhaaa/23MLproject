from __future__ import print_function
import argparse
import errno

import os
import shutil
from datetime import datetime
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
from library.train_loop_2 import TrainLoop
from library.optmization import Optmization
from library.optmization import ema_model, WeightEMA
from library.model import Conv_EEG
import math
from library.utils_2 import *
import yaml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import f1_score, accuracy_score
from copy import deepcopy
from library.losses import *
from library.ema import *
from library.lr_schedule import *
# from setproctitle import *

parser = argparse.ArgumentParser(description='PyTorch SSL Training')
# Optimization options
parser.add_argument('--debug', default=False , type=str2bool, help='debug mode')
parser.add_argument('--dataset', default='SEED-IV', type=str, metavar='N',
                    help='dataset name')
parser.add_argument('--method', default='PARSE', type=str, metavar='N',
                    help='method name')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=8, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate') #1e-3
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
parser.add_argument('--gpu', default='5', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--n-labeled', type=int, default=25,
                        help='Number of labeled data')
parser.add_argument('--out', default='result',
                        help='Directory to output the result')
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--T', default=1, type=float,
                    help='pseudo label temperature')
parser.add_argument('--w-da', default=1.0, type=float,
                    help='data distribution weight')
parser.add_argument('--weak-aug', default=0.2, type=float,
                    help='weak aumentation')
parser.add_argument('--strong-aug', default=0.8, type=float,
                    help='strong aumentation')
parser.add_argument('--threshold', default=0.95, type=float,
                    help='pseudo label threshold')
parser.add_argument('--lambda-u', default=75, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--init-weight', default=0, type=float)
parser.add_argument('--end-weight',  default=30, type=float)
parser.add_argument('--paradigm', default='sub2sub', type=str,
                    help='choose the experimental paradigm as follows: ses2ses(default), sub2sub')
parser.add_argument('--threshold1', default=0.95, type=float,
                        help='pseudo label threshold1')
parser.add_argument('--threshold2', default=0.4, type=float,
                        help='pseudo label threshold2')
parser.add_argument('--Temp', type=float, default=0.1,
                        help='temperature (default: 0.1)')
parser.add_argument('--warm_steps', type=int, default=100)
parser.add_argument('--prodim', type=int, default=3060)
parser.add_argument('--n_shuffle', type=float, required=False)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# setproctitle('Ssoy')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu

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
            '''
            compute_model==False => AUGMENTATION  
            '''
            input_s  = self.augmentation(input, args.strong_aug)
            input_w  = self.augmentation(input, args.weak_aug)

            output = (input_s, input_w)

        else:
            if args.method=='PARSE':
                output_c, output_d = self.model(input)
                output = (output_c, output_d)
            else:
                output = self.model(input)
        return output



def ssl_process(Net, X_train, X_test, Y_train, Y_test, n_labeled_per_class, random_seed,subject_train=None, subject_test=None, subject_num=None, session_num=None):


    data_train,  data_test  = np.expand_dims(X_train, axis=1), np.expand_dims(X_test, axis=1)
    label_train, label_test = Y_train, Y_test

    '''the choice of labeled and unlabeled samples'''
    if args.paradigm =="sub2sub":
        train_labeled_idxs, train_unlabeled_idxs = train_split_sub2sub(label_train, subject_train, n_labeled_per_class, random_seed,
                                                               config[args.dataset]['Class_No'])
        # train_labeled_idxs, train_unlabeled_idxs = train_split_sub2sub(label_train, subject_train, n_labeled_per_class, random_seed,
        #                                                        config[args.dataset]['Class_No'])
    else:
        train_labeled_idxs, train_unlabeled_idxs  = train_split(label_train, n_labeled_per_class, random_seed, config[args.dataset]['Class_No'])

    X_labeled,   Y_labeled = data_train[train_labeled_idxs],   label_train[train_labeled_idxs]
    X_unlabeled, Y_unlabeled = data_train[train_unlabeled_idxs], label_train[train_unlabeled_idxs]* (-1)
    if args.paradigm == "sub2sub":
        sublist_train, sublist_test = subject_train, subject_test
        subject_labeled = sublist_train[train_labeled_idxs]
        subject_unlabeled = sublist_train[train_unlabeled_idxs]
    batch_size = args.batch_size


    unlabeled_ratio = math.ceil(len(Y_unlabeled)/ len(Y_labeled))
    max_iterations  = math.floor(len(Y_unlabeled)/batch_size)
    X_labeled, Y_labeled = np.tile(X_labeled, (unlabeled_ratio,1,1)), np.tile(Y_labeled,(unlabeled_ratio,1))

    if args.paradigm =="sub2sub":
        subject_labeled = np.tile(subject_labeled, (unlabeled_ratio, 1))
        train_dataset_labeled = load_dataset_to_device(X_labeled, Y_labeled, batch_size=batch_size, class_flag=True,
                                                       shuffle_flag=True, subject_list=subject_labeled)
        train_dataset_unlabeled = load_dataset_to_device(X_unlabeled, Y_unlabeled, batch_size=batch_size,   class_flag=False, shuffle_flag=True,
                                                         subject_list=subject_unlabeled)
    else:
        train_dataset_labeled = load_dataset_to_device(X_labeled, Y_labeled, batch_size=batch_size, class_flag=True,
                                                       shuffle_flag=True)
        train_dataset_unlabeled = load_dataset_to_device(X_unlabeled, Y_unlabeled, batch_size=batch_size, class_flag=False, shuffle_flag=True)
    test_dataset            = load_dataset_to_device(data_test,   label_test,  batch_size=batch_size,   class_flag=True,  shuffle_flag=False)


    result_acc, result_loss, train_acc, test_loss = train(Net, train_dataset_labeled, train_dataset_unlabeled, test_dataset, max_iterations, subject_num, session_num)
    result_acc = np.squeeze(result_acc) if result_acc.ndim == 2 else result_acc
    result_loss = np.squeeze(result_loss) if result_loss.ndim == 2 else result_loss
    train_acc = np.squeeze(train_acc) if train_acc.ndim == 2 else train_acc
    test_loss = np.squeeze(test_loss) if test_loss.ndim == 2 else test_loss

    return result_acc, result_loss, train_acc, test_loss




def train(Net, train_dataset_labeled, train_dataset_unlabeled, test_dataset, max_iterations, subject_num=None, session_num=None):

    training_params = {'method': args.method, 'batch_size': args.batch_size, 'alpha': args.alpha,
                        'threshold': args.threshold, 'T': args.T,
                        'w_da': args.w_da, 'lambda_u': args.lambda_u}

    test_metric = np.zeros((args.epochs, 1))
    loss_metric = np.zeros((args.epochs, 1))
    loss_test =np.zeros((args.epochs, 1))
    train_metric = np.zeros((args.epochs, 1))
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
        train_ytrue_batch = []
        train_ypred_batch =[]
        if args.paradigm !="sub2sub":
            pbar_train.set_description(f"subject {subject_num} | session {session_num} | Train")
        else:
            pbar_train.set_description(f"Cross-sujbect scenario | Train")

        if args.method == 'FixMatch':
            ema_Net = ema_model(Model(Conv_EEG).to(device))
            a_optimizer = optim.Adam(Net.parameters(), args.lr)
            ema_optimizer= WeightEMA(Net, ema_Net, alpha=args.ema_decay, lr=args.lr)

        elif args.method == 'Proto':
            Net_ema = ModelEMA(Net, decay=0.99)
            proto_t = Prototype_t(C=args.num_classes, dim=args.prodim)  # prodim : in SEED-IV, 3060

        else:
            pass


        '''TRAINING PHASE'''
        Net.train()

        labeled_train_iter    = iter(train_dataset_labeled) #Split batch according to iteration
        unlabeled_train_iter  = iter(train_dataset_unlabeled)


        for batch_idx in range(max_iterations): #batch-wise training
            try:
                if args.paradigm == "sub2sub":
                    inputs_x, targets_x, sublist_x = labeled_train_iter.next()
                else:
                    inputs_x, targets_x = labeled_train_iter.next()
            except:
                labeled_train_iter = iter(train_dataset_labeled)
                if args.paradigm == "sub2sub":
                    inputs_x, targets_x, sublist_x = labeled_train_iter.next()
                else:
                    inputs_x, targets_x = labeled_train_iter.next() # x, y


            try:
                if args.paradigm == "sub2sub":
                    inputs_u, _, sublist_u = unlabeled_train_iter.next()
                else:
                    inputs_u, _ = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter  = iter(train_dataset_unlabeled)
                if args.paradigm == "sub2sub":
                    inputs_u, _, sublist_u = unlabeled_train_iter.next()
                else:
                    inputs_u, _ = unlabeled_train_iter.next()



            if args.method =='Proto':
                optimizer = optim.SGD(Net.parameters(), args.lr,
                          momentum=0.9, weight_decay=5e-4, nesterov=True)
            elif args.method == 'PARSUB':
                optimizer = optim.RMSprop(Net.parameters(),
                                          lr=args.lr, weight_decay=1e-5)
            else:
                optimizer = optim.Adam(Net.parameters(), args.lr)

            inputs_x, targets_x, inputs_u = inputs_x.to(device), targets_x.to(device, non_blocking=True), inputs_u.to(device)
            if args.paradigm == "sub2sub":
                sublist_u = sublist_u.to(device, non_blocking=True)
                sublist_x = sublist_x.to(device, non_blocking=True)

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

                loss,y_true_x, y_pred_x, y_pred_u = TrainLoop(training_params).train_step_parse(inputs_x, targets_x, inputs_u, Net, optimizer, unsupervised_weight)

            elif args.method == 'PARSUB':
                unsupervised_weight = Optmization(optmization_params).ada_weight()
                loss, y_true_x, y_pred_x, y_pred_u= TrainLoop(training_params).train_step_parsub(inputs_x, targets_x, inputs_u, Net, optimizer,
                                                                   unsupervised_weight, sublist_x, sublist_u)

            elif args.method =='Proto':
                schedular = InvLr(optimizer)
                # unsupervised_weight = Optmization(optmization_params).ada_weight()
                if args.paradigm =='sub2sub':
                    loss, each_loss_items = TrainLoop(training_params).train_step_proto(inputs_x=inputs_x, targets_x=targets_x, inputs_u=inputs_u,
                                                                                        model=Net, model_ema=Net_ema, optimizer=optimizer,
                                                                                        scheduler=schedular, proto_t=proto_t, batch_idx=batch_idx,
                                                                                        args=args, sublist_x = sublist_x,sublist_u=sublist_u)

                else:
                    loss, each_loss_items = TrainLoop(training_params).train_step_proto(inputs_x, targets_x, inputs_u, Net, Net_ema, optimizer,
                                                                   schedular, proto_t, batch_idx, args)
            else:
                raise Exception('Methods Name Error')

            train_loss_batch.append(loss)
            train_ytrue_batch.extend(y_true_x)
            train_ypred_batch.extend(y_pred_x)




            # Create a string for the train loss
        mean_train_loss = np.mean(train_loss_batch)
        train_acc = accuracy_score(train_ytrue_batch, train_ypred_batch)

        Net.eval()

        with torch.no_grad():
            test_loss_batch = []
            test_ytrue_batch = []
            test_ypred_batch = []

            for image_batch, label_batch in test_dataset:
                image_batch = image_batch.to(device)
                label_batch = label_batch.to(device)

                loss, y_true, y_pred  = TrainLoop(training_params).eval_step(image_batch,label_batch, Net)
                test_loss_batch.append(loss)

                test_ytrue_batch.append(y_true)
                test_ypred_batch.append(y_pred)


            test_ypred_epoch = np.array(test_ypred_batch).flatten()
            test_ytrue_epoch = np.array(test_ytrue_batch).flatten()
            mean_test_loss = np.mean(test_loss_batch)

        if args.dataset == 'AMIGOS':
            metric = f1_score(test_ytrue_epoch, test_ypred_epoch, average='macro')
        else:
            metric = accuracy_score(test_ytrue_epoch, test_ypred_epoch)

        test_metric[epoch] = metric
        loss_metric[epoch] = mean_train_loss
        train_metric[epoch] = train_acc
        loss_test[epoch] = mean_test_loss
        class_counts = np.bincount(test_ypred_epoch, minlength=4)

        pbar_train.set_postfix({' Train Loss': mean_train_loss , 'Train Acc':train_acc,  'Test Acc': metric, 'Test Loss': loss_test})
        print(f"\n Prediction distribution: "
              f"0: {class_counts[0]} |"
              f"1: {class_counts[1]} |"
              f"2: {class_counts[2]} |"
              f"3: {class_counts[3]}.\n")
    pbar_train.close()

    return test_metric, loss_metric, train_metric, loss_test #shape= (30,1)



def net_init(model):
    '''load and initialize the model'''
    Net = Model(model).to(device)
    Net.apply(WeightInit)
    Net.apply(WeightClipper)

    return Net



if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

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
    if args.debug :
        random_seed_arr = np.array([42])
    else :
        random_seed_arr = np.array([100, 42, 19, 57, 598])

    pbar_seed = tqdm(range(len(random_seed_arr)),
                total=len(range(len(random_seed_arr))),  ## 전체 진행수
                desc='SEED iteration',  ## 진행률 앞쪽 출력 문장
                ncols=100,  ## 진행률 출력 폭 조절
                ascii=' #',  ## 바 모양, 첫 번째 문자는 공백이어야 작동
                leave=True,  ## True 반복문 완료시 진행률 출력 남김. False 남기지 않음.
                )
    for seed in pbar_seed:
        pbar_seed.set_description(f'Current seed "{seed}"')

        random_seed = random_seed_arr[seed]

        n_labeled_per_class = args.n_labeled # number of labeled samples need to be chosen for each emotion class

        '''create result directory'''
        directory = './{}_result/ssl_method_{}/run_{}/'.format(args.dataset, args.method, seed+1)

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
        if args.dataset =='SEED':

            acc_array = np.zeros((dataset_dict['Subject_No'], dataset_dict['Session_No'], args.epochs))

            for subject_num in (range(1, dataset_dict['Subject_No']+1)):
                for session_num in range(1, dataset_dict['Session_No']+1):

                    Net = net_init(Conv_EEG) # Network Initilization

                    X_train = np.load(train_de.format(subject_num, session_num))
                    X_test  = np.load(test_de.format(subject_num, session_num))

                    '''Normalize EEG features to the range of [0,1] before fed into model'''
                    X = np.vstack((X_train, X_test))

                    scaler = MinMaxScaler()
                    X = scaler.fit_transform(X)

                    X_train = X[0: X_train.shape[0]]
                    X_test  = X[X_train.shape[0]:]

                    Y_train = np.load(train_label.format(subject_num, session_num))
                    Y_test  = np.load(test_label.format(subject_num, session_num))

                    acc_array[subject_num-1, session_num-1]  =  ssl_process(Net, X_train, X_test, Y_train, Y_test, n_labeled_per_class, random_seed)

                    torch.cuda.empty_cache()

            np.savetxt(os.path.join(directory, "acc_labeled_{}.csv").format(n_labeled_per_class), acc_array , delimiter=",")

        elif args.dataset == 'SEED-IV':

            if args.paradigm == 'ses2ses': #train session : 1~2 / test session : 3
                acc_array = np.zeros((dataset_dict['Subject_No'], dataset_dict['Session_No'], args.epochs))
                for subject_num in (range(1, dataset_dict['Subject_No'] + 1)):
                    Net = net_init(Conv_EEG)  # initializing the network
                    for session_num in range(1, dataset_dict['Session_No']):
                        X_train = np.load(train_de.format(subject_num, session_num))
                        X_tmp = np.load(test_de.format(subject_num, session_num))
                        X_train = np.vstack((X_train, X_tmp))

                        Y_train = np.load(train_label.format(subject_num, session_num))
                        Y_tmp = np.load(test_label.format(subject_num, session_num))
                        Y_train = np.concatenate((Y_train, Y_tmp))
                    X_test = np.load(test_de.format(subject_num, dataset_dict['Session_No']))
                    Y_test = np.load(test_label.format(subject_num, dataset_dict['Session_No']))

                    X = np.vstack((X_train, X_test))
                    X = np.reshape(X, (-1, 310))
                    scaler = MinMaxScaler()
                    X = scaler.fit_transform(X)
                    X_train = X[0: X_train.shape[0]]
                    X_test = X[X_train.shape[0]:]

                    acc_array[subject_num - 1, session_num - 1] = ssl_process(Net, X_train, X_test, Y_train, Y_test,
                                                                              n_labeled_per_class, random_seed, subject_num,
                                                                              session_num)
                    torch.cuda.empty_cache() #test

            elif args.paradigm == 'sub2sub':
                '''
                FOLLOWING LEAVE-ONE-SUBJECT-OUT ( 한서브젝트가 테스트서브젝트가 되고, 나머지가 트레이닝 서브젝트가 되는 방식)
                '''
                acc_array = np.zeros((dataset_dict['Subject_No'], args.epochs))
                loss_array = np.zeros((dataset_dict['Subject_No'], args.epochs))
                test_acc = np.zeros((dataset_dict['Subject_No'], args.epochs))
                test_loss = np.zeros((dataset_dict['Subject_No'], args.epochs))
                args.num_classes=4
                for test_sub_num in tqdm(range(1, dataset_dict['Subject_No'] + 1)):  # 모든 서브젝트가 한번씩 테스트 서브젝트가 됩니다.
                    print(f"\nTest subject number #{test_sub_num}")
                    Net = net_init(Conv_EEG)  # initializing the network


                    subject_list = np.random.permutation(dataset_dict['Subject_No']) + 1

                    # Split subjects into train and test groups
                    train_subject_list = np.delete(subject_list, np.where(subject_list == test_sub_num))
                    test_subject_list = np.array(test_sub_num)
                    # Train subjects
                    X_train_list, Y_train_list,  subject_train_list  = [], [], []
                    for subject_num in train_subject_list:  # 모든 트레이닝 서브젝트&모든 세션 합체
                        for session_num in range(1, dataset_dict['Session_No'] + 1):
                            #X_train
                            X_train_ = np.load(train_de.format(subject_num, session_num))
                            X_tmp = np.load(test_de.format(subject_num, session_num))
                            X_train_ = np.vstack((X_train_, X_tmp))
                            #Y_train
                            Y_train_ = np.load(train_label.format(subject_num, session_num))
                            Y_tmp = np.load(test_label.format(subject_num, session_num))
                            Y_train_ = np.concatenate((Y_train_, Y_tmp))
                            #Subject label
                            subject_label = np.array([subject_num] * len(Y_train_))
                            #Appending
                            X_train_list.append(X_train_)
                            Y_train_list.append(Y_train_)
                            subject_train_list.append(subject_label)

                    X_train = np.concatenate(X_train_list, axis=0)
                    Y_train = np.concatenate(Y_train_list, axis=0)
                    subject_train = np.concatenate(subject_train_list, axis=0)

                    X_test_list, Y_test_list, subject_test_list = [], [], []
                    subject_num = test_subject_list.item()# 모든 테스트 서브젝트 세션 합체
                    for session_num in range(1, 2):
                    #for session_num in range(1, dataset_dict['Session_No']):
                        # X_train
                        X_test_ = np.load(train_de.format(subject_num, session_num))
                        X_tmp = np.load(test_de.format(subject_num, session_num))
                        X_test_ = np.vstack((X_test_, X_tmp))
                        # Y_train
                        Y_test_ = np.load(train_label.format(subject_num, session_num))
                        Y_tmp = np.load(test_label.format(subject_num, session_num))
                        Y_test_ = np.concatenate((Y_test_, Y_tmp))
                        subject_label = np.array([subject_num] * len(Y_test_))

                        X_test_list.append(X_test_)
                        Y_test_list.append(Y_test_)
                        subject_test_list.append(subject_label)

                    X_test = np.concatenate(X_test_list, axis=0)
                    Y_test = np.concatenate(Y_test_list, axis=0)
                    subject_test = np.concatenate(subject_test_list, axis=0)


                    X_train = np.reshape(X_train, (-1, 310))
                    X_test = np.reshape(X_test, (-1, 310))
                    # X = np.vstack((X_train, X_test))
                    # X = np.reshape(X, (-1, 310))
                    # scaler = MinMaxScaler()
                    # X = scaler.fit_transform(X)
                    # X_train = X[0: X_train.shape[0]]
                    # X_test = X[X_train.shape[0]:]

                    # SSL process 부분에서 session_num입력하는 부분을 꼭 넣을필요없도록 바꿨어요.
                    acc_array[test_sub_num - 1], loss_array[test_sub_num-1], test_acc[test_sub_num-1], test_loss[test_sub_num-1] = ssl_process(Net, X_train, X_test,
                                                              Y_train, Y_test,
                                                              n_labeled_per_class,
                                                              random_seed, subject_train, subject_test)
                    torch.cuda.empty_cache()  # test
                # new_acc_array = []
                # new_loss_array = []
                # for i in range(0, len(acc_array)):
                #     nacc = []
                #     nloss = []
                #     for j in range(0, len(acc_array[i])):
                #         nacc.append(sum(acc_array[i][j]) / len(acc_array[i][j]))
                #         nloss.append(sum(loss_array[i][j]) / len(loss_array[i][j]))
                #     new_acc_array.append(nacc)
                #     new_loss_array.append(nloss)

                    # print(np.shape(new_acc_array))  # (15, 3)
                    # print("new_loss_array_shape:", np.shape(new_loss_array)) # (15, 3)
                    # np.savetxt(os.path.join(directory, 'acc_labeled_{}.csv').format(n_labeled_per_class), acc_array[-1] , delimiter=",")
                current_time = datetime.now().strftime("%m%d%H%M")
                np.savetxt(os.path.join(directory, f'sub2sub_new_acc_labeled_{n_labeled_per_class}_{current_time}.csv'),
                               acc_array, delimiter=",", fmt="%.4f")
                np.savetxt(os.path.join(directory, f'sub2sub_new_loss_labeled_{n_labeled_per_class}_{current_time}.csv'),
                               loss_array, delimiter=",", fmt="%.4f")
                np.savetxt(os.path.join(directory, f'sub2sub_new_acc_labeled_{n_labeled_per_class}_{current_time}.csv'),
                           test_acc, delimiter=",", fmt="%.4f")
                np.savetxt(os.path.join(directory, f'sub2sub_new_loss_labeled_{n_labeled_per_class}_{current_time}.csv'),
                           test_loss, delimiter=",", fmt="%.4f")

        # elif args.dataset == 'SEED-IV': #original
        #     acc_array = np.zeros((dataset_dict['Subject_No'], dataset_dict['Session_No'], args.epochs))
        #
        #     for subject_num in (range(1, dataset_dict['Subject_No']+1)):
        #         for session_num in range(1, dataset_dict['Session_No']+1):
        #
        #             Net = net_init(Conv_EEG) #initializing the network
        #
        #             X_train = np.load(train_de.format(subject_num, session_num))
        #             X_test  = np.load(test_de.format(subject_num, session_num))
        #
        #
        #             X = np.vstack((X_train, X_test))
        #             X = np.reshape(X, (-1,310)) # (trial , channel, frequency band ) -> (trial , ch x freq_b ): ch = 62, freq_b = 5
        #
        #             scaler = MinMaxScaler()
        #             X = scaler.fit_transform(X)
        #
        #             X_train = X[0: X_train.shape[0]]
        #             X_test  = X[X_train.shape[0]:]
        #
        #             Y_train = np.load(train_label.format(subject_num, session_num))
        #             Y_test  = np.load(test_label.format(subject_num, session_num))
        #
        #             #split dataset as unlabeled / labeled , and  label guessing with ssl
        #             acc_array[subject_num-1, session_num-1]  =  ssl_process(Net, X_train, X_test, Y_train, Y_test, n_labeled_per_class, random_seed, subject_num, session_num)
        #
        #             torch.cuda.empty_cache()
        #
        #     print(acc_array.shape)
        #     #save last epoch's acc
        #     np.savetxt(os.path.join(directory, 'acc_labeled_{}.csv').format(n_labeled_per_class), acc_array[-1] , delimiter=",")


        elif args.dataset == 'SEED-V':
            acc_array = np.zeros((dataset_dict['Subject_No'], dataset_dict['Fold_No'], args.epochs))

            for subject_num in range(1, dataset_dict['Subject_No']+1):

                X1 = np.load(data_addr.format(subject_num, 1))
                X2 = np.load(data_addr.format(subject_num, 2))
                X3 = np.load(data_addr.format(subject_num, 3))

                X  = np.vstack((X1, X2, X3))

                Y1 = np.load(label_addr.format(subject_num, 1))
                Y2 = np.load(label_addr.format(subject_num, 2))
                Y3 = np.load(label_addr.format(subject_num, 3))

                Y  = np.vstack((Y1, Y2, Y3))

                scaler=MinMaxScaler()
                X = scaler.fit_transform(X)

                for fold_num in range(dataset_dict['Fold_No']):

                    Net = net_init(Conv_EEG)

                    optimizer = optim.Adam(Net.parameters(), lr=args.lr)
                    # ema_optimizer= WeightEMA(Net, ema_Net, alpha=args.ema_decay)

                    fold_1_index = [i for i in range(0, len(X1))]
                    fold_2_index = [i for i in range(len(X1), len(X1)+len(X2))]
                    fold_3_index = [i for i in range(len(X1)+len(X2), len(X1)+len(X2)+len(X3))]

                    if fold_num ==0:
                        train_index, test_index = fold_1_index + fold_2_index, fold_3_index
                    elif fold_num ==1:
                        train_index, test_index = fold_2_index + fold_3_index, fold_1_index
                    else:
                        train_index, test_index = fold_3_index + fold_1_index, fold_2_index


                    X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]

                    acc_array[subject_num-1, fold_num] = ssl_process(Net, X_train, X_test, Y_train, Y_test, n_labeled_per_class, random_seed)

            np.savetxt(os.path.join(directory, 'acc_labeled_{}.csv').format(n_labeled_per_class), acc_array , delimiter=",")


        elif args.dataset == 'AMIGOS':

            '''P8, P24 and P28 were not excluded since these participants did not took part in the long videos experiment.'''
            '''As the result, data from 37 participants were used in our experiemnts, see http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/readme.html for more details'''

            exclude_list = [8, 24, 28]

            X = np.zeros((0, dataset_dict['Feature_No'])) # 105 extracted features
            Y = np.zeros((0, dataset_dict['Class_No']))    # two label categories: Valence and Arousal

            for participant in (range(1, 41)):
                if not any(participant == c for c in exclude_list):

                    temp_X = np.load(data_addr.format(participant))
                    temp_Y = np.load(label_addr.format(participant))

                    X = np.vstack((X, temp_X))
                    Y = np.vstack((Y, temp_Y))

            scaler=MinMaxScaler()
            X = scaler.fit_transform(X)


            X = np.reshape(X, (dataset_dict['Subject_No'], dataset_dict['Segment_No'], dataset_dict['Feature_No']))
            Y = np.reshape(Y, (dataset_dict['Subject_No'], dataset_dict['Segment_No'], dataset_dict['Class_No']))


            loo = LeaveOneOut()

            for label_index in range(dataset_dict['Class_No']):
                Y = Y[:,:,label_index]
                # Y= to_categorical(Y)

                f1_array  = np.zeros((dataset_dict['Subject_No'],args.epochs))
                count = 0
                for train_index, test_index in tqdm(loo.split(X)):

                    Net = net_init(Conv_EEG)

                    X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
                    # print(X_test.shape)


                    X_train  = np.reshape(X_train, (X_train.shape[0]*X_train.shape[1], -1))
                    X_test   = np.reshape(X_test, (X_test.shape[0]*X_test.shape[1],-1))
                    Y_train = np.reshape(Y_train, (Y_train.shape[0]*Y_train.shape[1], -1))
                    Y_test  = np.reshape(Y_test, (Y_test.shape[0]*Y_test.shape[1], -1))

                    # print(Y_train)
                    '''check nan in EEG_start'''

                    nan_list = []
                    for i in range(len(X_train)):
                        if np.isnan(X_train[i]).any():
                            nan_list.append(i)

                    X_train  = np.delete(X_train,  nan_list, axis=0)
                    Y_train  = np.delete(Y_train,  nan_list, axis=0)


                    nan_list = []
                    for i in range(len(X_test)):
                        if np.isnan(X_test[i]).any():
                            nan_list.append(i)

                    X_test  = np.delete(X_test,  nan_list, axis=0)
                    Y_test  = np.delete(Y_test,  nan_list, axis=0)


                    '''check nan in EEG_end'''
                    # print(data_train.shape, data_test.shape)

                    f1_array[count] = ssl_process(Net, X_train, X_test, Y_train, Y_test, n_labeled_per_class, random_seed)


                    count=count+1
                    torch.cuda.empty_cache()

                    if label_index == 0:
                        f1_addr = 'f1_valance_labeled_{}.csv'
                    else:
                        f1_addr = 'f1_arousal_labeled_{}.csv'

                np.savetxt(os.path.join(directory, f1_addr).format(n_labeled_per_class), f1_array, delimiter=",")

        else:
            raise Exception('Datasets Name Error')



pbar_seed.close()
#

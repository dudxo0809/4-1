import sys
sys.path.append('C:/Users/김영태/Desktop/4-1/캡스톤/STGCN/STGCN-Tf2-main')
path = 'C:/Users/김영태/Desktop/4-1/캡스톤/STGCN/STGCN-Tf2-main'

import numpy as np
import pandas as pd

import argparse
import datetime
from model.trainer import model_train
from model.tester import model_test
from data_loader.data_utils import *
from utils.math_graph import *
import tensorflow as tf
from os.path import join as pjoin
import os

if tf.test.is_built_with_cuda():
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    print("Has GPU!!!")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    print("Has No GPU!!!")
        
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dataset_path = pjoin(path, 'dataset')


v_file = f'PeMSD7_V_228.csv'
w_file = f'PeMSD7_W_228.csv'

'''
v_file = f'train1.csv'
w_file = f'w1.csv'
'''
file_path = pjoin(dataset_path, v_file)
n_feature = (pd.read_csv(file_path, header=None)).shape[1]

print(f"This data has {n_feature} features !!")

epochs = 30
n_route = n_feature

parser = argparse.ArgumentParser()
parser.add_argument('--n_route', type=int, default= n_route)
parser.add_argument('--n_his', type=int, default=12)
parser.add_argument('--n_pred', type=int, default=9)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default= epochs)
parser.add_argument('--ks', type=int, default=3)
parser.add_argument('--kt', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='RMSprop')
parser.add_argument('--inf_mode', type=str, default='merge')
parser.add_argument('--datafile', type=str, default=v_file)
parser.add_argument('--graph', type=str, default= w_file)
parser.add_argument('--scale_graph', action='store_true', default=False)
parser.add_argument('--channels', type=int, default=1)
parser.add_argument('--logs', type=str, default= os.path.join(path, 'output').replace("\\", "/"))

args = parser.parse_args()
print(f'Training configs: {args}')

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
args.logs = os.path.join(args.logs, current_time)

args.model_path = os.path.join(args.logs, 'model').replace("\\", "/")
args.logs = os.path.join(args.logs, 'logs').replace("\\", "/")

#args.model_path = pjoin(args.model_path)

os.makedirs(args.model_path, exist_ok=True)
os.makedirs(args.logs, exist_ok=True)
print(args.model_path)
print(args.logs)
n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
Ks, Kt = args.ks, args.kt
# blocks: settings of channel size in st_conv_blocks / bottleneck design
blocks = [[args.channels, 32, 64], [64, 32, 128]]

'''
Test correlation matrix -> weighted adjacency matrix
'''
W = weight_matrix_by_correlation(pjoin(dataset_path, args.datafile), scaling=args.scale_graph)
#W = weight_matrix(pjoin(dataset_path, args.graph), scaling=args.scale_graph)
# Calculate graph kernel
L = scaled_laplacian(W)
# Alternative approximation method: 1st approx - first_approx(W, n).
Lk = cheb_poly_approx(L, Ks, n)

PeMS = data_gen(pjoin(dataset_path, args.datafile), n, n_his + n_pred, args.channels)
print(f'>> Loading dataset with Mean: {PeMS.mean:.2f}, STD: {PeMS.std:.2f}')
print("Train shape:", PeMS.get_data("train").shape)
print("Val shape:", PeMS.get_data("val").shape)

if __name__ == '__main__':
    model_train(PeMS, Lk, blocks, args)
    model_test(PeMS, PeMS.get_len('test'), n_his, n_pred, args.inf_mode, args.model_path)

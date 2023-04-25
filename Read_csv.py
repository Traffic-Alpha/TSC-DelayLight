'''
@Author: Pang Aoyu
@Date: 2023-03-06 13:47:23
@Description: 测试不同的模型在不同环境下的结果
@LastEditTime: 2023-03-06 14:12:54
'''
import argparse
import shutil
import  xml.dom.minidom
import csv
import os
import numpy as np
import pandas as pd

from aiolos.utils.get_abs_path import getAbsPath
from aiolos.trafficLog.initLog import init_logging
pathConvert = getAbsPath(__file__)


from env import makeENV
from create_params import create_test_params
from SumoNets.NET_CONFIG import SUMO_NET_CONFIG

def  readData(
        model_name, net_name, net_env,
        n_stack, n_delay,
        singleEnv=False, fineTune=False,
    ):
    if model_name == 'None':
        model_name = ' '
    assert model_name in ['scnn', 'ernn', 'eattention', 'ecnn', 'inference', 'predict','ernn_P', 'ernn_C', 'inference_scnn'], f'Model name error, {model_name}'
    # args, 这里为了组合成模型的名字
    N_STACK = n_stack # 堆叠
    N_DELAY = n_delay # 时延
    FOLDER_NAME = net_env
    _net_name = net_name.split('.')[0] # 得到 NET 的名称
    #将节点取值 写入文档，但是有些繁琐
    root_PATH=pathConvert(f'./results/testData-mean_2/{model_name}_{n_stack}_test/')
    Data_PATH=pathConvert(f'./results/testData-mean_2/{model_name}_{n_stack}_test/{net_env}_{_net_name}.csv')
    if not os.path.exists(root_PATH):
        print('Data is not excit.')
    W=[]
    for i in range(0,5):
        W_temp=get_WT(Data_PATH,i)
        W_temp=np.array(W_temp)
        W_temp=W_temp.T
        W_temp=W_temp.reshape((-1,15))
        W_median_temp=np.median(W_temp,axis=0)
        W.append(W_median_temp)
    W_all=np.array(W)
    #print('all',W_all)
    #print('all',W_all.shape)
    #W_all=W_all.T
    #W_all=W_all.reshape((-1,15))
    #print('all',W_all)
    #print('all',W_all.shape)
    #W_median=np.median(W_all,axis=0)
    W_mean=W_all.sum(axis=0)/len(W_all) #按照列求和，归为第二维度上，然后求平均
    #print('mean',W_mean)
    _net_name = net_name.split('.')[0] # 得到 NET 的名称
    SAVE_PATH_root=pathConvert(f'./results/testData_median/{model_name}_{n_stack}_test/')
    if not os.path.exists(SAVE_PATH_root):
        os.makedirs(SAVE_PATH_root)
    W_all=W_all.T
    W_all=np.sort(W_all,axis=1)
    #print(W_all)
    np.savetxt(f'{SAVE_PATH_root}/{net_env}_{_net_name}.csv', W_mean, delimiter=' ', fmt='%.02f')

def get_WT(data_path='output_path',i=0):
    #获取节点取值
    dom = pd.read_csv(data_path,header=None)
    #print(dom)
    WT=dom[i]
    WT=WT[WT.index%2==0] #偶数行是全部数据，奇数行是已经求好的平均值
    #print(WT)
    return WT


   
if __name__ == '__main__':
    


    parser = argparse.ArgumentParser()

    parser.add_argument('--stack', type=int, default=6)
    parser.add_argument('--delay', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='ernn_P')
    parser.add_argument('--net_env', type=str, default='train_four_345')
    parser.add_argument('--net_name', type=str, default='4phases.net.xml')
    parser.add_argument('--singleEnv', default=False, action='store_true')
    parser.add_argument('--fineTune', default=False, action='store_true')
    args = parser.parse_args()

    readData(
        net_env=args.net_env,
        model_name=args.model_name, net_name=args.net_name,
        n_stack=args.stack, n_delay=args.delay,
        singleEnv=args.singleEnv, fineTune=args.fineTune
    )
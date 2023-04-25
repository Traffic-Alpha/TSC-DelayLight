'''
@Author: WANG Maonan
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
        model_name = ''
    assert model_name in ['scnn', 'ernn', 'eattention', 'ecnn', 'inference', 'predict', 'ernn_P', 'ernn_C', 'inference_scnn'], f'Model name error, {model_name}'
    # args, 这里为了组合成模型的名字
    N_STACK = n_stack # 堆叠
    N_DELAY = n_delay # 时延
    FOLDER_NAME = net_env
    W=[]
    for i in range(0,5): # 仅仅只读取一个路网
        route_name = SUMO_NET_CONFIG[FOLDER_NAME]['routes'][i] #选取绘制的车流数据
        _net_name = net_name.split('.')[0] # 得到 NET 的名称
        _route_name = route_name.split('.')[0] # 得到车流
        output_path = pathConvert(f'./results/test/output/{model_name}/{net_env}_{net_name}_{N_STACK}_{N_DELAY}/{_net_name}/{_route_name}/')
        WT=get_WT(output_path)
        WT=float(WT)
        W.append(WT)
    #将节点取值 写入文档，但是有些繁琐
    root_PATH=pathConvert(f'./results/testData_mean_3/{model_name}_{n_stack}_test/')
    if not os.path.exists(root_PATH):
        os.makedirs(root_PATH)
    W_all=np.array(W)
    W_mean=W_all.sum()/len(W_all)
    W=[]
    W.append(W_mean)
    _net_name=net_name.split('.')[0]
    WT_PATH=pathConvert(f'{root_PATH}/{net_env}_{_net_name}.csv')
    f=open(WT_PATH, 'a+')
    writer = csv.writer(f)
    writer.writerow(W_all)# 输出全部的数据
    writer.writerow(W)# 输出平均值
    f.close()
def get_WT(output_path='output_path'):
    #获取节点取值
    dom = xml.dom.minidom.parse(output_path+'/statistic.out.xml')
    root = dom.documentElement
    VTS=root.getElementsByTagName('vehicleTripStatistics')#获取节点
    WT=VTS[0].getAttribute("waitingTime")#h获取节点值
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
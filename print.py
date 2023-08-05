'''
@Author: Pang Aoyu
@Date: 2023-04-24 13:47:23
@Description: 绘制测试数据的图像
@LastEditTime: 2023-04-24 14:12:54
'''
import argparse
import shutil
import  xml.dom.minidom
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

from aiolos.utils.get_abs_path import getAbsPath
from aiolos.trafficLog.initLog import init_logging
pathConvert = getAbsPath(__file__)

from env import makeENV
from create_params import create_test_params
from SumoNets.NET_CONFIG import SUMO_NET_CONFIG

model_names=['scnn', 'ernn', 'eattention', 'ecnn', 'inference', 'predict','ernn_P', 'ernn_C', 'inference_scnn'] #需要读取的模型的名字
#model_names=[ 'ernn_C'] 
net_name=['4phases.net.xml','6phases.net.xml','3phases.net.xml']
net_env=['train_four_345','train_four_345','train_three_3']
n_stack = 8 # 堆叠
N_DELAY = [0, 1, 2, 3, 4, 8, 12, 16,  20,  24] # 时延
FOLDER_NAME = []
_net_name = net_name[0].split('.')[0]
_net_env=net_env[0]
total_data=[]
Label=[]
for model_name in model_names :
 Data_PATH=pathConvert(f'./results/testData_median/{model_name}_{n_stack}_test/{_net_env}_{_net_name}.csv')
 if not os.path.exists(Data_PATH):
  print('model',model_name)
  print('net_env',_net_env)
  print('net_name',_net_name)
  print('it is not exit')
  continue
 else:
   Label.append(model_name)
 temp_data=pd.read_csv(Data_PATH,header=None)
 #print(temp_data)
 temp_data=np.array(temp_data, dtype=float)
 #print(temp_data)
 #temp_data=temp_data.sum(axis=0)/len(temp_data)
 total_data.append(temp_data)
total_data=np.array(total_data)
print(total_data.shape)
total_data=total_data.reshape((-1,10))
df = pd.DataFrame(total_data.T,columns=Label,dtype=float)
print(df)
print(total_data.shape)
plt.plot(total_data.T, label=Label)
PNG_DATA=pathConvert(f'./results/png/')
if not os.path.exists(PNG_DATA):
    os.makedirs(PNG_DATA)
plt.legend(loc='upper left',frameon=True,edgecolor='black',facecolor='white',framealpha=1) 
plt.savefig(f'{PNG_DATA}/{_net_name}_total_data.png')
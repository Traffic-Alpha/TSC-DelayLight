'''
@Author: PANG Aoyu
@Date: 2023-03-31 
@Description: SCNN, use multi-channels to extract infos
@LastEditTime: 2023-03-31
'''
import gym
import numpy as np
import pandas as pd
import csv

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from aiolos.utils.get_abs_path import getAbsPath
pathConvert = getAbsPath(__file__)


class FlowData(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.Space, features_dim: int = 64, FLOWDATA_PATH: str='FLOWDATA_PATH'):
        """特征提取网络
        """
        super().__init__(observation_space, features_dim)
        net_shape = observation_space.shape # 每个 movement 的特征数量, 8 个 movement, 就是 (N, 8, K)
        # 这里 N 表示由 N 个 frame 堆叠而成

        self.view_conv = nn.Sequential(
            nn.Conv2d(net_shape[0], 128, kernel_size=(1, net_shape[-1]), padding=0), # N*8*K -> 128*8*1
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(8,1), padding=0), # 128*8*1 -> 256*1*1
            nn.ReLU(),
        )
        view_out_size = self._get_conv_out(net_shape)

        self.fc = nn.Sequential(
            nn.Linear(view_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim)
        )

        self.f = open(FLOWDATA_PATH, 'a+')
        


     # state [flow, mean_occupancy, max_occupancy, is_s, num_lane, mingreen, is_now_phase, is_next_phase]
    
    def data_out(self,observaions):

        writer = csv.writer(self.f)

        net_shape = observaions.shape
        #obs_out=observaions[-1,-1,:,0:3].clone()   #不错这么多改变 直接输出全部
        temp=observaions.cpu().clone()
        obs_out=temp[-1,-1,:,:].clone() #输出最后一片
        #obs_out[:,-1]=observaions[-1,-1,:,6].clone()
        obs_out=obs_out.reshape(-1)
        obs_out=np.array(obs_out)
        writer.writerow(obs_out)

        #print("obs_out",obs_out)
    

       

    def _get_conv_out(self, shape):
        o = self.view_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))


    def forward(self, observations):
        batch_size = observations.size()[0] # (BatchSize, N, 8, K)
        self.data_out(observations)
        #obs=observations[:,-1,:,:].clone() # 放入最后一片 
        #obs=obs.reshape((batch_size,1,observations.size()[2],observations.size()[3]))  #用RL+CNN
        conv_out = self.view_conv(observations).view(batch_size, -1)
        return self.fc(conv_out)

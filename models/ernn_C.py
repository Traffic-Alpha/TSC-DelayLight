'''
@Author: WANG Maonan
@Date: 2022-04-12 12:02:58
@Description: Use CNN to extract features and then use RNN
@LastEditTime: 2022-06-17 14:47:22
'''
import gym
import numpy as np

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ERNN_C(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 64):
        """利用 RNN 提取信息
        """
        super().__init__(observation_space, features_dim)
        self.net_shape = observation_space.shape # 每个 movement 的特征数量, 8 个 movement, 就是 (N, 8, K)
        # 这里 N 表示由 N 个 frame 堆叠而成

        self.view_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, self.net_shape[-1]), padding=0), # N*1*8*K -> N*64*8*1
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(8,1), padding=0), # N*64*8*1 -> N*128*1*1 (BatchSize, N, 128, 1, 1)
            nn.ReLU(),
        ) # 每一个 junction matrix 提取的特征
        view_out_size = self._get_conv_out(self.net_shape)
        LSTM_size=self.net_shape[1]*self.net_shape[2]

        self.extract_time_info = nn.LSTM(
                input_size=LSTM_size, 
                hidden_size=LSTM_size, 
                num_layers=2,
                batch_first=True
            )
        
        self.fc = nn.Sequential(
            nn.Linear(view_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim)
        )


    def _get_conv_out(self, shape):
        o = self.view_conv(torch.zeros(1, 1, *shape[1:])) # 进入卷积 看一下输出的结果
        return int(shape[0]*np.prod(o.size()))
    
    
    def forward(self, observations):
        batch_size = observations.size()[0] # (BatchSize, N, 8, K)
        N=observations.size()[1]
        # observations = torch.unsqueeze(observations,2) # (BatchSize, N, 1, 8, K)

        observations = observations.view(batch_size, N, -1) # (BatchSize, N, 8*K)
        
        lstm_out, _ = self.extract_time_info(observations)
        #print('lstm_out',lstm_out.size())
        lstm_out = lstm_out.reshape(batch_size*N, 1, 8, self.net_shape[-1]) # (BatchSize*N, 1, 8, K)  
        #RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead. 为什么用reshape可以 用view不可以
        #print('lstm_out',lstm_out.size())

        conv_out = self.view_conv(lstm_out).view(batch_size, self.net_shape[0], -1) # (BatchSize*N, 256) --> (BatchSize, N, 256)
        
        conv_out=conv_out.view(batch_size,-1)

        return self.fc(conv_out)
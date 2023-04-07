'''
@Author: PANG Aoyu
@Date: 2023-03-31 
@Description: SCNN, use multi-channels to extract infos
@LastEditTime: 2023-03-31
'''
import gym
import numpy as np

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Inference(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.Space, features_dim: int = 64):
        """特征提取网络
        """
        super().__init__(observation_space, features_dim)
        net_shape = observation_space.shape # 每个 movement 的特征数量, 8 个 movement, 就是 (N, 8, K)
        # 这里 N 表示由 N 个 frame 堆叠而成

        self.view_conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(1, net_shape[-1]), padding=0), # N*8*K -> 128*8*1
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(8,1), padding=0), # 128*8*1 -> 256*1*1
            nn.ReLU(),
        )

        view_out_size = self._get_conv_out(net_shape[1:]) #只放入一片

        self.fc = nn.Sequential(
            nn.Linear(view_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim)
        )
        


     # state [flow, mean_occupancy, max_occupancy, is_s, num_lane, mingreen, is_now_phase, is_next_phase]
    def infe(self,observaions):
        # 出现问题 这么写如果有GPU 就一定在GPU上跑
     
        net_shape = observaions.shape
        obs_inf=observaions[:,-1,:,:].clone()



        obs_inf[:,:,3:7]=observaions[:,-1,:,3:7].clone()
        
        #is_now_phase=torch.where(is_now_phase>0.5, 1, is_now_phase)
        #is_now_phase=torch.where(is_now_phase<0.5, -1, is_now_phase) # 不知道红绿灯的数值
        
        #import pdb; pdb.set_trace() # 打断点
        obs_inf[:,:,0]=self._get_weight_sum(observaions[:,:,:,0].clone()) #对flow 进行加权和
        obs_inf[:,:,1]=self._get_weight_sum(observaions[:,:,:,1].clone()) #对mean_occypancy进行加权和
        obs_inf[:,:,2]=self._get_weight_sum(observaions[:,:,:,2].clone()) #对max_occupancy进行加权和

        return obs_inf

        '''
        # 对occupancy 进行估计
        phase_mean_occupay=0.2
        mean_occupancy=mean_occupancy+is_now_phase*phase_mean_occupay
        mean_occupancy=torch.where(mean_occupancy<0,0,mean_occupancy)
        obs_inf[:,:,1]=mean_occupancy
        obs_inf=obs_inf
        '''
    
    
    
    def _get_weight_sum(self,data): # 得到data数据的加权和

        shape=data.shape
        W=torch.arange(start=1, end=shape[1]+1, step=1).to(device)# 将数据传入GPU
        W_tensor=torch.zeros(shape[0:3]).to(device)  #生成 batch*N*8 的权重
        W=W/((shape[1]/2)*(shape[1]+1))  #生成权重，使权重和为1
        for i in range(0,len(W)):
            W_tensor[:,i,:]=W[i]
        data=data*W_tensor
        data=torch.sum(data,dim=1)# 对数据进行加权求和
        
        return data


    def _get_conv_out(self, shape):
        o = self.view_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))


    def forward(self, observations):
        batch_size = observations.size()[0] # (BatchSize, N, 8, K)
        obs_inference=self.infe(observations)
        #print("obs_ore",observations[-1,-1,:,:])
        #print("obs_inf",obs_inference[-1])
        obs_inference=obs_inference.reshape((batch_size,1,8,8)) # 放入推断后的那片
        conv_out = self.view_conv(obs_inference).view(batch_size, -1)
        return self.fc(conv_out)

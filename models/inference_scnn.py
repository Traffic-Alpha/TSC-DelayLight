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
#device=torch.device('cpu')

class Inference_SCNN(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.Space, features_dim: int = 64, action_space: int=4):
        """特征提取网络
        """
        super().__init__(observation_space, features_dim)
        net_shape = observation_space.shape # 每个 movement 的特征数量, 8 个 movement, 就是 (N, 8, K)
        # 这里 N 表示由 N 个 frame 堆叠而成
        #print('net_shape',net_shape)
        #net_shape=net_shape[0]
        self.action_space=action_space
        self.view_conv = nn.Sequential(
            nn.Conv2d(net_shape[0]+1, 128, kernel_size=(1, net_shape[-1]), padding=0), # N*8*K -> 128*8*1
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(8,1), padding=0), # 128*8*1 -> 256*1*1
            nn.ReLU(),
        )

        view_out_size = self._get_conv_out([net_shape[0]+1,net_shape[1],net_shape[2]])

        self.fc = nn.Sequential(
            nn.Linear(view_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim))
        


    def infe(self,observaions):
        # 出现问题 这么写如果有GPU 就一定在GPU上跑
        net_shape = observaions.shape
        features = torch.randn(net_shape[0],net_shape[1]+1,net_shape[2],net_shape[3]).to(device)
        obs_inf=observaions[:,-1,:,:].clone()
        obs_inf[:,:,3:7]=observaions[:,-1,:,3:7].clone()
        obs_inf[:,:,0]=self.SV1_judge(observaions) #对flow 进行加权和
        obs_inf[:,:,1]=self._get_weight_sum(observaions[:,:,:,1].clone()) #对mean_occypancy进行加权和
        obs_inf[:,:,2]=self._get_weight_sum(observaions[:,:,:,2].clone()) #对max_occupancy进行加权和
        obs_inf[:,:,7]=self.phase_judge(observaions[:,-1,:,:].clone())
        #obs_inf=obs_inf.view(net_shape[0],1,net_shape[2],net_shape[3])
        features[:,0:net_shape[1],:,:]=observaions[:,:,:,:].clone() 
        #print('features',features.shape)
        #print('obs_inf',obs_inf.shape)
        features[:,-1,:,:]=obs_inf #放入最后一片
        #print('success')
        return features
    def SV1_judge(self,obs):
        shape=obs.shape
        batch=obs.shape[0]
        SV1 = torch.zeros(shape[0],shape[2]).to(device)
        SV1_temp=self._get_weight_sum(obs[:,:,:,0].clone()) 
        phase_list=obs[:,-1,:,6].clone()
        for i in  range(0,batch):
            SV1[i]=phase_list[i]*SV1_temp[i]
        return SV1
    def phase_judge(self,obs):
    #print('obs',obs)
        shape=obs.shape
        batch=obs.shape[0]
        phase_inf = torch.zeros(shape[0],shape[2]).to(device)
        #occupancys=obs[:,:,1]
        #now_pahse=obs[:,:,6]
        #obs_temp=obs[:,-1,:,:]
        #occupancys=occupancys.view(batch,-1)
        for i in  range(0,batch):
            phase_list=self.get_phase(self.action_space)
            phase_inf[i]=self.action_judge(obs[i,:,:],phase_list)
        #print('phase_inf',phase_inf)
        #phase_inf=phase_inf.to(device)
        return phase_inf
    def get_phase(self,action_space):
        phases_6=[[1, 1, 0, 0, 0, 0, 0, 0],
                [ 0, 1, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0]]
        phases_4=[[0, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 1, 0, 0, 0]]
        phases_3=[[0, 1, 0, 1,0,0,0,0],
                [1, 0, 0, 0,0,0,0,0],
                 [0, 0, 1, 0,0,0,0,0]]
        if(action_space==6):
            phase_list=phases_6
        if(action_space==4):
            phase_list=phases_4
        if(action_space==3):
            phase_list=phases_3
        phase_list=torch.Tensor(phase_list).to(device)
        return phase_list
    def action_judge(self,obs,phase_list):
        #obs=obs[:,-1,:,:]
        occupancy=obs[:,1]
        now_pahse=obs[:,6]
        occupancy=occupancy.reshape(-1)
        #print('now_action',action)
        #print('occupancy',occupancy.shape)
        occupancy_list=torch.zeros(phase_list.shape[0])
        #print('phase_list.shape[0]',phase_list.shape[0])
        for i in range(0,phase_list.shape[0]):
            occupancy_list[i]=(occupancy*phase_list[i]).sum()
        action=torch.argmax(occupancy_list)

        return phase_list[action]
    
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
        net_shape=observations.size()
        obs_inference=self.infe(observations)
        
        #obs_inference=obs_inference.reshape((batch_size,8,8)) # 放入推断后的那片


        #observations[:,0,::]=obs_inference
        
        conv_out = self.view_conv(obs_inference).view(batch_size, -1)
        return self.fc(conv_out)

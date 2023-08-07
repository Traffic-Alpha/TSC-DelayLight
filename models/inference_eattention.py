'''
@Author: PangAY
@Date: 2023-08-03 12:02:58
@Description: infe_Eattention
@LastEditTime: 2023-08-03 13:48:42
'''
import gym
import numpy as np

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device= torch.device('cpu')
class Infer_EAttention(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 64, action_space: int=4):

        super().__init__(observation_space, features_dim)
        self.net_shape = observation_space.shape # 每个 movement 的特征数量, 8 个 movement, 就是 (N, 8, K)
        # 这里 N 表示由 N 个 frame 堆叠而成
        self.frame_inf=self.net_shape[0]+1
        self.action_space=action_space
        self.view_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, self.net_shape[-1]), padding=0), # N*1*8*K -> N*64*8*1
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(8,1), padding=0), # N*64*8*1 -> N*128*1*1 (BatchSize, N, 128, 1, 1)
            nn.ReLU(),
        ) # 每一个 junction matrix 提取的特征
        view_out_size = self._get_conv_out(self.net_shape)

        self.junction_embedding = nn.Linear(view_out_size, 64)
        self.pos_embed = nn.Parameter(
                torch.zeros(1, self.frame_inf, 64)
        ) # 可以学习的位置编码
        encoder_layer = nn.TransformerEncoderLayer(
                                        d_model=64, # the number of expected features in the input
                                        nhead=8,
                                        dim_feedforward=32, # the dimension of the feedforward network model 
                                        activation='gelu',
                                        dropout=0.2,
                                        batch_first=True
                                    )
        self.encoder = nn.TransformerEncoder(
                            encoder_layer=encoder_layer, 
                            num_layers=2
                        )
        self.layernorm1 = nn.LayerNorm(64, eps=1e-6)
        self.map_feature = nn.Linear(64*self.frame_inf, features_dim)
        self.layernorm2 = nn.LayerNorm(features_dim, eps=1e-6)
    
    def infe(self,observaions):
        # 出现问题 这么写如果有GPU 就一定在GPU上跑
        net_shape = observaions.shape
        features = torch.randn(net_shape[0],net_shape[1]+1,net_shape[2],net_shape[3]).to(device)
        obs_inf=observaions[:,-1,:,:].clone()
        obs_inf[:,:,3:7]=observaions[:,-1,:,3:7].clone()
        obs_inf[:,:,0]=self._get_weight_sum(observaions[:,:,:,0].clone()) #对flow 进行加权和
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
        o = self.view_conv(torch.zeros(1, 1, *shape[1:]))
        return int(np.prod(o.size()))


    def forward(self, observations):
        batch_size = observations.size()[0] # (BatchSize, N, 8, K)
        obs_inference=self.infe(observations)
        obs_inference = obs_inference.view(-1, 1, 8, self.net_shape[-1]) # (BatchSize*N, 1, 8, K)
        conv_out = self.view_conv(obs_inference).view(batch_size, self.net_shape[0]+1, -1) # (BatchSize*N, 128) --> (BatchSize, N, 128)
        # embedding
        #print('conv_out',conv_out.shape)
        x = self.junction_embedding(conv_out) # (BatchSize, N, 128) --> (BatchSize, N, 64)
        x = x + self.pos_embed # (BatchSize, N, 64), 加上位置编码
        x = self.layernorm1(self.encoder(x)).view(batch_size, -1) # (BatchSize, N, 64)  --> (BatchSize, N*64)
        x = self.layernorm2(self.map_feature(x)) # (BatchSize, N*64) --> (BatchSize, 32)
        return x
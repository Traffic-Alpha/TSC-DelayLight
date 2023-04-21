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
from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Predict(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.Space, features_dim: int = 64, PREDICT_MODEL_PATH: str='PREDICT_MODEL_PATH'):
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

        self.Flow_Predict = lstm_reg(net_shape[0],10)


        self.Flow_Predict.load_state_dict(torch.load(PREDICT_MODEL_PATH) )


     # state [flow, mean_occupancy, max_occupancy, is_s, num_lane, mingreen, is_now_phase, is_next_phase]
    def infe(self,observaions):
        # 出现问题 这么写如果有GPU 就一定在GPU上跑
     
        net_shape = observaions.shape
        obs_inf=observaions[:,-1,:,:].clone() #初始化空间
        obs_inf[:,:,3:7]=observaions[:,-1,:,3:7].clone()
        
        flow_data=observaions[:,:,:,0:3].clone() #取出需要的数据
        flow_data[:,:,:,-1]=observaions[:,:,:,6].clone()
        flow_data=flow_data.reshape((net_shape[0],-1,net_shape[1]))#更改为输入的维度 batchsize*（4*K）*8
        var_data = Variable(flow_data)
        #import pdb; pdb.set_trace()
        flow_predict=self.Flow_Predict(var_data)
        flow_predict=flow_predict.reshape((net_shape[0],net_shape[2],-1))

        #import pdb; pdb.set_trace() # 打断点
        obs_inf[:,:,0:2]=flow_predict[:,:,0:2]# 赋值 [flow, mean_occupancy, max_occupancy]
        obs_inf[:,:,6]=flow_predict[:,:,-1] # [is_now_phase]

        return obs_inf

    def _get_conv_out(self, shape):
        o = self.view_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))


    def forward(self, observations):
        batch_size = observations.size()[0] # (BatchSize, N, 8, K)
        obs_inference=self.infe(observations)
        obs_inference=obs_inference.reshape((batch_size,1,8,8)) # 放入推断后的那片
        conv_out = self.view_conv(obs_inference).view(batch_size, -1)
        return self.fc(conv_out)


class lstm_reg(nn.Module):
    def __init__(self,input_size=24,hidden_size=4, output_size=1,num_layers=2):
        super(lstm_reg,self).__init__()
            #super() 函数是用于调用父类(超类)的一个方法，直接用类名调用父类
        self.rnn = nn.LSTM(input_size,hidden_size,num_layers) #LSTM 网络
        self.reg = nn.Linear(hidden_size,output_size) #Linear 函数继承于nn.Module
    def forward(self,x):   #定义model类的forward函数
        x, _ = self.rnn(x)
        s,b,h = x.shape   #矩阵从外到里的维数
                    #view()函数的功能和reshape类似，用来转换size大小
        x = x.view(s*b, h) #输出变为（s*b）*h的二维
        x = self.reg(x)
        x = x.view(s,b,-1) #卷积的输出从外到里的维数为s,b,一列
        return x
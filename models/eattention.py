'''
@Author: WANG Maonan
@Date: 2022-04-12 12:02:58
@Description: Use CNN to extract features and then use Transformer rather than LSTM
@LastEditTime: 2023-03-06 13:48:42
'''
import gym
import numpy as np

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class EAttention(BaseFeaturesExtractor):
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

        self.junction_embedding = nn.Linear(view_out_size, 64)
        self.pos_embed = nn.Parameter(
                torch.zeros(1, self.net_shape[0], 64)
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
        self.map_feature = nn.Linear(64*self.net_shape[0], features_dim)
        self.layernorm2 = nn.LayerNorm(features_dim, eps=1e-6)


    def _get_conv_out(self, shape):
        o = self.view_conv(torch.zeros(1, 1, *shape[1:]))
        return int(np.prod(o.size()))


    def forward(self, observations):
        batch_size = observations.size()[0] # (BatchSize, N, 8, K)
        observations = observations.view(-1, 1, 8, self.net_shape[-1]) # (BatchSize*N, 1, 8, K)
        conv_out = self.view_conv(observations).view(batch_size, self.net_shape[0], -1) # (BatchSize*N, 128) --> (BatchSize, N, 128)
        # embedding
        x = self.junction_embedding(conv_out) # (BatchSize, N, 128) --> (BatchSize, N, 64)
        x = x + self.pos_embed # (BatchSize, N, 64), 加上位置编码
        x = self.layernorm1(self.encoder(x)).view(batch_size, -1) # (BatchSize, N, 64)  --> (BatchSize, N*64)
        x = self.layernorm2(self.map_feature(x)) # (BatchSize, N*64) --> (BatchSize, 32)
        return x
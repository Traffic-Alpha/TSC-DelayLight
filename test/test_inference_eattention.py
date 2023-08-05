'''
@Author: Pang Aoyu
@Date: 2022-04-04 12:02:58
@Description: 测试模型 inference 的输出 
@LastEditTime: 2022-06-21 22:05:00
'''
from aiolos.utils.get_abs_path import getAbsPath

pathConvert = getAbsPath(__file__)
import sys
sys.path.append(pathConvert('../'))

import gym
import torch
import numpy as np

from models.inference_eattention import Infer_EAttention


if __name__ == '__main__':
    # Input 是一个 N*8*8 的矩阵
    observation_space = gym.spaces.Box(
            low=0, 
            high=5,
            shape=(4,8,8)
        ) # obs 空间
    net = Infer_EAttention(observation_space, features_dim=32)

    movement_info = np.array([
        0.3*np.ones((4,8,8)), 
        0.5*np.ones((4,8,8)),
        0.7*np.ones((4,8,8)),
    ]) # (3, 4, 8, 8)

    movement_info = torch.from_numpy(movement_info).to(torch.float32) # batch_size*movement_num*movement_info_dim
    result = net(movement_info)
    print(result.shape) # 提取特征

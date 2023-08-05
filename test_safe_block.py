'''
@Author: WANG Maonan
@Date: 2023-03-06 13:47:23
@Description: 测试不同的模型在不同环境下的结果
@LastEditTime: 2023-03-06 14:12:54
'''
import argparse
import shutil
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from aiolos.utils.get_abs_path import getAbsPath
from aiolos.trafficLog.initLog import init_logging
pathConvert = getAbsPath(__file__)

from env import makeENV
from create_params import create_test_params
import numpy as np

def test_model(
        model_name, net_name, net_env,
        n_stack, n_delay,
        singleEnv=False, fineTune=False,
    ):
    if model_name == 'None':
        model_name = ''
    assert model_name in ['scnn', 'ernn', 'eattention', 'ecnn', 'inference', 'predict', 'ernn_P', 'ernn_C','inference_scnn', 'cnn_Pro'], f'Model name error, {model_name}'
    # args, 这里为了组合成模型的名字
    SAFE_BLOCk=True
    N_STACK = n_stack # 堆叠
    N_DELAY = n_delay # 时延
    Model_DELAY=0

    MODEL_PATH = pathConvert(f'./results/models/{model_name}/{net_env}_{net_name}_{N_STACK}_{Model_DELAY}/best_model.zip')
    VEC_NORM = pathConvert(f'./results/models/{model_name}/{net_env}_{net_name}_{N_STACK}_{Model_DELAY}/best_vec_normalize.pkl')
    LOG_PATH = pathConvert(f'./results/test/log/{model_name}/{net_env}_{net_name}_{N_STACK}_{N_DELAY}/') # 存放仿真过程的数据
    output_path = pathConvert(f'./results/test/output/{model_name}/{net_env}_{net_name}_{N_STACK}_{N_DELAY}/')

    eval_params = create_test_params(

            net_env=net_env, net_name=net_name, output_folder=output_path,
            N_DELAY=N_DELAY, N_STACK=N_STACK, 
            LOG_PATH=LOG_PATH,
        )
    for _key, eval_param in eval_params.items():
        # The environment for evaluating
        eval_env = SubprocVecEnv([makeENV.make_env(env_index=f'test_{N_STACK}_{N_DELAY}', **eval_param) for i in range(1)])#干什么用
        eval_env = VecNormalize.load(load_path=VEC_NORM, venv=eval_env) # 不进行标准化
        eval_env.training = False # 测试的时候不要更新
        eval_env.norm_reward = False
       
        #print('trip_info', eval_param['trip_info'])
        # ###########
        # start train
        # ###########
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PPO.load(MODEL_PATH, env=eval_env, device=device)
        action_space=eval_env.action_space.n    #获取action_space大小
        phase_list=get_phase(action_space)
        #import pdb; pdb.set_trace()
        # #########
        # 开始测试
        # #########
        obs = eval_env.reset()
        done = False # 默认是 False
        phase_time=np.ones(int(action_space))
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            #print('action_test',action, obs)
            if(SAFE_BLOCk==True and action_space== 4 ):
                action,phase_time=safe_judge(action,obs,phase_time,phase_list)
            # action = np.array([0]) # 对于 discrete 此时绿灯时间就是 5
            obs, reward, done, info= eval_env.step(action) # 随机选择一个动作, 从 phase 中选择一个 # 干什么用
            
        eval_env.close()

        # 拷贝生成的 tls 文件
        _net, _route = _key.split('__')
        
        shutil.copytree(
            src=pathConvert(f'./SumoNets/{net_env}/add/'),
            dst=f'{output_path}/add/',
            ignore=shutil.ignore_patterns('*.add.xml'),
            dirs_exist_ok=True,
        )
def get_phase(action_space):
    phases_6=[[1, 1, 0, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 1, 1],
       [0, 0, 0, 0, 1, 1, 0, 0],
       [0, 0, 0, 1, 0, 1, 0, 0],
       [0, 0, 1, 1, 0, 0, 0, 0]]
    phases_4=[[0, 1, 0, 0, 0, 0, 0, 1],
       [1, 0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 1, 0, 1, 0, 0],
       [0, 0, 1, 0, 1, 0, 0, 0]]
    phases_3=[[0, 1, 0, 1],
       [1, 0, 0, 0],
       [0, 0, 1, 0]]
    if(action_space==6):
        phase_list=phases_6
    if(action_space==4):
        phase_list=phases_4
    if(action_space==3):
        phase_list=phases_3
    phase_list=np.array(phase_list,float)
    return phase_list
def safe_judge(action,obs,phase_time,phase_list):
    #print('obs',obs)
    act=action[0]
    obs=obs[:,-1,:,:]
    occupancy=obs[:,:,1]
    now_pahse=obs[:,:,6]
    #print('occupancy',occupancy.shape)
    occupancy=occupancy.reshape(-1)
    #print('now_action',action)
    #print('occupancy',occupancy.shape)
    occupancy_list=np.ones(phase_list.shape[0])
    for i in range(0,phase_list.shape[0]):
        occupancy_list[i]=(occupancy*phase_list[i]).sum()
        phase_time=np.array(phase_time)
    max=phase_time.max()
    #加入时间长度控制，一个相位不能太久没被随机到
    if(max>=23):
        for item in range(0,phase_list.shape[0]):
            if phase_time[item]>=23:
                action[0]=item
    else:
        #选取的action的平均占有率不能过低
        if(occupancy_list[act]>0.3):
            action[0]=act
        else:
             action[0]=occupancy_list.argmax()
        #选取的相位不能持续过久时间 如果持续太久时间 则选取占有率第二位的
        if phase_time[action[0]]>=8:
            occupancy_list[action[0]]=0
            action[0]=occupancy_list.argmax()
    #参数更新
    phase_time[action[0]]=0
    phase_time=phase_time+np.ones(phase_list.shape[0])
    return action,phase_time

if __name__ == '__main__':
    init_logging(log_path=pathConvert('./test_log/'), log_level=0)
    parser = argparse.ArgumentParser()

    parser.add_argument('--stack', type=int, default=8)
    parser.add_argument('--delay', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='eattention')
    parser.add_argument('--net_env', type=str, default='train_four_345')
    parser.add_argument('--net_name', type=str, default='4phases.net.xml')
    parser.add_argument('--singleEnv', default=False, action='store_true')
    parser.add_argument('--fineTune', default=False, action='store_true')
    args = parser.parse_args()

    test_model(
        net_env=args.net_env,
        model_name=args.model_name, net_name=args.net_name,
        n_stack=args.stack, n_delay=args.delay,
        singleEnv=args.singleEnv, fineTune=args.fineTune
    )
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

from sumo_env import makeENV
from create_params import create_test_params

def test_model(
        model_name, net_name, net_env,
        n_stack, n_delay,
        singleEnv=False, fineTune=False,
    ):
    if model_name == 'None':
        model_name = ''
    assert model_name in ['scnn', 'ernn','eattention','ecnn','inference'], f'Model name error, {model_name}'
    # args, 这里为了组合成模型的名字
    N_STACK = n_stack # 堆叠
    N_DELAY = n_delay # 时延


    MODEL_PATH = pathConvert(f'./results/models/{model_name}/{net_env}_{net_name}_{N_STACK}_0/best_model.zip')
    VEC_NORM = pathConvert(f'./results/models/{model_name}/{net_env}_{net_name}_{N_STACK}_0/best_vec_normalize.pkl')
    LOG_PATH = pathConvert(f'./results/test/log_2/{model_name}/{net_env}_{net_name}_{N_STACK}_{N_DELAY}/') # 存放仿真过程的数据
    output_path = pathConvert(f'./results/test/output_2/{model_name}/{net_env}_{net_name}_{N_STACK}_{N_DELAY}/')

    eval_params = create_test_params(

            net_env=net_env, net_name=net_name, output_folder=output_path,
            N_DELAY=N_DELAY, N_STACK=N_STACK, 
            LOG_PATH=LOG_PATH,
        )
    for _key, eval_param in eval_params.items():
        # The environment for evaluating
        eval_env = SubprocVecEnv([makeENV.make_env(env_index=f'test_{N_STACK}_{N_DELAY}', **eval_param) for i in range(1)])#干什么用
        eval_env = VecNormalize.load(load_path=VEC_NORM, venv=eval_env) # 进行标准化
        eval_env.training = False # 测试的时候不要更新
        eval_env.norm_reward = False

        # ###########
        # start train
        # ###########
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PPO.load(MODEL_PATH, env=eval_env, device=device)

        # #########
        # 开始测试
        # #########
        obs = eval_env.reset()
        done = False # 默认是 False

        while not done:
            action, _state = model.predict(obs, deterministic=True)
            # action = np.array([0]) # 对于 discrete 此时绿灯时间就是 5
            obs, reward, done, info = eval_env.step(action) # 随机选择一个动作, 从 phase 中选择一个 # 干什么用
            
        eval_env.close()

        # 拷贝生成的 tls 文件
        _net, _route = _key.split('__')
        shutil.copytree(
            src=pathConvert(f'./SumoNets/{net_env}/add/'),
            dst=f'{output_path}/{_net}/{_route}/add/',
            ignore=shutil.ignore_patterns('*.add.xml'),
            dirs_exist_ok=True,
        )


if __name__ == '__main__':
    init_logging(log_path=pathConvert('./test_log/'), log_level=0)
    parser = argparse.ArgumentParser()

    parser.add_argument('--stack', type=int, default=6)
    parser.add_argument('--delay', type=int, default=1)
    parser.add_argument('--model_name', type=str, default='scnn')
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
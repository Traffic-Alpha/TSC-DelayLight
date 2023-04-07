'''
@Author: Pang Aoyu
@Date: 2023-04-01
@Description: 创建训练和测试环境的参数，这里有三个创建参数的函数：
- create_params，创建训练使用的参数
- create_test_params，创建测试使用的参数
- create_singleEnv_params，创建单个环境的参数
@LastEditTime: 2023-04-01 23:20:10
'''
import os
from aiolos.utils.get_abs_path import getAbsPath

from SumoNets.TRAIN_CONFIG import TRAIN_SUMO_CONFIG
#from SumoNets.EVAL_CONFIG import EVAL_SUMO_CONFIG
from SumoNets.NET_CONFIG import SUMO_NET_CONFIG # 测试模型时候训练和测试的路网一起进行测试
#from SumoNets.TEST_CONFIG import TEST_SUMO_CONFIG

def create_params(
        is_eval:bool, 
        N_STACK:int, N_DELAY:int, LOG_PATH:str, net_env:str,net_name:str,
    ):
    pathConvert = getAbsPath(__file__)

    FOLDER_NAME = net_env # 不同类型的路口
    cfg_name=net_env+'.sumocfg'
    tls_id = SUMO_NET_CONFIG[FOLDER_NAME]['tls_id'] # 路口 id
    #cfg_name = SUMO_NET_CONFIG[FOLDER_NAME]['sumocfg'] # sumo config
    #net_name = SUMO_NET_CONFIG[FOLDER_NAME]['nets'][0] # network file
    route_name = SUMO_NET_CONFIG[FOLDER_NAME]['routes'][0] # route file
    start_time = SUMO_NET_CONFIG[FOLDER_NAME]['start_time'] # route 开始的时间



    # 转换为文件路径
    cfg_xml = pathConvert(f'./SumoNets/{FOLDER_NAME}/env/{cfg_name}')
    net_xml = [pathConvert(f'./SumoNets/{FOLDER_NAME}/env/{net_name}')]
    route_xml = [pathConvert(f'./SumoNets/{FOLDER_NAME}/routes/{route_name}')]
    
    FOLDER_NAME= net_env
    net_name=net_name
    

    if is_eval: # 如果是测试的 reward，就只使用一个环境进行测试（使用 test 路网进行测试）
        env_dict = {
            _folder: {
            'cfg': pathConvert(f'./SumoNets/{FOLDER_NAME}/env/{cfg_name}'),
            'net':[pathConvert(f'./SumoNets/{FOLDER_NAME}/env/{net_name}')],
            'route':[pathConvert(f'./SumoNets/{FOLDER_NAME}/routes/{_route}') for _route in SUMO_NET_CONFIG[FOLDER_NAME]['routes']]
            }
            for _folder in ['train_four_345']
        }
    else: # 训练的时候多个路网同时进行训练
        

        env_dict = {
            FOLDER_NAME: {
            'cfg': pathConvert(f'./SumoNets/{FOLDER_NAME}/env/{cfg_name}'),
            'net':[pathConvert(f'./SumoNets/{FOLDER_NAME}/env/{net_name}')],
            'route':[pathConvert(f'./SumoNets/{FOLDER_NAME}/routes/{_route}') for _route in SUMO_NET_CONFIG[FOLDER_NAME]['routes']]
            }
        }
        '''
        net_name='4phases.net.xml'
        env_dict = {
            _folder: {
                'cfg': pathConvert(f'./SumoNets/{_folder}/env/{TRAIN_SUMO_CONFIG[_folder]["sumocfg"]}'),
                'net':[pathConvert(f'./SumoNets/{_folder}/env/{net_name}')],
                'route':[pathConvert(f'./SumoNets/{_folder}/routes/{_route}') for _route in TRAIN_SUMO_CONFIG[_folder]['routes']]
            }
            for _folder in [ 'train_four_345']
        }
        '''


    params = {
        'tls_id':tls_id,
        'begin_time':start_time,
        'num_seconds':3600,
        'sumo_cfg':cfg_xml,
        'net_files':net_xml,
        'route_files':route_xml,
        'num_stack':N_STACK,
        'num_delayed':N_DELAY,
        'is_libsumo':True,
        'use_gui':False,
        'min_green':5,
        'log_file':LOG_PATH,
        'env_dict':env_dict,
        'net_env':net_env
    }

    return params






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


def create_test_params(
        net_env:str,net_name:str, output_folder:str,
        N_STACK:int, N_DELAY:int, LOG_PATH:str
    ):
    """对训练路网和测试路网都进行测试，这里的 mode=eval
    """
    pathConvert = getAbsPath(__file__)

    FOLDER_NAME = net_env # 要测试的路网名称

    cfg_name=net_env+'.sumocfg'
    net_name=net_name
    tls_id = SUMO_NET_CONFIG[FOLDER_NAME]['tls_id'] # 路口 id
    #cfg_name = SUMO_NET_CONFIG[FOLDER_NAME]['sumocfg'] # sumo config
    #net_name = SUMO_NET_CONFIG[FOLDER_NAME]['nets'][0] # network file
    route_name = SUMO_NET_CONFIG[FOLDER_NAME]['routes'][4] # route file 用固定的第0个车流进行训练 1 2 3 4
    start_time = SUMO_NET_CONFIG[FOLDER_NAME]['start_time'] # route 开始的时间


    # 转换为文件路径
    cfg_xml = pathConvert(f'./SumoNets/{FOLDER_NAME}/env/{cfg_name}')
    net_xml = [pathConvert(f'./SumoNets/{FOLDER_NAME}/env/{net_name}')]
    route_xml = [pathConvert(f'./SumoNets/{FOLDER_NAME}/routes/{route_name}')]

    route_params = dict() # 给每一个 route 生成一组参数

    for _net in [net_name]:
        for _route in [route_name]:
            _net_name = _net.split('.')[0] # 得到 NET 的名称
            _route_name = _route.split('.')[0] # 得到车流
            print('_net_name',_net_name,' _route_name',_route_name)
            route_output_folder = os.path.join(output_folder, f'{_net_name}/{_route_name}') # 模型输出文件夹
            os.makedirs(route_output_folder, exist_ok=True) # 创建文件夹
            trip_info = os.path.join(route_output_folder, f'tripinfo.out.xml')
            statistic_output = os.path.join(route_output_folder, f'statistic.out.xml')
            summary = os.path.join(route_output_folder, f'summary.out.xml')
            queue_output = os.path.join(route_output_folder, f'queue.out.xml')
            tls_add = [
                # 探测器
                pathConvert(f'./SumoNets/{FOLDER_NAME}/detectors/e1_internal.add.xml'),
                pathConvert(f'./SumoNets/{FOLDER_NAME}/detectors/e2.add.xml'),
                # 信号灯
                pathConvert(f'./SumoNets/{FOLDER_NAME}/add/tls_programs.add.xml'),
                pathConvert(f'./SumoNets/{FOLDER_NAME}/add/tls_state.add.xml'),
                pathConvert(f'./SumoNets/{FOLDER_NAME}/add/tls_switch_states.add.xml'),
                pathConvert(f'./SumoNets/{FOLDER_NAME}/add/tls_switches.add.xml')
            ]
        
            env_dict = {
                FOLDER_NAME: {
                    'cfg': pathConvert(f'./SumoNets/{FOLDER_NAME}/env/{cfg_name}'),
                    'net':[pathConvert(f'./SumoNets/{FOLDER_NAME}/env/{_net}')],
                    'route':[pathConvert(f'./SumoNets/{FOLDER_NAME}/routes/{_route}')]
                }
            }

            params = {
                'tls_id':tls_id,
                'begin_time':start_time,
                'num_seconds':3600,
                'sumo_cfg':cfg_xml,
                'net_files':net_xml,
                'route_files':route_xml,
                'trip_info':trip_info,
                'statistic_output':statistic_output,
                'summary':summary,
                'queue_output':queue_output,
                'tls_state_add':tls_add,
                # 下面是数据增强的参数
                'num_stack':N_STACK,
                'num_delayed':N_DELAY,
                # 下面是仿真器的参数
                'is_libsumo':True,
                'use_gui':False,
                'min_green':5,
                'log_file':LOG_PATH,
                'env_dict':env_dict,
                'mode':'eval'
            }

            _key = f'{_net_name}__{_route_name}'
            route_params[_key] = params

    return route_params




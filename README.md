# TSC-DelayLight
## Getting Start

### model free

- `train.py`，可从模型库指定模型训练；
- - **Example**：python train.py --stack=6 --delay=0 --model_name=scnn --net_env=train_four_345 --net_name=4phases.net.xml

- - **model zoo**： scnn ernn eattention ecnn inference predict 

- - **env**：‘train_four_3  3phases.net.xml‘  ’train_four_345 4phases.net.xml‘ ’train_four_345 6phases.net.xml‘


- `test.py`，对训练好的模型进行测试，指定模型的类型，和需要测试的环境名称，测试环境需要有路口类型和相位选择 模型在读取时默认读取delay=0时训练的模型，可自行更改。
- 
- - **Example**：python test.py --stack=6 --delay=0 --model_name=ernn --net_name=train_four_3 

### predict model

- `FlowData_create.py`,车流数据生成，用于预测模型的训练
- `RNN_predict.py`, 用于预测模型的训练，此处使用是LSTM
- `train_predict.py`, 用于基于预测的RL模型训练，此处和model free相比，多传入一个预测模型权重参数

## Scripts

训练使用的脚本文件，可以在scripts中找到

## 使用问题

使用中可能会出现问题，欢迎留言，逐步完善。

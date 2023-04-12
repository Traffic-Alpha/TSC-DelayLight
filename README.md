# TSC-DelayLight
## Getting Start

- `train.py`，可从模型库指定模型训练；
- - **Example**：python train.py --stack=6 --delay=0 --model_name=scnn --net_env=train_four_345 --net_name=4phases.net.xml

- - **model zoo**： scnn ernn eattention ecnn inference predict 

- - **env**：‘train_four_3  3phases.net.xml‘  ’train_four_345 4phases.net.xml‘ ’train_four_345 6phases.net.xml‘


- `test.py`，对训练好的模型进行测试，指定模型的类型，和需要测试的环境名称，测试环境需要有路口类型和相位选择 模型在读取时默认读取delay=0时训练的模型，可自行更改。
- 
- - **Example**：python test.py --stack=6 --delay=0 --model_name=ernn --net_name=train_four_3 


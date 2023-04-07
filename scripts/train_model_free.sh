#!/usr/bin/zsh
###
 # @Author: Pang Aoyu
 # @Date: 2022-08-22 21:40:18
 # @Description: 测试 scnn esnn ernn eattention  在不同的数据增强方法的效果
 # @Command: nohup bash scripts/train_model_free.sh > model_free.log &
 # @LastEditTime: 2023-03-02 10:59:13
###
FOLDER="/home/aoyu/traffic/Traffic_Delay_Light" 
python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=scnn --net_env=train_four_345 --net_name=4phases.net.xml

echo '完成 SCNN.'

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=ecnn --net_env=train_four_345 --net_name=4phases.net.xml

echo '完成 ECNN.'

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=ernn --net_env=train_four_345 --net_name=4phases.net.xml

echo '完成 ERNN.'

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=eattention --net_env=train_four_345 --net_name=4phases.net.xml

echo '完成Eattention'

echo 'over train_four_345 4phases.net.xml'

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=scnn --net_env=train_four_345 --net_name=6phases.net.xml

echo '完成 SCNN.'

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=ecnn --net_env=train_four_345 --net_name=6phases.net.xml

echo '完成 ECNN.'

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=ernn --net_env=train_four_345 --net_name=6phases.net.xml
echo '完成 ERNN.'

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=eattention --net_env=train_four_345 --net_name=6phases.net.xml

echo '完成Eattention'

echo 'over train_four_345 6phases.net.xml'

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=scnn --net_env=train_three_3 --net_name=3phases.net.xml

echo '完成 SCNN.'

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=ecnn --net_env=train_three_3 --net_name=3phases.net.xml

echo '完成 ECNN.'

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=ernn --net_env=train_three_3 --net_name=3phases.net.xml
echo '完成 ERNN.'

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=eattention --net_env=train_three_3 --net_name=3phases.net.xml

echo 'train_three_3 3phases.net.xml'
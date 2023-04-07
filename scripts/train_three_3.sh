#!/usr/bin/zsh
###
 # @Author: Pang Aoyu
 # @Date: 2023-04-05 15:30:18
 # @Description: 测试 scnn esnn ernn eattention  在不同的数据增强方法的效果
 # @Command: nohup bash scripts/train_three_3.sh > model_free_1.log &
 # @LastEditTime: 2023-03-05 15:32:02
###
FOLDER="/home/aoyu/traffic/Traffic_Delay_Light" 

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=scnn --net_env=train_three_3 --net_name=3phases.net.xml

echo '完成 SCNN.'

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=ecnn --net_env=train_three_3 --net_name=3phases.net.xml

echo '完成 ECNN.'

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=ernn --net_env=train_three_3 --net_name=3phases.net.xml
echo '完成 ERNN.'

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=eattention --net_env=train_three_3 --net_name=3phases.net.xml

echo 'train_three_3 3phases.net.xml'

echo '完成'
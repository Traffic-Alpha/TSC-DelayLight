#!/usr/bin/zsh
###
 # @Author: WANG Maonan
 # @Date: 2022-08-22 21:40:18
 # @Description: 测试 scnn 在不同的数据增强方法的效果
 # @Command: nohup bash scripts/train_scnn.sh >  train_baseline.log &
 # @LastEditTime: 2023-03-02 10:59:13
###
FOLDER="/home/aoyu/TSC-DelayLight" 

python ${FOLDER}/train.py --stack=1 --delay=0 --model_name=scnn --net_env=train_three_3 --net_name=3phases.net.xml

echo '完成 train_three_3.'

python ${FOLDER}/train.py --stack=1 --delay=0 --model_name=scnn --net_env=train_four_345 --net_name=4phases.net.xml

echo '完成 train_four_345 4phases.net.xml.'

python ${FOLDER}/train.py --stack=1 --delay=0 --model_name=scnn --net_env=train_four_345 --net_name=6phases.net.xml
echo '完成 train_four_345 6phases.net.xml.'

echo '完成 baseline'  
#!/usr/bin/zsh
###
 # @Author: WANG Maonan
 # @Date: 2022-08-22 21:40:18
 # @Description: 测试 scnn 在不同的数据增强方法的效果
 # @Command: nohup bash scripts/train_scnn.sh > train_scnn.log &
 # @LastEditTime: 2023-03-02 10:59:13
###
FOLDER="/home/aoyu/traffic/Traffic_Delay_Light" 

echo ${FOLDER}

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=scnn
echo '完成 False, False, False, False.'

echo '完成 True, True, True, True.'

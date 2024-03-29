#!/usr/bin/zsh
###
 # @Author: WANG Maonan
 # @Date: 2022-08-22 21:40:18
 # @Description: 测试 X 在不同的数据增强方法的效果
 # @Command: nohup bash scripts/train_X.sh >  train_inference_ecnn.log &
 # @LastEditTime: 2023-03-02 10:59:13
###

FOLDER="/home/aoyu/traffic/TSC-DelayLight" 

model=inference_ecnn

stack_n=8

python ${FOLDER}/train.py --stack=$stack_n --delay=0 --model_name=$model  --net_env=train_three_3 --net_name=3phases.net.xml &

echo '完成 train_three_3.' 

python ${FOLDER}/train.py --stack=$stack_n --delay=0 --model_name=$model  --net_env=train_four_345 --net_name=4phases.net.xml &

echo '完成 train_four_345 4phases.net.xml.' 

python ${FOLDER}/train.py --stack=$stack_n --delay=0 --model_name=$model  --net_env=train_four_345 --net_name=6phases.net.xml &

echo '完成 train_four_345 6phases.net.xml.' 
wait
echo '完成' $model 
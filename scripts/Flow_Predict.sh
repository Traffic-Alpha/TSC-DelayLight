#!/usr/bin/zsh
###
 # @Author: Pang Aoyu
 # @Date: 2023-04-05 15:30:18
 # @Description: 测试 scnn esnn ernn eattention  在不同的数据增强方法的效果
 # @Command: nohup bash scripts/Flow_Predict.sh > Flow_Predict.log &
 # @LastEditTime: 2023-03-05 15:32:02
###S
FOLDER="/home/aoyu/TSC-DelayLight"  

python ${FOLDER}/RNN_predict.py --stack=8 --delay=0 --model_name=flowdata --net_env=train_three_3 --net_name=3phases.net.xml  >1.csv 

echo '完成 train_three_3.'

python ${FOLDER}/RNN_predict.py --stack=8 --delay=0 --model_name=flowdata --net_env=train_four_345 --net_name=4phases.net.xml >2.csv 

echo '完成 train_four_345 4phases.net.xml.'

python ${FOLDER}/RNN_predict.py --stack=8 --delay=0 --model_name=flowdata --net_env=train_four_345 --net_name=6phases.net.xml >3.csv 
echo '完成 train_four_345 6phases.net.xml.'

echo '完成 FlowData'  
#!/usr/bin/zsh
###
 # @Author: Pang Aoyu
 # @Date: 2023-04-05 15:30:18
 # @Description: 测试 scnn esnn ernn eattention  infrence  predict 读取数据
 # @Command: nohup bash scripts/train_all.sh >  train_all.log &
 # @LastEditTime: 2023-03-05 15:32:02
 #scnn ernn eattention ecnn inference ernn_P ernn_C
###

FOLDER="/home/aoyu/TSC-DelayLight" 

delay_time=0

for model_name in scnn ernn eattention ecnn inference ernn_P ernn_C inference_scnn
do    
        python ${FOLDER}/train.py --stack=8 --delay=$delay_time --model_name=$model_name --net_env=train_three_3  --net_name=3phases.net.xml &
    
        echo 'Finish' $model_name  $delay_time  'train_three_3 3phases.net.xml' 

        python ${FOLDER}/train.py --stack=8 --delay=$delay_time --model_name=$model_name --net_env=train_four_345  --net_name=4phases.net.xml &
    
        echo 'Finish' $model_name  $delay_time  'train_four_345 4phases.net.xml' 

        python ${FOLDER}/train.py --stack=8 --delay=$delay_time --model_name=$model_name --net_env=train_four_345  --net_name=6phases.net.xml &
        
        echo 'Finish' $model_name  $delay_time  'train_four_345 6phases.net.xml'

        wait 
done

python ${FOLDER}/train_predict.py --stack=8 --delay=0 --model_name=predict --net_env=train_three_3 --net_name=3phases.net.xml &

echo '完成 train_three_3.'

python ${FOLDER}/train_predict.py --stack=8 --delay=0 --model_name=predict --net_env=train_four_345 --net_name=4phases.net.xml & 

echo '完成 train_four_345 4phases.net.xml.'

python ${FOLDER}/train_predict.py --stack=8 --delay=0 --model_name=predict --net_env=train_four_345 --net_name=6phases.net.xml &
echo '完成 train_four_345 6phases.net.xml.'

wait

echo '完成 predict'  

echo ' ALL Finish'
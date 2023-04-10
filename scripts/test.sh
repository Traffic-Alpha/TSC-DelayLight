#!/usr/bin/zsh
###
 # @Author: Pang Aoyu
 # @Date: 2023-04-05 15:30:18
 # @Description: 测试 scnn esnn ernn eattention  infrence 在不同时延情况下的效果
 # @Command: nohup bash scripts/test.sh >  test.log &
 # @LastEditTime: 2023-03-05 15:32:02
###

FOLDER="/home/aoyu/traffic/Traffic_Delay_Light" 

for delay_time in 0 5 10 15 20 30 40 50 60 70 80 90 100 110 120
do
    for model_name in scnn ernn eattention ecnn inference
    do    
        python ${FOLDER}/test.py --stack=6 --delay=$delay_time --model_name=$model_name --net_env=train_three_3  --net_name=3phases.net.xml
    
        echo 'Finish' $model_name  $delay_time  'train_three_3 3phases.net.xml' 

        python ${FOLDER}/test.py --stack=6 --delay=$delay_time --model_name=$model_name --net_env=train_four_345  --net_name=4phases.net.xml
    
        echo 'Finish' $model_name  $delay_time  'train_four_345 4phases.net.xml' 

        python ${FOLDER}/test.py --stack=6 --delay=$delay_time --model_name=$model_name --net_env=train_four_345  --net_name=6phases.net.xml
        
        echo 'Finish' $model_name  $delay_time  'train_four_345 6phases.net.xml' 

    done
done

echo ' ALL Finish'
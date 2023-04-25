#!/usr/bin/zsh
###
 # @Author: Pang Aoyu
 # @Date: 2023-04-05 15:30:18
 # @Description: 测试 scnn esnn ernn eattention  infrence 在不同时延情况下的效果
 # @Command: nohup bash scripts/Mtest.sh >  Mtest.log &
 # @LastEditTime: 2023-03-05 15:32:02
###
FOLDER="/home/aoyu/TSC-DelayLight" 
STACK=6
for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 
do
    for delay_time in 0 1 2 3 4 6 8 10 12 14 16 18 20 22 24
    do
    for model_name in scnn ernn eattention ecnn inference predict ernn_P ernn_C  inference_scnn
    do    
        python ${FOLDER}/test.py --stack=$STACK --delay=$delay_time --model_name=$model_name --net_env=train_three_3  --net_name=3phases.net.xml &
    
        echo 'Finish' $model_name  $delay_time  'train_three_3 3phases.net.xml' 

        python ${FOLDER}/test.py --stack=$STACK  --delay=$delay_time --model_name=$model_name --net_env=train_four_345  --net_name=4phases.net.xml &
    
        echo 'Finish' $model_name  $delay_time  'train_four_345 4phases.net.xml' 

        python ${FOLDER}/test.py --stack=$STACK  --delay=$delay_time --model_name=$model_name --net_env=train_four_345  --net_name=6phases.net.xml &
        
        echo 'Finish' $model_name  $delay_time  'train_four_345 6phases.net.xml' 
        wait
    done
    done

    echo 'Finish' $i 'test' 

    for delay_time in 0 1 2 3 4 6 8 10 12 14 16 18 20 22 24
    do
    for model_name in scnn ernn eattention ecnn inference predict ernn_P ernn_C  inference_scnn
    do    
        python ${FOLDER}/TestData_Read.py --stack=$STACK  --delay=$delay_time --model_name=$model_name --net_env=train_three_3  --net_name=3phases.net.xml
    
        echo 'Finish' $model_name  $delay_time  'train_three_3 3phases.net.xml' 

        python ${FOLDER}/TestData_Read.py --stack=$STACK  --delay=$delay_time --model_name=$model_name --net_env=train_four_345  --net_name=4phases.net.xml
    
        echo 'Finish' $model_name  $delay_time  'train_four_345 4phases.net.xml' 

        python ${FOLDER}/TestData_Read.py --stack=$STACK  --delay=$delay_time --model_name=$model_name --net_env=train_four_345  --net_name=6phases.net.xml
        
        echo 'Finish' $model_name  $delay_time  'train_four_345 6phases.net.xml' 

    done
    done
    echo 'Finish' $i 'Read' 
done
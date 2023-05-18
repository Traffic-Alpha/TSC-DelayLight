#!/usr/bin/zsh
###
 # @Author: Pang Aoyu
 # @Date: 2023-04-05 15:30:18
 # @Description: 测试 scnn esnn ernn eattention  infrence  predict 读取数据 全部数据读取
 # @Command: nohup bash scripts/Read_csv.sh >  Read_csv.log &
 # @LastEditTime: 2023-03-05 15:32:02
###

FOLDER="/home/aoyu/TSC-DelayLight" 
 
for model_name in scnn ernn eattention ecnn inference predict ernn_P ernn_C inference_scnn
    do    
        python ${FOLDER}/Read_csv.py --stack=6  --model_name=$model_name --net_env=train_three_3  --net_name=3phases.net.xml &
    
        echo 'Finish' $model_name  $delay_time  'train_three_3 3phases.net.xml' 

        python ${FOLDER}/Read_csv.py --stack=6 --model_name=$model_name --net_env=train_four_345  --net_name=4phases.net.xml &
    
        echo 'Finish' $model_name  $delay_time  'train_four_345 4phases.net.xml' 

        python ${FOLDER}/Read_csv.py --stack=6  --model_name=$model_name --net_env=train_four_345  --net_name=6phases.net.xml &
        
        echo 'Finish' $model_name  $delay_time  'train_four_345 6phases.net.xml' 
       
        wait
    done


echo ' ALL Finish'
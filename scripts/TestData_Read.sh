#!/usr/bin/zsh
###
 # @Author: Pang Aoyu
 # @Date: 2023-04-05 15:30:18
 # @Description: 测试 scnn esnn ernn eattention  infrence  predict 读取数据
 # @Command: nohup bash scripts/TestData_Read.sh >  testdata_read.log &
 # @LastEditTime: 2023-03-05 15:32:02
###

FOLDER="/home/aoyu/TSC-DelayLight" 

for delay_time in 0 1 2 3 4 6 8 10 12 14 16 18 20 22 24
do
    for model_name in ernn_C
    do    
        python ${FOLDER}/TestData_Read.py --stack=6 --delay=$delay_time --model_name=$model_name --net_env=train_three_3  --net_name=3phases.net.xml &
    
        echo 'Finish' $model_name  $delay_time  'train_three_3 3phases.net.xml' 

        python ${FOLDER}/TestData_Read.py --stack=6 --delay=$delay_time --model_name=$model_name --net_env=train_four_345  --net_name=4phases.net.xml &
    
        echo 'Finish' $model_name  $delay_time  'train_four_345 4phases.net.xml' 

        python ${FOLDER}/TestData_Read.py --stack=6 --delay=$delay_time --model_name=$model_name --net_env=train_four_345  --net_name=6phases.net.xml &
        
        echo 'Finish' $model_name  $delay_time  'train_four_345 6phases.net.xml' 
        wait
    done
done

echo ' ALL Finish'
#!/usr/bin/zsh
###
 # @Author: Pang Aoyu
 # @Date: 2022-08-22 21:40:18
 # @Description: 测试 scnn esnn ernn eattention  在不同的数据增强方法的效果
 # @Command: nohup bash scripts/train_all.sh > train_all.log &
 # @LastEditTime: 2023-03-02 10:59:13
 #当predict训练所需要的车流和预测模型训练好后，才可执行此脚本
###
FOLDER="/home/aoyu/TSC-DelayLight" 
python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=scnn --net_env=train_four_345 --net_name=4phases.net.xml

echo '完成 SCNN.'

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=ecnn --net_env=train_four_345 --net_name=4phases.net.xml

echo '完成 ECNN.'

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=ernn --net_env=train_four_345 --net_name=4phases.net.xml

echo '完成 ERNN.'

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=eattention --net_env=train_four_345 --net_name=4phases.net.xml

echo '完成Eattention'

echo 'over train_four_345 4phases.net.xml'

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=scnn --net_env=train_four_345 --net_name=6phases.net.xml

echo '完成 SCNN.'

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=ecnn --net_env=train_four_345 --net_name=6phases.net.xml

echo '完成 ECNN.'

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=ernn --net_env=train_four_345 --net_name=6phases.net.xml
echo '完成 ERNN.'

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=eattention --net_env=train_four_345 --net_name=6phases.net.xml

echo '完成Eattention'

echo 'over train_four_345 6phases.net.xml'

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=scnn --net_env=train_three_3 --net_name=3phases.net.xml

echo '完成 SCNN.'

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=ecnn --net_env=train_three_3 --net_name=3phases.net.xml

echo '完成 ECNN.'

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=ernn --net_env=train_three_3 --net_name=3phases.net.xml
echo '完成 ERNN.'

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=eattention --net_env=train_three_3 --net_name=3phases.net.xml

echo 'train_three_3 3phases.net.xml'

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=inference --net_env=train_three_3 --net_name=3phases.net.xml

echo '完成 train_three_3.'

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=inference --net_env=train_four_345 --net_name=4phases.net.xml

echo '完成 train_four_345 4phases.net.xml.'

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=inference --net_env=train_four_345 --net_name=6phases.net.xml
echo '完成 train_four_345 6phases.net.xml.'

echo '完成 inference'  

python ${FOLDER}/train_predict.py --stack=6 --delay=0 --model_name=predict --net_env=train_three_3 --net_name=3phases.net.xml

echo '完成 train_three_3.'

python ${FOLDER}/train_predict.py --stack=6 --delay=0 --model_name=predict --net_env=train_four_345 --net_name=4phases.net.xml

echo '完成 train_four_345 4phases.net.xml.'

python ${FOLDER}/train_predict.py --stack=6 --delay=0 --model_name=predict --net_env=train_four_345 --net_name=6phases.net.xml
echo '完成 train_four_345 6phases.net.xml.'

echo '完成 predict'  
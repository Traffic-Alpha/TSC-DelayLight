import numpy as np 
import pandas as pd 
import csv

import os
import argparse
import torch 
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from aiolos.utils.get_abs_path import getAbsPath
from aiolos.trafficLog.initLog import init_logging
    
pathConvert = getAbsPath(__file__)
  
  #定义模型 输入维度input_size ，隐藏层维度hidden_size可任意指定，这里为4
class lstm_reg(nn.Module):
    def __init__(self,input_size=64,hidden_size=10, output_size=1,num_layers=2):
        super(lstm_reg,self).__init__()
            #super() 函数是用于调用父类(超类)的一个方法，直接用类名调用父类
        self.rnn = nn.LSTM(input_size,hidden_size,num_layers) #LSTM 网络
        self.reg = nn.Linear(hidden_size,output_size) #Linear 函数继承于nn.Module
   
    def forward(self,x):   #定义model类的forward函数
        x, _ = self.rnn(x)
        s,b,h = x.shape   #矩阵从外到里的维数
                    #view()函数的功能和reshape类似，用来转换size大小
        x = x.view(s*b, h) #输出变为（s*b）*h的二维
        x = self.reg(x)
        x = x.view(s,b,-1) #卷积的输出从外到里的维数为s,b,一列
        return x
    
def predict_model(
        net_name,net_env,n_stack, n_delay, model_name,
    ):
    
    #LSTM（Long Short-Term Memory）是长短期记忆网络
    # args
    N_STACK = n_stack # 堆叠
    N_DELAY = n_delay # 时延

    FLOWDATA_PATH=pathConvert(f'./FlowData/{net_env}_{net_name}_{N_STACK}_{N_DELAY}.csv')#存储Flow数据
    NET_PATH=pathConvert(f'./FlowData/{net_env}_{net_name}_{N_STACK}_{N_DELAY}.pkl')

    data_csv = pd.read_csv(FLOWDATA_PATH,header=None)
    Long=data_csv.shape[0]
    data=np.array(data_csv)
    data=data[0:Long]
    shape=data.shape
    Line=int(shape[1]/8) # /8 是因为一行有八个特征
    #数据预处理

    dataset = data.astype('float32')   #astype(type):实现变量类型转换  


    ALL_Data  = create_dataset(dataset,N_STACK+1)
    #data_X: 2*142     data_Y: 1*142

    #划分训练集和测试集，70%作为训练集
    train_size = int(len(ALL_Data) * 0.7)
    test_size = len(ALL_Data)-train_size
    
    Train_Data = ALL_Data[:train_size]
    
    Test_Data = ALL_Data[train_size:]
    


    Train_Data = Train_Data.reshape(-1,shape[1],N_STACK+1) #reshape中，-1使元素变为一行，然后输出为1列，每列2个子元素
    Test_Data = Test_Data.reshape(-1,shape[1],N_STACK+1)

    
    Train_Data = torch.from_numpy(Train_Data) #torch.from_numpy(): numpy中的ndarray转化成pytorch中的tensor(张量)
    Test_Data = torch.from_numpy(Test_Data)

    net = lstm_reg(N_STACK,10) #input_size=2，hidden_size=4
    
    criterion = nn.MSELoss()  #损失函数均方差
    optimizer = torch.optim.Adam(net.parameters(),lr=1e-2)
    #构造一个优化器对象 Optimizer,用来保存当前的状态，并能够根据计算得到的梯度来更新参数
    #Adam 算法:params (iterable)：可用于迭代优化的参数或者定义参数组的 dicts   lr:学习率
    
    train_loader = DataLoader(dataset=Train_Data,
                    batch_size=50,
                    shuffle=False)
    test_loader = DataLoader(dataset=Train_Data,
                    batch_size=50,
                    shuffle=False)
    #构建测试集
    print('train_loader',train_loader)
    print('test_loader',test_loader)
    min_Loss=float('inf')

    for epoch in range(200):

        
        for batch, x in enumerate(train_loader):
        

            var_x = Variable(x[:,:,0:N_STACK]) #转为Variable（变量）
            var_y = Variable(x[:,:,-1])
            var_y=np.expand_dims(var_y, axis=2)
            var_y=torch.tensor(var_y)
            out = net(var_x)
            loss = criterion(out, var_y)
            optimizer.zero_grad() #把梯度置零，也就是把loss关于weight的导数变成0.
            loss.backward()  #计算得到loss后就要回传损失，这是在训练的时候才会有的操作，测试时候只有forward过程
            optimizer.step() #回传损失过程中会计算梯度，然后optimizer.step()根据这些梯度更新参数

        Loss=0

        for batch, _data in enumerate(test_loader):
            val_x=Variable(_data[:,:,0:N_STACK]) 
            val_y = Variable(_data[:,:,-1])
            val_y=np.expand_dims(val_y, axis=2)
            out = net(val_x)
            val_y=torch.tensor(val_y)
            loss=criterion(out, val_y)
            Loss+=loss
        
        print('Epoch: {}, Loss:{:.5f}'.format(epoch+1, Loss.item()))

        if(Loss<min_Loss):
            min_Loss=Loss
            torch.save(net.state_dict(), NET_PATH)   #保存预测的最好的模型

     
    #state_dict 是一个简单的python的字典对象,将每一层与它的对应参数建立映射关系
 
 

def create_dataset(dataset,look_back=6):#look_back 以前的时间步数用作输入变量来预测下一个时间段

    dataX =[] 
 
    for i in range(dataset.shape[0]- look_back):
        #print('look_back',look_back)
        a = dataset[i:(i+look_back),:]  #i和i+1赋值
        dataX.append(a)  #i+2赋值

    return np.array(dataX)  #np.array构建数组
 

        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--stack', type=int, default=8)
    parser.add_argument('--delay', type=int, default=0)
    parser.add_argument('--net_env', type=str, default='train_four_345')
    parser.add_argument('--net_name', type=str, default='4phases.net.xml')
    parser.add_argument('--model_name', type=str, default='flowdata')
    args = parser.parse_args()

    predict_model(
        net_env=args.net_env, net_name=args.net_name , n_stack=args.stack, n_delay=args.delay,
        model_name=args.model_name,
    )
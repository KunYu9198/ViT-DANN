# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 16:59:29 2022

@author: PC
"""

import argparse
import scipy.io as sio   # 导入.mat格式数据用
from torch.utils.data import TensorDataset, DataLoader   #  随机划分数据用
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.functional import softmax
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np
import math
from models import KKViTModel

# ----------------------利用GPU计算-----------------------------------#
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#--------------------------------------------------------------------#

# ---------------------------定义模型参数 ------------------------------ #
parser = argparse.ArgumentParser(description='PyTorch Partial GAN')

parser.add_argument('--num_epochs', default=400, type=int, metavar='N', help='number of total epochs to run')

parser.add_argument('--batch_size', default=20, type=int, metavar='N', help='train batchsize') # 带标签训练样本中每一个小块包含的样本个数

parser.add_argument('--class_num', default=4, type=int, metavar='N', help='class number')

parser.add_argument('--num_patches', default=100, type=int, metavar='N', help='num_patches')

parser.add_argument('--embed_dim', default=100, type=int, metavar='N', help='embed_dim')

parser.add_argument('--num_heads', default=10, type=int, metavar='N', help='num_heads')

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}    # 将上述定义的参数可视化   

# --------------------------- 训练和测试数据设置 ------------------------  #
# ---- 训练数据 ---- #
# ---- 导入源域训练数据 ---- #
data = sio.loadmat('F:\\MATLAB_code\\Code\\A1-转速900四种故障-添加0db噪声.mat') 
train_data_src = data['Vib_Train_data_1']
train_label_src = data['Vib_Train_label']

num_train_instances_src = len(train_label_src)    # 训练集样本行数
train_data_src = train_data_src.reshape(num_train_instances_src,args.num_patches,args.embed_dim)   # 每一行向量转化为num_patches*embed_dim的矩阵
# 处理振动信号部分
train_data_src = torch.from_numpy(train_data_src).type(torch.FloatTensor)

# 处理标签部分
train_label_src = torch.from_numpy(train_label_src).type(torch.LongTensor)
train_label_src = train_label_src.view(num_train_instances_src)      # 把张量维度改成1维

#  全部转化成张量形式  #
train_dataset_src = TensorDataset(train_data_src, train_label_src)
train_data_loader_src = DataLoader(dataset=train_dataset_src, batch_size=args.batch_size, shuffle=True)

# ---- 导入目标域数据 ---- # 
data = sio.loadmat('F:\\MATLAB_code\\Code\\A1-转速1500四种故障-添加0db噪声.mat')
train_data_tgt = data['Vib_Train_data_1']
train_label_tgt = data['Vib_Train_label']

num_train_instances_tgt = len(train_label_tgt)    # 训练集样本行数
train_data_tgt = train_data_tgt.reshape(num_train_instances_tgt,args.num_patches,args.embed_dim)   # 每一行向量转化为num_patches*embed_dim的矩阵
# 处理振动信号部分
train_data_tgt = torch.from_numpy(train_data_tgt).type(torch.FloatTensor)

# 处理标签部分
train_label_tgt = torch.from_numpy(train_label_tgt).type(torch.LongTensor)
train_label_tgt = train_label_tgt.view(num_train_instances_tgt)

#  全部转化成张量形式  #
train_dataset_tgt = TensorDataset(train_data_tgt, train_label_tgt)
train_data_loader_tgt = DataLoader(dataset=train_dataset_tgt, batch_size=args.batch_size, shuffle=True)

# ---- 测试数据 ---- #
data = sio.loadmat('F:\\MATLAB_code\\Code\\A1-转速1500四种故障-添加0db噪声.mat')
test_data = data['Vib_Test_data_1']
test_label = data['Vib_Test_label']

num_test_instances = len(test_data)      # 测试样本行数
test_data = test_data.reshape(num_test_instances,args.num_patches,args.embed_dim)   # 每一行向量转化为num_patches*embed_dim的矩阵
# 处理振动信号部分
test_data = torch.from_numpy(test_data).type(torch.FloatTensor)

# 处理标签部分
test_label = torch.from_numpy(test_label).type(torch.LongTensor)
test_label = test_label.view(num_test_instances)
# 全部转化成张量形式
test_dataset = TensorDataset(test_data, test_label)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

# 引用网络
Net = KKViTModel.DAN
# GPU运行
Model = Net(num_patches=args.num_patches, embed_dim=args.embed_dim, num_heads=args.num_heads, num_classes=args.class_num)
Model = Model.to(DEVICE)
# 损失函数
class_criterion = nn.CrossEntropyLoss()
domian_criterion = nn.BCELoss()
# 优化器
optimizer = torch.optim.Adam(Model.parameters(), lr=0.0001, betas=(0.5, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5, last_epoch=-1) 


for epoch in range(args.num_epochs):
    # 把模型打开用于训练
    Model.train()
    
    # ============================ #
    train_src_iter = iter(train_data_loader_src)
    train_tgt_iter = iter(train_data_loader_tgt)
    total_iter = int(num_train_instances_src/args.batch_size)
    # ---- 参数设置 ---- #
    p = float(epoch / args.num_epochs)
    alpha = 2. / (1. + np.exp(-10 * p)) - 1
    # ============================ #
    for batch_idx in range(total_iter):
        # ---- 导入源域数据 ---- #
        try:
            vib_samples_src, labels_src = train_src_iter.next()
            vib_samples_src = Variable(vib_samples_src.to(DEVICE))
            labels_src = Variable(labels_src.to(DEVICE))
        except:
            train_src_iter = iter(train_data_loader_src)
            vib_samples_src, labels_src = train_src_iter.next()
            vib_samples_src = Variable(vib_samples_src.to(DEVICE))
            labels_src = Variable(labels_src.to(DEVICE))
        
        # ---- 导入目标域数据 ---- #    
        try:
            vib_samples_tgt, labels_tgt = train_tgt_iter.next()
            vib_samples_tgt = Variable(vib_samples_tgt.to(DEVICE))
        except:
            train_tgt_iter = iter(train_data_loader_tgt)
            vib_samples_tgt, labels_tgt = train_tgt_iter.next()
            vib_samples_tgt = Variable(vib_samples_tgt.to(DEVICE))
        
        # prepare domain label
        size_src = len(vib_samples_src)
        size_tgt = len(vib_samples_tgt)
        domain_label_src = torch.zeros(size_src).float().to(DEVICE)  # source 0
        domain_label_tgt = torch.ones(size_tgt).float().to(DEVICE)  # target 1
        
        # train on source domain
        src_class_output, src_domain_output = Model(vib_samples_src, alpha=alpha)
        src_loss_class = class_criterion(src_class_output, labels_src)
        src_loss_domain = domian_criterion(src_domain_output.view(-1), domain_label_src)
        
        # train on target domain
        _, tgt_domain_output = Model(vib_samples_tgt, alpha=alpha)
        tgt_loss_domain = domian_criterion(tgt_domain_output.view(-1), domain_label_tgt) 
        
        # 更新损失函数
        loss = src_loss_class + 0.5*(src_loss_domain + tgt_loss_domain)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        cross_loss.append(src_loss_class.item())
        source_loss.append(src_loss_domain.item())
        target_loss.append(tgt_loss_domain.item())
    
    scheduler.step()
    print(f'Epoch: {epoch} 的alpha: {alpha:.4f}')
    print('epoch {}, 总的loss = {:g}'.format((epoch+1), loss.item()))
    print('epoch {}, 源域分类loss = {:g}'.format((epoch+1), src_loss_class.item()))
    print('epoch {}, 源域域判断loss = {:g}'.format((epoch+1), src_loss_domain.item()))
    print('epoch {}, 目标域域判断loss = {:g}'.format((epoch+1), tgt_loss_domain.item()))


from __future__ import division
from __future__ import print_function

# 路径初始化
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append('E:\\Anaconda\\lib\\site-packages\\')
# print(sys.path)
print('Path initialization finished！\n')

# 可视化增加路径
from time import time
from sklearn import manifold, datasets

# visdom显示模块
from visdom import Visdom

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN

def show_Hyperparameter(args):
    argsDict = args.__dict__
    print('the settings are as following:')
    for key in argsDict:
        print(key,':',argsDict[key])

def train(epoch):
    t = time.time()
    '''将模型转为训练模式，并将优化器梯度置零'''
    model.train()
    optimizer.zero_grad()
    '''计算输出时，对所有的节点计算输出'''

    # ----------------------------------debug----------------------------------
    # print(features.size(),adj.size())
    # torch.Size([2708, 1433]) torch.Size([2708, 2708])
    # 也就是说，每次对整个图进行计算，但优化的时候，仅仅针对训练集样本产生的损失。
    # ----------------------------------debug----------------------------------
    output = model(features, adj)

    '''损失函数，仅对训练集节点计算，即：优化仅对训练集数据进行'''
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    # 计算准确率
    acc_train = accuracy(output[idx_train], labels[idx_train])
    # 反向传播
    loss_train.backward()
    # 优化
    optimizer.step()

    '''fastmode ? '''
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    '''验证集 loss 和 accuracy '''
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    '''输出训练集+验证集的 loss 和 accuracy '''
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

def test():
    model.eval()    # model转为测试模式
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return output   # 可视化返回output

# t-SNE 降维
def t_SNE(output, dimention):
    # output:待降维的数据
    # dimention：降低到的维度
    tsne = manifold.TSNE(n_components=dimention, init='pca', random_state=0)
    result = tsne.fit_transform(output)
    return result

# Visualization with visdom
def Visualization(vis, result, labels,title):
    # vis: Visdom对象
    # result: 待显示的数据，这里为t_SNE()函数的输出
    # label: 待显示数据的标签
    # title: 标题
    vis.scatter(
        X = result,
        Y = labels+1,           # 将label的最小值从0变为1，显示时label不可为0
       opts=dict(markersize=4,title=title),
    )

'''代码主函数开始'''
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--no_visual', action='store_true', default=False,
                    help='visualization of ground truth and test result')

args = parser.parse_args()


# 显示args
show_Hyperparameter(args)

# 是否使用CUDA
args.cuda = not args.no_cuda and torch.cuda.is_available()

# 设置随机数种子
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()           # 返回可视化要用的labels

# Model
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,     # 对Cora数据集，为7，即类别总数。
            dropout=args.dropout)
# optimizer
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

# to CUDA
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
output=test()           # 返回output

if not args.no_visual:
    # 计算预测值
    preds = output.max(1)[1].type_as(labels)

    # output的格式转换
    output = output.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    preds = preds.cpu().detach().numpy()

    # Visualization with visdom
    vis = Visdom(env='pyGCN Visualization')

    # ground truth 可视化
    result_all_2d = t_SNE(output, 2)
    Visualization(vis, result_all_2d, labels,
                  title='[ground truth of all samples]\n Dimension reduction to %dD' % (result_all_2d.shape[1]))
    result_all_3d = t_SNE(output, 3)
    Visualization(vis, result_all_3d, labels,
                  title='[ground truth of all samples]\n Dimension reduction to %dD' % (result_all_3d.shape[1]))

    # 预测结果可视化
    result_test_2d = t_SNE(output[idx_test.cpu().detach().numpy()], 2)
    Visualization(vis, result_test_2d, preds[idx_test.cpu().detach().numpy()],
                  title='[prediction of test set]\n Dimension reduction to %dD' % (result_test_2d.shape[1]))
    result_test_3d = t_SNE(output[idx_test.cpu().detach().numpy()], 3)
    Visualization(vis, result_test_3d, preds[idx_test.cpu().detach().numpy()],
                  title='[prediction of test set]\n Dimension reduction to %dD' % (result_test_3d.shape[1]))





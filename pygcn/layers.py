import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    '''定义对象的属性'''
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))           # in_features × out_features
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    '''生成权重'''
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)          # .uniform()：将tensor用从均匀分布中抽样得到的值填充。
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    '''前向传播 of 一层之内：即本层的计算方法：A * X * W '''
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)          # torch.mm：Matrix multiply，input和weight实现矩阵点乘。
        output = torch.spmm(adj, support)               # torch.spmm：稀疏矩阵乘法，sp即sparse。
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    '''把一个对象用字符串的形式表达出来以便辨认，在终端调用的时候会显示信息'''
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

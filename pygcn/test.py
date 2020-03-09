import numpy as np
import scipy.sparse as sp
import torch

def encode_onehot(labels):
    classes = set(labels)       # set() 函数创建一个无序不重复元素集
    print('number of classes:',classes)
    # number of classes: {'Genetic_Algorithms', 'Reinforcement_Learning', 'Case_Based',
    # 'Probabilistic_Methods', 'Theory', 'Neural_Networks', 'Rule_Learning'}
    # 一共7个类别。

    # enumerate()函数生成序列，带有索引i和值c。
    # 这一句将string类型的label变为int类型的label，建立映射关系
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    # map() 会根据提供的函数对指定序列做映射。
    # 这一句将string类型的label替换为int类型的label

    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    # 返回int类型的label
    return labels_onehot

# # demo of torch.mm()：矩阵点乘
# markov_states = torch.Tensor([[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]])
# transition_matrix = torch.Tensor([[1,1,1],[2,2,2],[3,3,3]])
#
# # Apply one transition
# new_state = torch.mm(markov_states, transition_matrix)
# print(markov_states)
# print(transition_matrix)
# print(new_state)
# print(new_state.shape)      # torch.Size([5, 3])
# ----------------------------------------------------------------
# a = torch.Tensor([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
# print(a[:,1:-1])
# print(a[:,-1])
# ----------------------------------------------------------------
path="C:/Users/73416/PycharmProjects/PyGCN_Visualization/data/cora/"
dataset="cora"
edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
# print(edges_unordered)
# # [[     35    1033]
# #  [     35  103482]
# #  [     35  103515]
# #  ...
# #  [ 853118 1140289]
# #  [ 853155  853118]
# #  [ 954315 1155073]]

# print(type(edges_unordered))
# # <class 'numpy.ndarray'>

# print(edges_unordered.shape)
# # (5429, 2)

idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
labels = encode_onehot(idx_features_labels[:, -1])
print('labels:\n',labels)
print('labels.shape:\n',labels.shape)

idx_map = {j: i for i, j in enumerate(idx)}
# print(idx_map)
# # {31336: 0, 1061127: 1, ...24043: 2707}，2707+1=2708 --> 样本数。
'''论文编号没有用，需要重新的其进行编号（从0开始），然后对原编号进行替换。
所以目的是把离散的原始的编号，变成0-2707的连续编号'''
edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
# print(edges)
# print(edges.shape)
# # (5429, 2)
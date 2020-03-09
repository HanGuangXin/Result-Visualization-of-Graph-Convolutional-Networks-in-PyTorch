import numpy as np
import scipy.sparse as sp
from pygcn.utils import normalize

'''测试归一化函数'''
# 读取原始数据集
path="C:/Users/73416/PycharmProjects/PyGCN_Visualization/data/cora/"
dataset = "cora"
idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))

RawFeature = idx_features_labels[:, 1:-1]
RawFeature=RawFeature.astype(int)
sample1_label=RawFeature[0,:]
sumA=sample1_label.sum()
print("原始的feature\n",RawFeature)
# type ndarray
# [['0' '0' '0'... '0' '0' '0']
#  ['0' '0' '0'... '0' '0' '0']
#  ['0' '0' '0'...'0' '0' '0']
#  ...
#  ['0' '0' '0'...'0' '0' '0']
# ['0' '0' '0'... '0' '0' '0']
# ['0' '0' '0'...'0' '0' '0']]
print(RawFeature.shape)
# (2708, 1433)

features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
# <2708x1433 sparse matrix of type '<class 'numpy.float32'>'
# 	with 49216 stored elements in Compressed Sparse Row format>
print("csr_matrix之后的feature\n",features)
# type csr_matrix
# (0, 0)          0.0
# (0, 1)          0.0
# (0, 2)          0.0
# (0, 3)          0.0
# (0, 4)          0.0
# ::
# (2707, 1428)    0.0
# (2707, 1429)    0.0
# (2707, 1430)    0.0
# (2707, 1431)    0.0
# (2707, 1432)    0.0
print(features.shape)
# (2708, 1433)

# features = normalize(features)
rowsum = np.array(features.sum(1))      # (2708, 1)
r_inv = np.power(rowsum, -1).flatten()  # (2708,)
r_inv[np.isinf(r_inv)] = 0.             # 处理除数为0导致的inf
r_mat_inv = sp.diags(r_inv)
# <2708x2708 sparse matrix of type '<class 'numpy.float32'>'
# 	with 2708 stored elements (1 diagonals) in DIAgonal format>
mx = r_mat_inv.dot(features)
print('normalization之后的feature\n',mx)
# (0, 176)    0.05
# (0, 125)    0.05
# (0, 118)    0.05
# (1, 1425)   0.05882353
# (1, 1389)   0.05882353
# (1, 1263)   0.05882353
# ::
# (2707, 136)   0.05263158
# (2707, 67)    0.05263158
# (2707, 19)    0.05263158



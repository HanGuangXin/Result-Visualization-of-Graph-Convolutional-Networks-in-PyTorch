from pygcn.utils import encode_onehot
import numpy as np

'''测试onehot编码'''
'''labels的onehot编码，前后结果对比'''
# 读取原始数据集
path="C:/Users/73416/PycharmProjects/PyGCN_Visualization/data/cora/"
dataset = "cora"
idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))

RawLabels=idx_features_labels[:, -1]
print("原始论文类别（label）：\n",RawLabels)
# ['Neural_Networks' 'Rule_Learning' 'Reinforcement_Learning' ...
# 'Genetic_Algorithms' 'Case_Based' 'Neural_Networks']
print(len(RawLabels))       # 2708

classes = set(RawLabels)       # set() 函数创建一个无序不重复元素集
print("原始标签的无序不重复元素集\n", classes)
# {'Genetic_Algorithms', 'Probabilistic_Methods', 'Reinforcement_Learning', 'Neural_Networks', 'Theory', 'Case_Based', 'Rule_Learning'}


# enumerate()函数生成序列，带有索引i和值c。
# 这一句将string类型的label变为onehot编码的label，建立映射关系
classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
print("原始标签与onehot编码结果的映射字典\n",classes_dict)
#  {'Genetic_Algorithms': array([1., 0., 0., 0., 0., 0., 0.]), 'Probabilistic_Methods': array([0., 1., 0., 0., 0., 0., 0.]),
#   'Reinforcement_Learning': array([0., 0., 1., 0., 0., 0., 0.]), 'Neural_Networks': array([0., 0., 0., 1., 0., 0., 0.]),
#   'Theory': array([0., 0., 0., 0., 1., 0., 0.]), 'Case_Based': array([0., 0., 0., 0., 0., 1., 0.]),
#   'Rule_Learning': array([0., 0., 0., 0., 0., 0., 1.])}

# map() 会根据提供的函数对指定序列做映射。
# 这一句将string类型的label替换为onehot编码的label
labels_onehot = np.array(list(map(classes_dict.get, RawLabels)),
                             dtype=np.int32)
print("onehot编码的论文类别（label）：\n",labels_onehot)
# [[0 0 0... 0 0 0]
#  [0 0 0... 1 0 0]
#  [0 1 0 ... 0 0 0]
#  ...
#  [0 0 0 ... 0 0 1]
#  [0 0 1 ... 0 0 0]
#  [0 0 0 ... 0 0 0]]
print(labels_onehot.shape)
# (2708, 7)
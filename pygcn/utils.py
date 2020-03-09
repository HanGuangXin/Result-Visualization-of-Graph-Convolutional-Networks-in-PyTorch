import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)       # set() 函数创建一个无序不重复元素集

    # enumerate()函数生成序列，带有索引i和值c。
    # 这一句将string类型的label变为onehot编码的label，建立映射关系
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    # map() 会根据提供的函数对指定序列做映射。
    # 这一句将string类型的label替换为onehot编码的label
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    # 返回int类型的label
    return labels_onehot

'''数据读取'''
# 更改路径。由../改为C:\Users\73416\PycharmProjects\PyGCN
def load_data(path="C:/Users/73416/PycharmProjects/PyGCN_Visualization/data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    ''' 
    cora.content 介绍：
    cora.content共有2708行，每一行代表一个样本点，即一篇论文。
    每一行由三部分组成:
    是论文的编号，如31336；
    论文的词向量，一个有1433位的二进制；
    论文的类别，如Neural_Networks。总共7种类别（label）
    第一个是论文编号，最后一个是论文类别，中间是自己的信息（feature）
    '''

    '''读取feature和label'''
    # 以字符串形式读取数据集文件：各自的信息。
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))

    # csr_matrix：Compressed Sparse Row marix，稀疏np.array的压缩
    # idx_features_labels[:, 1:-1]表明跳过论文编号和论文类别，只取自己的信息（feature of node）
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)

    # idx_features_labels[:, -1]表示只取最后一个，即论文类别，得到的返回值为int类型的label
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    # idx_features_labelsidx_features_labels[:, 0]表示取论文编号
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)

    # 通过建立论文序号的序列，得到论文序号的字典
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    # 进行一次论文序号的映射
    # 论文编号没有用，需要重新的其进行编号（从0开始），然后对原编号进行替换。
    # 所以目的是把离散的原始的编号，变成0 - 2707的连续编号
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    # coo_matrix()：系数矩阵的压缩。分别定义有那些非零元素，以及各个非零元素对应的row和col，最后定义稀疏矩阵的shape。
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    # np.multiply()函数，数组和矩阵对应位置相乘，输出与相乘数组/矩阵的大小一致
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # feature和adj归一化
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))     # adj在归一化之前，先引入自环

    # train set, validation set, test set的分组。
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    # 数据类型转tensor
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # 返回数据
    return adj, features, labels, idx_train, idx_val, idx_test


'''归一化函数'''
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))                # (2708, 1)
    r_inv = np.power(rowsum, -1).flatten()      # (2708,)
    r_inv[np.isinf(r_inv)] = 0.                 # 处理除数为0导致的inf
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

'''计算accuracy'''
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

'''稀疏矩阵转稀疏张量'''
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

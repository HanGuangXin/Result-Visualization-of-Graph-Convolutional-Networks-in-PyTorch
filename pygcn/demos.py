import numpy as np
import scipy.sparse as sp
from pygcn.utils import normalize,sparse_mx_to_torch_sparse_tensor,encode_onehot
import torch

'''测试论文编号处理'''
a = np.arange(9).reshape((3,3))
print(np.where(a==np.max(a,axis=1)))
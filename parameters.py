import numpy as np
classes = {'FAVOR': np.array([1, 0, 0]), 'NONE': np.array([0, 1, 0]), 'AGAINST': np.array([0, 0, 1])}
classes_ = np.array(['FAVOR', 'NONE', 'AGAINST'])

# bitterlemons数据集
bitlem_classes = {'ISRAELI': np.array([1, 0]), 'PALESTINIAN': np.array([0, 1])}
bitlem_classes_ = np.array(['ISRAELI', 'PALESTINIAN'])

# 三个twitter数据集的label映射
twitter_classes = {}
twitter_classes_ = np.array([])

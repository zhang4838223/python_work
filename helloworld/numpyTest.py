import numpy as np

#创建4维数组
arr = np.random.rand(4,4)
print(arr)
#将数组转为矩阵
mat = np.mat(arr)
#获取矩阵的逆矩阵
mi = mat.I
print(mat * mi)

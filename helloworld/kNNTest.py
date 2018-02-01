from kNN import *
import matplotlib
import matplotlib.pyplot as plt
#简单分类算法demo1
#group,labels = createDataSet()
#res = classify0([0, 0], group, labels, 3)
#print(res)

#简单分类算法demo2
datingDataMat,datingLabels = fill2matrix('data/test.txt')
# print(datingDataMat)
# print(datingLabels[0:20])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
plt.show()
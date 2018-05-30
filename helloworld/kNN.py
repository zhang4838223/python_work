from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return  group, labels


def classify0(inX, dataSet, labels, k):
    #获取训练集的行数
    dataSize = dataSet.shape[0]
    #测试集复制行数后 - 训练集
    diffMat = tile(inX, (dataSize, 1)) - dataSet
    #平方差矩阵
    sqDiffMat = diffMat**2
    #平方差矩阵各个维度求和
    sqDistance =  sqDiffMat.sum(axis=1)
    #开方计算
    distances = sqDistance**0.5
    #计算距离数组升序后排序的索引位
    sortIndex = distances.argsort()
    classCout = {}

    #最近的k个元素对应的标签
    for i in range(k):
        voteLabel = labels[sortIndex[i]]
        classCout[voteLabel] = classCout.get(voteLabel, 0) + 1
        sortedClassCount = sorted(classCout.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def fill2matrix(filename):
    #打开文件
    fr = open(filename)
    #获取文件行数组
    fileLines = fr.readlines()
    #获取行数
    lineCounts = len(fileLines)
    # print(fileLines)
    #创建返回的训练数据矩阵
    returnMat = zeros((lineCounts, 3))
    #创建返回的分类结果
    classVetor = []
    index = 0

    for line in fileLines:
        #去掉回车
        str = line.strip()
        # print(line)
        #根据,号将字符串切割为数组
        listLine = str.split("#")
        # print(listLine)
        listLine = list(map(float, listLine))
        #将前三个元素赋值给训练矩阵
        returnMat[index,:] = listLine[0:3]
        #将最后一列元素添加到分类集合中
        # print(listLine[-1])
        index += 1
        classVetor.append(listLine[-1])
    return returnMat,classVetor

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))

    return normDataSet, ranges, minVals

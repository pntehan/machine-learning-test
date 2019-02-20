import numpy as np
import operator
import os

def createDataSet():
    # 创建数据集
    group = np.array([[1.0, 1.1],
        [1.0, 1.0],
        [0, 0],
        [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    # 判断输入数据类别
    dataSetSize = dataSet.shape[0]
    # 距离计算
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    dsitances = sqDistances**0.5
    sortedDistIndicies = dsitances.argsort()
    classCount = {}
    # 选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        # print(classCount)
        sortedClassCount = sorted(classCount.items(),
            key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    # 读取文件数据和类型返回数据数据和类别数组
    with open(filename) as fr:
        arrayOlines = fr.readlines()
    numberOlines = len(arrayOlines)
    # 创建数据
    returnMat = np.zeros((numberOlines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOlines:
        # 提取数据和类别
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0: 3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

def img2vector(filename):
    # 图片转为数组数据
    returnVect = np.zeros((1, 1024))
    with open(filename) as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

def autoNorm(dataSet):
    # 将数据归一化
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTsetVecs = int(m*hoRatio) # 训练次数
    errorCount = 0 # 错误次数
    for i in range(numTsetVecs):
        classifierResult = classify0(normMat[i, :],
                                     normMat[numTsetVecs:m, :],
                                     datingLabels[numTsetVecs:m],
                                     3)
        print('分类器给出的类别是:{}, 此测试数据真正类别是:{}'.format(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1
    print('错误率是:{}%'.format(errorCount/float(numTsetVecs)*100))

def classifyPerson():
    resultList = ['毫无魅力',
                  '有点魅力',
                  '很有魅力']
    percentTats = float(input('玩游戏的时间比例:'))
    ffMiles = float(input('每年的飞行里程:'))
    iceCream = float(input('每年消费冰淇淋升数:'))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,
                                 normMat, datingLabels, 3)
    print('你肯属于{}的人...'.format(resultList[classifierResult-1]))

def handwritingClassTest():
    # 手写字体识别
    hwLabels = []
    trainingFileList = os.listdir('number/trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('number/trainingDigits/%s'%fileNameStr)
    testFileList = os.listdir('number/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('number/testDigits/%s'%fileNameStr)
        classifierResult = classify0(vectorUnderTest,
                                     trainingMat,
                                     hwLabels,
                                     3)
        if classifierResult != classNumStr:
            errorCount += 1
            print('[RESULT]\n分类器识别为{}, 数字本身为{}...'.format(classifierResult, classNumStr))
    print('<错误率>\n{}%'.format((errorCount/float(mTest))*100))





























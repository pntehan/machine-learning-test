from numpy import *
from kNN import *
import matplotlib
import matplotlib.pyplot as plt

# # 读取文本数据并输出
# datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
# print(datingDataMat)
# print(datingLabels)

# # 读取训练数据绘图观察
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1],
# 		   15.0*array(datingLabels),
# 		   15.0*array(datingLabels))
# plt.show()

# # 训练数据归一化
# normMat, ranges, minVals = autoNorm(datingDataMat)
# print('<DataMat>:\n{}'.format(normMat))
# print('<Range>:\n{}'.format(ranges))
# print('<MinValues>:\n{}'.format(minVals))

# # 测试集检验错误率
# datingClassTest()
# # 输入数据得到类别
# classifyPerson()

# # 读取图片转换为数组
# data = img2vector('number/testDigits/0_13.txt')
# print(data[0, 0:31])

# # 测试手写字体
# handwritingClassTest()

















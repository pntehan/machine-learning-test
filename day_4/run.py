import logRegres
from numpy import *

# # 基于测试文档的梯度上升算法选择最优的Logistic回归函数
# dataArr, labelMat = logRegres.loadDataSet()
# result = logRegres.gradAscent(dataArr, labelMat)
# print(result)
#
# # 根据最优logistic画图
# logRegres.plotBestFit(result)
#
# # 随机梯度上升算法
# weights = logRegres.stocGradAscent1(array(dataArr), labelMat, numIter=500)
# print(weights)
# logRegres.plotBestFit(weights)

# 随机梯度上升算法最优logistic回归函数实战
logRegres.multiTest()







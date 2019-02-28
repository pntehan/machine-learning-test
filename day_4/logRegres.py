# logistic回归梯度上升优化算法
from numpy import *

def loadDataSet():
	# 导入测试算法数据集
	dataMat, labelMat = [], []
	with open('testSet.txt') as f:
		for line in f.readlines():
			lineArr = line.strip().split()
			dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
			labelMat.append(int(lineArr[2]))
		return dataMat, labelMat

def sigmoid(inX):
	# sigmoid阶跃函数表达式
	return 1.0/(1+exp(-inX))

def gradAscent(dataMath, classLabels):
	# 梯度上升算法
	dataMatrix = mat(dataMath)
	labelMat = mat(classLabels).transpose()
	m, n = shape(dataMatrix)
	alpha = 0.001
	maxCycles = 500
	weights = ones((n, 1))
	for k in range(maxCycles):
		h = sigmoid(dataMatrix*weights)
		error = (labelMat - h)
		weights = weights + alpha * dataMatrix.transpose() * error
	return weights

def plotBestFit(wei):
	# 根据数据集和算出的最优logistic回归最佳拟合直线的函数图
	import matplotlib.pyplot as plt
	weights = wei
	# 导入数据
	dataMat, labelMat = loadDataSet()
	dataArr = array(dataMat)
	n = shape(dataArr)[0]
	xcord1, ycord1 = [], []
	xcord2, ycord2 = [], []
	for i in range(n):
		if int(labelMat[i]) == 1:
			xcord1.append(dataArr[i,1])
			ycord1.append(dataArr[i,2])
		else:
			xcord2.append(dataArr[i, 1])
			ycord2.append(dataArr[i, 2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
	ax.scatter(xcord2, ycord2, s=30, c='green')
	x = arange(-3.0, 3.0, 0.1)
	y = (-weights[0]-weights[1]*x)/weights[2]
	ax.plot(x, y)
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.show()

def stocGradAscent0(dataMatrix, classLabels):
	# 随机梯度上升算法
	m, n = shape(dataMatrix)
	alpha = 0.01
	weights = ones(n)
	for i in range(m):
		h = sigmoid(sum(dataMatrix[i]*weights))
		error = classLabels[i] - h
		weights = weights + alpha * error * dataMatrix[i]
	return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
	# 随机梯度上升算法的优化版
	m, n = shape(dataMatrix)
	weights = ones(n)
	for j in range(numIter):
		dataIndex = list(range(m))
		for i in range(m):
			# alpha每次迭代需要调整
			alpha = 4/(1.0+j+i)+0.01
			# 随机选取更新
			randIndex = int(random.uniform(0, len(dataIndex)))
			h = sigmoid(sum(dataMatrix[randIndex]*weights))
			error = classLabels[randIndex] - h
			weights = weights + alpha * error * dataMatrix[randIndex]
			del dataIndex[randIndex]
	return weights

def classifyVector(inX, weights):
	# logistic算法实现
	prob = sigmoid(sum(inX*weights))
	if prob>0.5:
		return 1
	else:
		return 0

def colicTest():
	# 将训练数据和测试数据导入进行计算测试
	frTrain = open('horseColicTraining.txt')
	frTest = open('horseColicTest.txt')
	trainingSet, trainingLabels = [], []
	for line in frTrain.readlines():
		currline = line.strip().split('\t')
		lineArr = []
		for i in range(21):
			lineArr.append(float(currline[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(currline[21]))
	frTrain.close()
	trainWeights = stocGradAscent1(array(trainingSet),
									trainingLabels,
									numIter=500)
	errorCount, numTestVec = 0, 0.0
	for line in frTest.readlines():
		numTestVec += 1.0
		currline = line.strip().split('\t')
		lineArr = []
		for i in range(21):
			lineArr.append(float(currline[i]))
		if int(classifyVector(array(lineArr), trainWeights)) != int(currline[21]):
			errorCount += 1
	errorRate = (float(errorCount)/numTestVec)
	print('The error rate of this test is: {}'.format(errorRate))
	return errorRate

def multiTest():
	# 算法测试十次记录每次错误率
	numTests, errorSum = 10, 0.0
	for k in range(numTests):
		errorSum += colicTest()
	print('After {} iterations the average error rate is: {}'.format(numTests, errorSum/float(numTests)))





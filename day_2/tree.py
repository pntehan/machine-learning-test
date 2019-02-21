from math import log
import operator

def calcShannonEnt(dataSet):
	# 计算数据集的熵值
	numEntries = len(dataSet)
	labelsCounts = {}
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel not in labelsCounts.keys():
			labelsCounts[currentLabel] = 0
		labelsCounts[currentLabel] += 1
	shannonEnt = 0.0
	for key in labelsCounts:
		prob = float(labelsCounts[key]/numEntries)
		shannonEnt -= prob * log(prob, 2)
	return shannonEnt

def createDataSet():
	dataSet = [['有', '有', '鸟'],
				['无', '无', '不是鸟'],
				['无', '有', '不是鸟'],
				['有', '无', '鸟']]
	labels = ['翅膀', '羽毛']
	return dataSet, labels

def splitDataSet(dataSet, axis, value):
	# 按特征划分数据集
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet

def chooseBestFeatureToSplit(dataSet):
	# 选择最好的数据划分方式
	numFeatures = len(dataSet[0]) - 1
	baseEntropy = calcShannonEnt(dataSet)
	bestInfoGain, bestFeature = 0.0, 0
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)
		newEntroy = 0.0
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet)
			newEntroy += prob*calcShannonEnt(subDataSet)
			infoGain = baseEntropy - newEntroy
			if infoGain > bestInfoGain:
				bestInfoGain = infoGain
				bestFeature = i
	return bestFeature

def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 0
		classCount += 1
	sortedClassCount = sorted(classCount.items(),
								key=operator.itemgetter(1),
								reverse=True)
	return sortedClassCount[0][0]

def createTree(dataSet, labels):
	# 递归创建决策树
	classList = [example[-1] for example in dataSet]
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)
	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	myTree = {bestFeatLabel: {}}
	del labels[bestFeat]
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)
	for value in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
	return myTree

def classify(inputTree, featLabels, testVec):
	# 使用决策树分类数据
	firstStr = list(inputTree.keys())[0]
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr)
	for key in list(secondDict.keys()):
		if testVec[featIndex] == key:
			if type(secondDict[key]).__name__ == 'dict':
				classLabel = classify(secondDict[key],
									  featLabels,
									  testVec)
			else:
				classLabel = secondDict[key]
	return classLabel

def storeTree(inputTree, filename):
	# 将决策树写入文件
	import pickle
	with open(filename, 'w') as f:
		print(inputTree)
		pickle.dump(inputTree, f)

def grabTree(filename):
	# 从文件中读取决策树写入内存
	import pickle
	with open(filename) as f:
		return pickle.load(f)











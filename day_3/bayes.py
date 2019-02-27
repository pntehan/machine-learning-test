from numpy import *

def loadDataSet():
	# 构建词向量
	postingList = [['my', 'dog', 'has', 'flea', 'problems', 'please'],
					['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
					['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
					['stop', 'posting', 'stupid', 'worthless', 'garbage'],
					['my', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
					['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0, 1, 0, 1, 0, 1]
	return postingList, classVec

def createVocabList(dataSet):
	# 将输入的数据集转换为不重复词的列表
	vocabSet = set([])
	for document in dataSet:
		if isinstance(document, int):
			pass
		else:
			vocabSet = vocabSet | set(document)
	return list(vocabSet)

def set0fWords2Vec(vocaList, inputSet):
	# 贝叶斯词集模型
	returnVec = [0] * len(vocaList)
	for word in inputSet:
		if word in vocaList:
			returnVec[vocaList.index(word)] = 1
		else:
			print("The word <{}> is not in my Vocabulary!".format(word))
	return returnVec

def bag0fWords2VecMN(vocaList, inputSet):
	# 贝叶斯词袋模型
	returnVec = [0] * len(vocaList)
	for word in inputSet:
		if word in vocaList:
			returnVec[vocaList.index(word)] += 1
		else:
			print("The word <{}> is not in my Vocabulary!".format(word))
	return returnVec

def trainNBO(trainMatrix, trainCategory):
	# 朴素贝叶斯分类器训练函数
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	pAbusive = sum(trainCategory)/float(numTrainDocs)
	p0Num, p1Num = ones(numWords), ones(numWords)
	p0Denom, p1Denom = 2.0, 2.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	p1Vect = log(p1Num/p1Denom)
	p0Vect = log(p0Num/p0Denom)
	return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	# 朴素贝叶斯分类函数
	p1 = sum(vec2Classify * p1Vec) + log(pClass1)
	p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
	if p1 > p0:
		return 1
	else:
		return 0

def testingNB():
	# 测试贝叶斯分类器
	listOPost, listClasses = loadDataSet()
	myVocabList = createVocabList(listOPost)
	trainMat = []
	for postinDoc in listOPost:
		trainMat.append(set0fWords2Vec(myVocabList, postinDoc))
	p0V, p1V, pAb = trainNBO(array(trainMat), array(listClasses))
	testEntry = ['love', 'my', 'dalmation']
	thisDoc = array(set0fWords2Vec(myVocabList, testEntry))
	print('{}测试集的分类结果是{}'.format(
		testEntry, classifyNB(thisDoc, p0V, p1V, pAb)
	))
	testEntry = ['stupid', 'garbage']
	thisDoc = array(set0fWords2Vec(myVocabList, testEntry))
	print('{}测试集的分类结果是{}'.format(
		testEntry, classifyNB(thisDoc, p0V, p1V, pAb)
	))

def textParse(bigString):
	# 文本数据切割
	import re
	listOfTokens = re.split(r'\W*', bigString)
	wordlist = [tok.lower() for tok in listOfTokens if len(tok) > 2]
	# print(wordlist)
	return wordlist

def spamTest():
	# 朴素分类器邮件实战
	docList, fullText, classList = [], [], []
	for i in range(1, 26):
		with open('email/spam/%d.txt'%i) as f:
			data = f.read()
		wordlist = textParse(data)
		docList.append(wordlist)
		fullText.append(wordlist)
		classList.append(1)
		with open('email/ham/%d.txt'%i) as f:
			data = f.read()
		wordlist = textParse(data)
		docList.append(wordlist)
		fullText.append(wordlist)
		classList.append(0)
	vocabList = createVocabList(docList)
	trainingSet = list(range(50))
	testSet = []
	for i in range(10):
		randIndex = int(random.uniform(0, len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del trainingSet[randIndex]
	trainMat, trainClasses = [], []
	for docIndex in trainingSet:
		trainMat.append(set0fWords2Vec(vocabList,
										docList[docIndex]))
		trainClasses.append(classList[docIndex])
	p0V, p1V, pSpam = trainNBO(array(trainMat), array(trainClasses))
	errorCount = 0
	for docIndex in testSet:
		wordVector = set0fWords2Vec(vocabList, docList[docIndex])
		if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
			errorCount += 1
			print('测试文档\n{}\n本身类别为{}, 测试类别为 {}...'.format(docList[docIndex], classList[docIndex], classifyNB(array(wordVector), p0V, p1V, pSpam)))
	print('本分类器的错误是{}'.format(float(errorCount)/len(testSet)))
















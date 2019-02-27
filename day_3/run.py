__author__ = 'Pntehan'

import bayes
from numpy import *
import re

# # 创建不重复词列表
# listPosts, listClasses = bayes.loadDataSet()
# myVocabList = bayes.createVocabList(listPosts)
# print(myVocabList)

# # 根据列表离取特征值
# result = bayes.set0fWords2Vec(myVocabList, listPosts[0])
# print(listPosts[0])
# print(result)

# # 实现文档的二分类
# list0Posts, listClasses = bayes.loadDataSet()
# myVocabList = bayes.createVocabList(list0Posts)
# print(myVocabList)
# trainMat = []
# for postinDoc in list0Posts:
# 	trainMat.append(bayes.set0fWords2Vec(myVocabList, postinDoc))
# p0V, p1V, pAb = bayes.trainNBO(trainMat, listClasses)
# print(p0V)
# print(p1V)
# print(pAb)

# # 朴素贝叶斯分类器的测试
# bayes.testingNB()

# # 文本数据切割
# regEx = re.compile('\\W*')
# with open('email/ham/6.txt') as f:
# 	emailText = f.read()
# listOfTokens = regEx.split(emailText)
# print(listOfTokens)

# 贝叶斯分类器实战
bayes.spamTest()











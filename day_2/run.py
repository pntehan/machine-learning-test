import tree
import matplotlib.pyplot as plt
import treePlot

# # 计算数据集的熵值
# myDataSet, myLabel = tree.createDataSet()
# print(tree.calcShannonEnt(myDataSet))

# # 按照特征划分数据集
# myDataSet, myLabel = tree.createDataSet()
# result = tree.splitDataSet(myDataSet, 0, 1)
# print(result)

# # 选择最好的划分方式
# myDataSet, myLabel = tree.createDataSet()
# result = tree.chooseBestFeatureToSplit(myDataSet)
# print(result)

# 递归创建决策树
# myDataSet, myLabel = tree.createDataSet()
# myTree = tree.createTree(myDataSet, myLabel)
# print(myTree)

# # 绘制决策树
# myTree = treePlot.retrieveTree()
# treePlot.createPlot(myTree )

# # 测试决策树分类器
# # 训练数据
# myDat, Labels = tree.createDataSet()
# # 导出训练集决策树
# myTree = treePlot.retrieveTree()
# # 输入测试集
# result = tree.classify(myTree, Labels, ['无', '无'])
# print('结果是:%s...'%(result))

# # 决策树读写内存操作
# myDat, Labels = tree.createDataSet()
# # 导出训练集决策树
# myTree = tree.createTree(myDat, Labels)
# tree.storeTree(myTree, 'classifierStorage.txt')
# print(tree.grabTree('classifierStorage.txt'))

# 读取训练集文件数据输出决策树
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
with open('lenses.txt') as fp:
	lenses = [inst.strip('\n').split('\t') for inst in fp.readlines()]
	lensesTree = tree.createTree(lenses, lensesLabels)
	print(lensesTree)
# treePlot.createPlot(lensesTree)

















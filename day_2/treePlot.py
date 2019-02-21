import matplotlib.pyplot as plt
# 绘制决策树的图形

decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')

def plotNode(nodeText, centerPt, parentPt, nodeType):
	# 设置节点属性
	createPlot.ax1.annotate(nodeText, xy=parentPt,
							xycoords='axes fraction',
							xytext=centerPt,
							textcoords='axes fraction',
							va='center', ha='center',
							bbox=nodeType,
							arrowprops=arrow_args)

def getNumLeafs(myTree):
	# 得到树的叶子数
	numleafs = 0
	keys = list(myTree.keys())
	firstStr = keys[0]
	secondDict = myTree[firstStr]
	for key in list(secondDict.keys()):
		if type(secondDict[key]).__name__ == 'dict':
			numleafs += getNumLeafs(secondDict[key])
		else:
			numleafs += 1
	return numleafs

def getTreeDepth(myTree):
	# 得到树的深度
	maxDepth = 0
	keys = list(myTree.keys())
	firstStr = keys[0]
	secondDict = myTree[firstStr]
	for key in list(secondDict.keys()):
		if type(secondDict[key]).__name__ == 'dict':
			treeDepth = 1 + getTreeDepth(secondDict[key])
		else:
			treeDepth = 1
		if treeDepth > maxDepth:
			maxDepth = treeDepth
	return maxDepth

def retrieveTree():
	listOfTrees = [{'翅膀': {'无': '不是鸟', '有': '鸟'}}]
	return listOfTrees[0]

def plotMidText(cntrPt, parentPt, txtString):
	xMid = (parentPt[0]-cntrPt[0]/2.0)+cntrPt[0]
	yMid = (parentPt[1]-cntrPt[1]/2.0)+cntrPt[1]
	createPlot.ax1.text(xMid, yMid, txtString,
						va='center', ha='center',
						rotation=30)

def plotTree(myTree, parentPt, nodeTxt):
	numLeafs = getNumLeafs(myTree)
	depth = getTreeDepth(myTree)
	firstStr = list(myTree.keys())[0]
	# print(plotTree.totalW, plotTree.totalD, plotTree.xOff, plotTree.xOff)
	cntrPt = (plotTree.xOff+(1.0+float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
	plotMidText(cntrPt, parentPt, nodeTxt)
	plotNode(firstStr, cntrPt, parentPt, decisionNode)
	secondDict = myTree[firstStr]
	plotTree.yOff = plotTree.yOff-1.0/plotTree.totalD
	for key in list(secondDict.keys()):
		if type(secondDict[key]).__name__ == 'dict':
			plotTree(secondDict[key], cntrPt, str(key))
		else:
			plotTree.xOff = plotTree.xOff+1.0/plotTree.totalW
			plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
			plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
			plotTree.yOff = plotTree.yOff+1.0/plotTree.totalD


def createPlot(inTree):
	# 创建节点
	fig = plt.figure(1, facecolor='white')
	fig.clf()
	axprops = dict(xticks=[], yticks=[])
	createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
	plotNode('决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
	plotNode('叶节点', (0.8, 0.1), (0.3, 0.8), leafNode)
	plotTree.totalW = float(getNumLeafs(inTree))
	plotTree.totalD = float(getTreeDepth(inTree))
	plotTree.xOff = -0.5 / plotTree.totalW
	plotTree.yOff = 1.0
	plotTree(inTree, (0.5,1.0), '')
	plt.show()


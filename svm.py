# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 10:17:16 2017

@author: dell
"""

import random
from numpy import *
import operator
import time

def loadDataSet(path):
    """
    读取数据集和类别，六维数据
    """
    dataMat=[]; labelMat=[]
    with open(path) as fr:
        for line in fr.readlines():
            line = line.strip().split('\t')
            dataMat.append([float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5])])
            labelMat.append(float(line[6]))
    return dataMat, labelMat

def get_train_and_test(dataMat, labelMat):
    train_dataMat = dataMat.copy()
    train_labelMat = labelMat.copy()
    test_dataMat = []
    test_labelMat = []
    for i in range(47):
        j = int(random.uniform(0,len(train_dataMat)))
        test_dataMat.append(train_dataMat[j])
        test_labelMat.append(train_labelMat[j])
        del train_dataMat[j]
        del train_labelMat[j]
    return train_dataMat, train_labelMat, test_dataMat, test_labelMat

def newLabel(dataMat, labelMat, labelNum):
    labelMatLen = len(labelMat)
    newLabelMat = zeros((labelNum, labelMatLen))
    for i in range(labelNum):
        for j in range(labelMatLen):
            if (labelMat[j] == i):
                newLabelMat[i][j] = 1
            else:
                newLabelMat[i][j] = -1
    return newLabelMat

def nSVC(dataMat, labelMat, labelNum):
    pass

def clipAlpha(aj, H, L):
    """
    修剪alpha到其规定范围内
    """
    if aj>H:
        return H
    if aj<L:
        return L
    return aj

def setJrand(i, m):
    """
    0到m的随机整型数据
    """
    j = i
    while (j==i):
        j = int(random.uniform(0, m))
    return j

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        """
        初始化：特征集，类别，常数C
        """
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(self.X)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))

def calEk(os, k):
    """
    Ek:真实值和预测值的差值
    """
    fXk = float(multiply(os.alphas * os.labelMat).T * (os.X * os.X[k, :].T)) + os.b
    Ek = fXk - float(os.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):
    """
    选择一个alpha
    """
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej      

def updateEk(oS, k):
    """
    更新Ek
    """
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if (L == H):
            # print("L == H")
            return 0
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0:
            # print("eta >= 0")
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            # print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0          

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    """
    输入：数据集, 类别标签, 常数C, 容错率, 最大循环次数
    输出：目标b, 参数alphas
    """
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
    iterr = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iterr < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
            # print("fullSet, iter: %d i:%d, pairs changed %d" % (iterr, i, alphaPairsChanged))
            iterr += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                # print("non-bound, iter: %d i:%d, pairs changed %d" % (iterr, i, alphaPairsChanged))
            iterr += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        # print("iteration number: %d" % iterr)
    return oS.b, oS.alphas

def smoSimple(dataMatIn, classLabels, C, toler, maxIter): 
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    b = 0; m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))
    iter = 0
    while(iter<maxIter):  #迭代次数
        alphaPairsChanged=0 
        for i in range(m):  #在数据集上遍历每一个alpha
            #print alphas 
            #print labelMat
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            #fXi=float(np.multiply(alphas, labelMat).T*dataMatrix*dataMatrix[i, :].T)+b  #.T也是转置
            Ei=fXi-float(labelMat[i]) 
            if((labelMat[i]*Ei<-toler) and (alphas[i]<C)) or ((labelMat[i]*Ei>toler) and (alphas[i]>0)): 
                j=selectJrand(i, m)  #从m中选择一个随机数，第2个alpha j
                fXj=float(multiply(alphas, labelMat).T*dataMatrix*dataMatrix[j, :].T)+b 
                Ej=fXj-float(labelMat[j]) 
                
                alphaIold=alphas[i].copy()  #复制下来，便于比较
                alphaJold=alphas[j].copy() 
                
                if(labelMat[i]!=labelMat[j]):  #开始计算L和H
                    L=max(0, alphas[j]-alphas[i]) 
                    H=min(C, C+alphas[j]-alphas[i]) 
                else: 
                    L=max(0, alphas[j]+alphas[i]-C) 
                    H=min(C, alphas[j]+alphas[i]) 
                if L==H: 
#                    print ('L==H') 
                    continue 
                
                #eta是alphas[j]的最优修改量，如果eta为零，退出for当前循环
                eta=2.0*dataMatrix[i, :]*dataMatrix[j, :].T-\
                    dataMatrix[i, :]*dataMatrix[i, :].T-\
                    dataMatrix[j, :]*dataMatrix[j, :].T 
                if eta>=0: 
#                    print ('eta>=0')
                    continue 
                alphas[j]-=labelMat[j]*(Ei-Ej)/eta  #调整alphas[j] 
                alphas[j]=clipAlpha(alphas[j], H, L)  
                if(abs(alphas[j]-alphaJold)<0.00001):  #如果alphas[j]没有调整
#                    print ('j not moving enough')
                    continue 
                alphas[i]+=labelMat[j]*labelMat[i]*(alphaJold-alphas[j])  #调整alphas[i]
                b1=b-Ei-labelMat[i]*(alphas[i]-alphaIold)*\
                    dataMatrix[i, :]*dataMatrix[i, :].T-\
                    labelMat[j]*(alphas[j]-alphaJold)*\
                    dataMatrix[i, :]*dataMatrix[j, :].T 
                b2=b-Ej-labelMat[i]*(alphas[i]-alphaIold)*\
                    dataMatrix[i, :]*dataMatrix[j, :].T-\
                    labelMat[j]*(alphas[j]-alphaJold)*\
                    dataMatrix[j, :]*dataMatrix[j, :].T 
                
                if(0<alphas[i]) and (C>alphas[i]): 
                    b=b1 
                elif(0<alphas[j]) and (C>alphas[j]): 
                    b=b2 
                else: 
                    b=(b1+b2)/2.0 
                alphaPairsChanged+=1 
                
#                print ('iter: %d i: %d, pairs changed %d' %(iter, i, alphaPairsChanged))
        if(alphaPairsChanged==0): 
            iter+=1 
        else: 
            iter=0 
#        print ('iteration number: %d' %iter)
    return b, alphas 


def calWs(alphas, dataArr, classLabels):
    """
    输入：alphas, 数据集, 类别标签
    输出：目标w
    """
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w

def calF(ws, b, testDataSet):
    res = []
    dataLen = len(testDataSet)
    dataMat = mat(testDataSet)
    ws = mat(ws)
    for i in range(dataLen):
        fx = float(ws.T * dataMat[i].T) + b
        res.append(fx)
    return res
    

def multiWsAlpha(dataMat, labelMat, testDataSet, testLabel, labelNum):
    dataLen = len(testDataSet)
    res = zeros((labelNum, dataLen))
    for i in range(labelNum):
        b, alphas = smoSimple(dataMat, labelMat[i], 0.6, 0.0001, 40)
        ws = calWs(alphas, dataMat, labelMat[i])
        res[i] = calF(ws, b, testDataSet)
    return res

def voteRes(res, testLabel, labelNum):
    finalRes = []
    labelLen = len(testLabel[0])
    for i in range(labelLen):
        maxRes = -999
        for j in range(labelNum):
            if (maxRes < res[j][i]):
                maxRes = res[j][i]
                maxI = j
        finalRes.append(maxI)
    return finalRes

def testRes(finalRes, originTrainLabel):
    count = 0
    for i in range(len(finalRes)):
        if finalRes[i]==originTrainLabel[i]:
            count+=1
    return (count/len(finalRes))

def main():
    labelNum = 5
    trainDataSet, trainLabel = loadDataSet('dataset0718.txt')
    trainDataSet, trainLabel, testDataSet, testLabel =\
        get_train_and_test(trainDataSet, trainLabel)
    originTrainLabel = testLabel
    trainLabel = newLabel(trainDataSet, trainLabel, labelNum)
    testLabel = newLabel(testDataSet, testLabel, labelNum)
    res = multiWsAlpha(trainDataSet, trainLabel, testDataSet, testLabel, labelNum)
    finalRes = voteRes(res, testLabel, labelNum)
    acc = testRes(finalRes, originTrainLabel)
    print(acc)

if __name__ == '__main__':
    start = time.clock()
    main()
    end = time.clock()
    print("finish in time: %s" % str(end - start))
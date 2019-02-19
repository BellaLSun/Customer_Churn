#author: xiaolinhan_daisy
#date: 2018/01/13
#site: YueJiaZhuang
from numpy import *

def loadDataSet():
    x = [0, 1, 2, 3, 4, 5]
    y = [1, 1, -1, -1, 1, -1]
    return x, y

def generateGxList(x):
    gxlist = []
    for i in range(len(x) - 1):
        gx = (x[i] + x[i + 1]) / 2
        gxlist.append(gx)
    return gxlist

def calcErrorNum(gx, x, y, weight):
    #判断以gx为切分点的两种方式里，哪种会让误差更小
    error1 = 0
    errorNeg1 = 0
    ygx = 1
    for i in range(len(x)):
        if i < gx and y[i] != 1: error1 += weight[i]
        if i > gx and y[i] != -1: error1 += weight[i]
        if i < gx and y[i] != -1: errorNeg1 += weight[i]
        if i > gx and y[i] != 1: errorNeg1 += weight[i]
    if errorNeg1 < error1:
        return errorNeg1, -1 #x>gx,则fgx = 1
    return error1, 1 #x<gx, 则fgx = 1

def calcAlpha(minError):
    alpha = 1/2 * log((1-minError)/minError)
    return alpha

def calcNewWeight(alpha,ygx, weight, gx, y):
    newWeight = []
    sumWeight = 0
    for i in range(len(weight)):
        flag = 1
        if i < gx and y[i] != ygx: flag = -1
        if i > gx and y[i] != -ygx: flag = -1
        weighti = weight[i]*exp(-alpha*flag)
        newWeight.append(weighti)
        sumWeight += weighti
    newWeight = newWeight / sumWeight

    return newWeight

def trainfxi(fx, i, x, y, weight):
    minError = inf
    bestGx = 0.5
    gxlist = generateGxList(x)
    bestygx = 1
    # 计算基本分类器
    for xi in gxlist:
        error, ygx = calcErrorNum(xi, x, y, weight)
        if error < minError:
            minError = error
            bestGx = xi
            bestygx = ygx
    fx[i]['gx'] = bestGx
    #计算alpha
    alpha = calcAlpha(minError)
    fx[i]['alpha'] = alpha
    fx[i]['ygx'] = bestygx
    #计算新的训练数据权值
    newWeight = calcNewWeight(alpha,bestygx, weight, bestGx, y)
    return newWeight

def calcFxError(fx, n, x, y):
    errorNum = 0
    for i in range(len(x)):
        fi = 0
        for j in range(n):
            fxiAlpha = fx[j]['alpha']
            fxiGx = fx[j]['gx']
            ygx = fx[j]['ygx']
            if i < fxiGx: fgx = ygx
            else: fgx = -ygx
            fi += fxiAlpha * fgx
        if sign(fi) != y[i]: errorNum += 1

    return errorNum/len(x)

def trainAdaBoost(x, y, errorThreshold, maxIterNum):
    fx = {}
    weight = []
    xNum = len(x)
    for i in range(xNum):
        w = float(1/xNum)
        weight.append(w)

    for i in range(maxIterNum):
        fx[i] = {}
        newWeight = trainfxi(fx, i, x, y, weight)
        weight = newWeight
        fxError = calcFxError(fx, (i+1), x, y)
        if fxError < errorThreshold: break

    return fx

if __name__ == '__main__':
    x, y = loadDataSet()
    errorThreshold = 0.01
    maxIterNum = 10
    fx = trainAdaBoost(x, y, errorThreshold, maxIterNum)
    print(fx)
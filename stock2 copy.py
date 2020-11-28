#stock.py
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import grad
from torch.optim import Optimizer
import math
from torch.utils.data import Dataset, DataLoader
import csv
import os
import scipy
import scipy.linalg
import scipy.stats
import time
from scipy.stats import norm
import copy

def loadnpz(name, allow_pickle=False):
    if allow_pickle:
        data = np.load(name, allow_pickle=True)
    else:
        data = np.load(name)
    data = data.f.arr_0
    return data

def giveFNames():
    fnames = os.listdir('./inputData/ITCH/rawCSV/')
    fnames.remove('.DS_Store')
    names = []
    nums = []
    for a in range(0, len(fnames)):
        name = fnames[a][:-8]
        name1 = int(name[:4])
        name2 = int(name[4:6])
        name3 = int(name[6:])
        #print (name1, name2, name3)
        #print (name)
        num = (name1 * 100000) + (name2 * 100) + name3
        nums.append(num)
        names.append(fnames[a][:-8])
    args = np.argsort(np.array(nums))
    names = list(np.array(names)[args])

    names = names#[:59]
    return names


def stefan_nextAfter(ar1, ar2, withCut=False):
    #First after or same.
    #ar1_argsort = np.argsort(ar1)
    #ar2_argsort = np.argsort(ar2)
    #ar1 = ar1[ar1_argsort]
    #ar2 = ar2[ar2_argsort]
    #print (time.time() - 1585183000)
    #print ("A")
    #group = np.array(list(np.zeros(ar1.shape[0])) + list(np.ones(ar2.shape[0]))).astype(int)
    group = np.concatenate((np.zeros(ar1.shape[0]), np.ones(ar2.shape[0])), axis=0).astype(int)
    #print (time.time() - 1585183000)
    #times = np.array(list(ar1) + list(ar2)).astype(int)
    times = np.concatenate((ar1*2, (ar2*2)+1), axis=0).astype(int) #Todo, just added 0.1
    #print (time.time() - 1585183000)
    time_argsort = np.argsort(times)
    #print (time.time() - 1585183000)
    group = group[time_argsort]
    #print (time.time() - 1585183000)
    group_sum = np.cumsum(group)
    #print (time.time() - 1585183000)
    ar2_indices = group_sum[group == 0]
    #print (time.time() - 1585183000)

    #ar2_indices[ar1_argsort] = np.copy(ar2_indices)
    #ar2_indices = ar2_argsort[ar2_indices]

    if withCut:
        ar2_indices = ar2_indices[ar2_indices < ar2.shape[0]]
    return ar2_indices

def initialITCHProcessing(data):
    #info1 = np.array(['Time', 'original T', 'order num', 'T', 'shares', 'price', 'exchange', 'original shares'])
    #print (info1[np.array([0, 4, 5, 3, 1, 7, 2, 6])])
    #print (['Time', 'Shares', 'Price', 'T', 'original T',  'original shares', 'order num', 'exchange'])
    #quit()

    data = data[1:]
    orders = data[:, 2]
    #zeroIndex = np.squeeze(np.argwhere(orders == "0"))
    unique1, indices1, counts1 = np.unique(orders, return_index=True, return_counts=True)
    unique2, indices2 = np.unique(orders[-1::-1], return_index=True)
    indices2 = orders.shape[0] - 1 - indices2

    args = np.arange(counts1.shape[0])[counts1 == 2]
    indices_1, indices_2 = indices1[args], indices2[args]
    data[indices_2, -3] = data[indices_1, -3]
    data[indices_2, -4] = data[indices_1, -4]
    data[indices_2, -1] = data[indices_1, 4]
    data[indices_1, -1] = data[indices_1, 4]

    #types = data[:, 3]
    #types1 = types[indices_1]
    #types2 = types[indices_2]
    data[indices_2, 1] = data[indices_1, 3]
    data[indices_1, 1] = data[indices_1, 3]
    #typeCombo1 = np.array(['BD', 'SD', 'BF', 'SF'])
    #comboNum1 = np.zeros(types1.shape)
    #comboNum1[types1 == "S"] = comboNum1[types1 == "S"] + 1
    #comboNum1[types1 == "F"] = comboNum1[types1 == "F"] + 2
    #data[indices_2, 3] = typeCombo1[comboNum1.astype(int)]

    typetemp = []
    args = np.arange(counts1.shape[0])[counts1 > 2]
    indices_1, indices_2, unique1 = indices1[args], indices2[args], unique1[args]
    count1 = 0
    for a in range(0, indices_1.shape[0]):
        vals = orders[indices_1[a]:indices_2[a]+1]
        val = unique1[a]
        args1 = np.squeeze(np.argwhere(vals == val)) + indices_1[a]
        #print (data[args1[0], 3])
        #typetemp.append(data[args1[0], 3])
        #if args1.shape[0] > 3:
        #    print (data[args1[1:]])
        typetemp = typetemp + list(data[args1[1:], 3])
        data[args1, 1] = data[args1[0], 3]
        data[args1[1:], -3] = data[args1[0], -3]
        sharesUsed = np.sum(data[args1[1:-1], -4].astype(int))
        data[args1[-1], -4] = str(int(data[args1[0], -4]) - sharesUsed)
        data[args1, -1] = data[args1[0], 4]
        #if args1.shape[0] > 3:
        #    print (data[args1[0]])
        #    print (data[args1[1:]])
        #    print (sharesUsed)
        #    count1 += 1
        #    if count1 == 10:
        #        quit()
        #quit()

    #print (data[:5])
    #['Time', 'original T', 'order num', 'T', 'shares', 'price', 'exchange', 'original shares']
    data = data[:, np.array([0, 4, 5, 3, 1, 7, 2, 6])] #['Time' 'Shares' 'Price' 'T', 'original T',  'original shares', 'order num', 'exchange']
    #print (['Time', 'Shares', 'Price', 'T', 'original T',  'original shares', 'order num', 'exchange'])
    #np.save('./inputData/ITCH/temporary/20200304_SPY_proc1.npy', data)
    return data



def convertITCHtoNumpy(name):
    import csv
    data = []
    #20200304
    with open('./inputData/ITCH/rawCSV/' + name + '_SPY.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        a = 0
        for row in spamreader:
            if len(row) > 0:
                data.append(row[0].split(","))
            a += 1
    data = np.array(data)
    np.savez_compressed('./inputData/ITCH/np/' + name + '_SPY.npz', data)



def saveFullITCH(names):
    for a in range(0, len(names)):
        print (a)
        name = names[a]#"20200303"
        #'''
        print ("A")
        convertITCHtoNumpy(name)
        print ("B")


        data = loadnpz('./inputData/ITCH/np/' + name + '_SPY.npz')
        data = initialITCHProcessing(data)
        np.savez_compressed('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz', data)
        print ("C")
        #quit()

        '''
        data = np.load('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npy')
        data = makeTransactionRelativePrices(data)
        data = filterOrderType(data)
        data = addUnusualShares(data)
        np.save('./inputData/ITCH/preXYprocessedFull/' + name + '_SPY.npy', data)
        print ("D")

        data = np.load('./inputData/ITCH/preXYprocessedFull/' + name + '_SPY.npy')
        data1 = xProcessingData(data)
        np.save('./inputData/ITCH/XinputITCH/' + name + '_SPY.npy', data1)
        del data1

        print ("E")
        data = quantizeITCH(data)
        np.save('./inputData/ITCH/QauntITCH/' + name + '_SPY.npy', data)
        print ("F")
        #'''

#names = giveFNames()
#names = names[:-52]
#print (len(names))
#saveFullITCH(names)
#quit()

class fastCancelModel(nn.Module):
    def __init__(self):
        super(fastCancelModel, self).__init__()
        self.nonlin = torch.tanh
        #self.nonlin = torch.relu

        self.lin1 = nn.Linear(80, 10)
        self.lin2 = nn.Linear(10, 15)
        self.lin3 = nn.Linear(15, 1) #39 or 2


    def forward(self, X):
        X = self.lin1(X)
        X = self.nonlin(X)
        X = self.lin2(X)
        X = self.nonlin(X)
        X = self.lin3(X)
        X = torch.sigmoid(X)
        X = X.reshape((X.shape[0],))
        return X

def multiProcessor(func1, start1, end1, chunk1, doPrint = False):
    from multiprocessing import Process
    len1 = end1 - start1
    len2 = ((len1 - 1) // chunk1) + 1

    for a in range(0, len2):
        time1 = time.time()
        ar1 = []
        start2 = (a * chunk1) + start1
        min1 = min(chunk1, end1 - start2)
        for b in range(0, min1):
            ar1.append(Process(target=func1, args=(start2 + b, )))
            ar1[b].start()
        for b in range(0, min1):
            ar1[b].join()

        if doPrint:
            print ((time.time() - time1)/60.0)

def statCor(m3, m4):
    #print (scipy.stats.pearsonr(m3, m4))
    m3, m4 = m3 - np.mean(m3), m4 - np.mean(m4)
    m3, m4 = m3 / (np.mean(m3**2.0)**0.5), m4 / (np.mean(m4**2.0)**0.5)
    #expectedm3m4 = 2 / np.pi
    absm3m4 = np.sum((m3 * m4) ** 2.0) ** 0.5
    m3m4 = np.sum(m3 * m4)
    x = (m3m4 / absm3m4)
    #prob = ((np.pi * 2.0) ** -0.5) * np.exp(-0.5 * (x ** 2.0))
    cor = m3m4 / m3.shape[0]
    prob = norm.cdf(-np.abs(x))
    #print (m3m4 / m3.shape[0], norm.cdf(-np.abs(x)))
    #print (prob)
    return cor, prob

    #print (expectedm3m4)
    #print (m3m4)
    'hi'

def norm1(x):
    return x / (np.mean(x**2.0) ** 0.5)

def makePositive(x):
    return (np.abs(x) + x) / 2

def stefan_cummax(ar):
    import pandas as pd
    ar = pd.Series(ar)
    ar = ar.cummax()
    ar = np.array(ar)
    return ar

def stefan_cummaxArg(ar):
    _, index1, ar1 = np.unique(ar, return_inverse=True, return_index=True)
    cummax1 = stefan_cummax(ar1)
    index1 = index1[cummax1]
    return ar[index1], index1

def stefan_firstBefore(ar1, ar2):
    #Last before or same
    return stefan_nextAfter(ar1+1, ar2) - 1

def stefan_cumMaxN(A, N):
    Amin = np.min(A)
    A = A - Amin + 1
    cumMaxFull = np.zeros((A.shape[0], N)).astype(int)
    for i in range(0, N):
        cumMax = stefan_cummax(A)
        cumMaxFull[:, i] = cumMax
        if i != N - 1:
            argsMax = np.argwhere((cumMax[1:] - cumMax[:-1]) != 0)[:, 0] + 2 # + 1 + 1
            argsMax = np.concatenate((np.array([0, 1]), argsMax))
            A1 = np.concatenate((np.array([-1]), A))
            A[argsMax[1:] - 1] = A1[argsMax[:-1]]

    A = A + Amin - 1
    cumMaxFull = cumMaxFull + Amin - 1
    return cumMaxFull

def firstAwithBafterC(A, B, C):
    #Assume C sorted
    #C_argsort = np.argsort(C)
    #C = C[C_argsort]
    B_argsort = np.argsort(B)
    B = B[B_argsort]
    A = A[B_argsort]

    #Any index in indexOfFirstBafterC after indexOfFirstBafterC[x] is an index with B[indexOfFirstBafterC[i]] after C[x]
    #A = A[indexOfFirstBafterC]
    #We need to find smallest A after each val.
    A = -A[-1::-1]
    #Now this is largest A before val.
    A = stefan_cummax(A)
    A = -A[-1::-1] #revert

    #Assume B and C sorted
    indexOfFirstBafterC = stefan_nextAfter(C, B, withCut=True)
    #indexOfFirstBafterC = stefan_nextAfter((C * 2) + 1, B * 2, withCut=True)

    A = A[indexOfFirstBafterC]
    #indexOfFirstBafterC = B_argsort[indexOfFirstBafterC]

    return A

def firstAwithBafterC2(A, B, C):
    #Assume C sorted
    #C_argsort = np.argsort(C)
    #C = C[C_argsort]
    A1 = np.copy(A)
    B_argsort = np.argsort(B)
    B = B[B_argsort]
    A = A[B_argsort]

    #Any index in indexOfFirstBafterC after indexOfFirstBafterC[x] is an index with B[indexOfFirstBafterC[i]] after C[x]
    #A = A[indexOfFirstBafterC]
    #We need to find smallest A after each val.
    A = -A[-1::-1]
    #Now this is largest A before val.
    A, index1 = stefan_cummaxArg(A)
    index1 = A.shape[0] - 1 - index1
    A = -A[-1::-1] #revert
    index1 = index1[-1::-1]

    #Assume B and C sorted
    indexOfFirstBafterC = stefan_nextAfter(C, B, withCut=True)
    #indexOfFirstBafterC = stefan_nextAfter((C * 2) + 1, B * 2, withCut=True)

    #A = A[indexOfFirstBafterC]
    #indexOfFirstBafterC = B_argsort[indexOfFirstBafterC]

    indexOfFirstBafterC = B_argsort[index1[indexOfFirstBafterC]]
    A = A1[indexOfFirstBafterC]

    return A, indexOfFirstBafterC

def firstNAwithBafterC(A, B, C, n):
    B_argsort = np.argsort(B)
    B = B[B_argsort]
    A = A[B_argsort]

    A = -A[-1::-1]

    #Now this is largest A before val.
    A = stefan_cumMaxN(A, n)
    A = -A[-1::-1] #revert
    A[1-n:] = -1

    #Assume B and C sorted
    indexOfFirstBafterC = stefan_nextAfter(C, B, withCut=True)
    A = A[indexOfFirstBafterC]
    A = A[A[:, -1] != -1]

    return A

def fastAllArgwhere(ar):
    ar_argsort = np.argsort(ar)
    ar1 = ar[ar_argsort]
    _, indicesStart = np.unique(ar1, return_index=True)
    _, indicesEnd = np.unique(ar1[-1::-1], return_index=True) #This is probably needless and can be found from indicesStart
    indicesEnd = ar1.shape[0] - indicesEnd - 1
    return ar_argsort, indicesStart, indicesEnd

def stefan_isinCorrespondence(ar1, ar2):
    #want index of intersection in ar1, and corresponding index in ar2
    #ar2_argsort = np.argsort(ar2)
    #ar2 = ar2[ar2_argsort]
    in_ar2 = np.argwhere(np.isin(ar1, ar2))[:, 0]
    ar1 = ar1[in_ar2]
    ar1_unique, ar1_inverse = np.unique(ar1, return_inverse=True)
    tup1 = np.intersect1d(ar1_unique, ar2, return_indices=True)
    indexIn1 = tup1[1]
    indexIn2 = tup1[2]
    indexIn1_inverted = np.zeros(indexIn1.shape[0]).astype(int)
    indexIn1_inverted[indexIn1] = np.arange(indexIn1.shape[0]).astype(int)
    output2 = indexIn2[indexIn1_inverted[ar1_inverse]]
    output1 = in_ar2
    return output1, output2

def stefan_squeeze(ar):
    #Not needed anymore
    if ar.shape[0] > 1:
        return np.squeeze(ar).astype(int)
    elif ar.shape[0] == 1:
        return ar[0].astype(int)
    elif ar.shape[0] == 0:
        return np.array([]).astype(int)

def stefan_firstCorrespondence(ar1, ar2):
    #note, ar1 = orders[argsTransact]
    ar1, ar1_inverse = np.unique(ar1, return_inverse=True)
    tup1 = np.intersect1d(ar1, ar2, assume_unique=False, return_indices=True)
    transOrderSortedArgs = tup1[1]
    startArgs = tup1[2]
    #transOrderSortArgs = np.argsort(argsTransact[transOrderSortedArgs])
    #startArgs = startArgs[transOrderSortArgs]
    #note, ar1 = ar1[ar1_inverse]
    startArgs = startArgs[ar1_inverse]
    return startArgs

def forcedTransationProfit(transArgsSell, startArgsSell, transArgsBuy, transAfterBuy, times, transProfitData):

    usedTransTimes = times[transAfterBuy]
    sellTransTimes = times[transArgsSell]
    timesIntersect = np.intersect1d(usedTransTimes, sellTransTimes)
    repeatingIndividual = np.squeeze(np.argwhere(np.isin(usedTransTimes, timesIntersect)))
    repeatArgs = np.argwhere(np.isin(sellTransTimes, timesIntersect))[:, 0]

    sellTransTimes = sellTransTimes[repeatArgs]
    usedTransTimes = usedTransTimes[repeatingIndividual]
    transArgsSell, startArgsSell, transArgsBuy, transAfterBuy = transArgsSell[repeatArgs], startArgsSell[repeatArgs], transArgsBuy[repeatingIndividual], transAfterBuy[repeatingIndividual]

    validTransArgs = transProfitData[:, 0]
    profits = transProfitData[:, 1]
    sellCorresponding1, sellCorresponding2 = stefan_isinCorrespondence(transArgsSell, validTransArgs)
    sellProfits = np.zeros(transArgsSell.shape[0])
    sellProfits[sellCorresponding1] = profits[sellCorresponding2]

    ar_argsort, indicesStart, indicesEnd = fastAllArgwhere(sellTransTimes)

    argsSingleAtTime = np.argwhere((indicesEnd - indicesStart) == 0)[:, 0]
    argsMultipleAtTime = np.argwhere((indicesEnd - indicesStart) != 0)[:, 0]
    areMultiple1 = np.ones(ar_argsort.shape[0])
    areMultiple1[indicesStart[argsSingleAtTime]] = 0
    areMultiple1 = np.argwhere(areMultiple1 == 1)[:, 0]
    ar_argsort = ar_argsort[areMultiple1]
    validTimes = np.unique(sellTransTimes[ar_argsort])

    #print (ar_argsort)
    #quit()

    indicesLength = indicesEnd - indicesStart + 1
    indicesLength = indicesLength[argsMultipleAtTime]
    indicesEnd = np.cumsum(indicesLength) - 1
    indicesStart = np.zeros(argsMultipleAtTime.shape[0]).astype(int)
    indicesStart[1:] = indicesEnd[:-1] + 1
    #Note, validTimes = sellTransTimes[ar_argsort[indicesStart]]
    #print (np.mean(np.abs(sellTransTimes[ar_argsort[indicesStart]] - validTimes)))

    #validBuy = np.argwhere(np.isin(usedTransTimes, validTimes))[:, 0]
    validBuy, StartIndexArgs = stefan_isinCorrespondence(usedTransTimes, validTimes)

    #print (transArgsBuy.shape[0], np.max(validBuy))

    StartIndexArgs_argsort = np.argsort(StartIndexArgs)
    validBuy, StartIndexArgs = validBuy[StartIndexArgs_argsort], StartIndexArgs[StartIndexArgs_argsort]


    #print (usedTransTimes[validBuy][0::20][:10])
    #print (validTimes[StartIndexArgs][0::20][:10])
    #quit()

    #usedTransTimes, transArgsBuy = usedTransTimes[validBuy], transArgsBuy[validBuy]

    indicesLength, indicesStart, indicesEnd = indicesLength[StartIndexArgs], indicesStart[StartIndexArgs], indicesEnd[StartIndexArgs]
    #ar_argsort_new = np.zeros(np.sum(indicesLength)).astype(int)
    ar_argsort_new_derivative = np.ones(np.sum(indicesLength)).astype(int)
    positions = np.cumsum(indicesLength)[:-1]
    ar_argsort_new_derivative[0] = indicesStart[0]
    ar_argsort_new_derivative[positions] = indicesStart[1:] - indicesEnd[:-1]
    ar_argsort_newArgs = np.cumsum(ar_argsort_new_derivative)

    ar_argsort = ar_argsort[ar_argsort_newArgs]
    indicesEnd = np.cumsum(indicesLength) - 1
    indicesStart = np.zeros(indicesLength.shape[0]).astype(int)
    indicesStart[1:] = indicesEnd[:-1] + 1

    #print (transArgsBuy.shape[0], np.max(validBuy))

    startArgsSell_ar = startArgsSell[ar_argsort]
    transArgsSell_ar = transArgsSell[ar_argsort]
    sellProfits_ar = sellProfits[ar_argsort]
    #print (np.mean(sellProfits_ar))

    validBuy2 = np.zeros(np.sum(indicesLength)).astype(int)
    #print (np.cumsum(indicesLength[:-1])[:10])
    validBuy2[np.cumsum(indicesLength[:-1]).astype(int)] = 1
    validBuy2 = np.cumsum(validBuy2).astype(int)
    #print (np.mean(np.abs(np.unique(validBuy2) - np.arange(np.unique(validBuy2).shape[0]))))
    #quit()
    validBuy2 = validBuy[validBuy2]

    transArgsBuy[validBuy] #Doesn't work hmm

    transArgsBuy_ar = transArgsBuy[validBuy2]
    transAfterBuy_ar = transAfterBuy[validBuy2]


    #print (validBuy2[validArgs][:10])
    #print (transArgsBuy_ar[validArgs][:10])
    #print (validBuy2[:10])
    #print (transArgsBuy_ar[:10])
    #print (transAfterBuy_ar[:10])
    #print (transArgsSell_ar[:10])
    #quit()
    #print ()

    validAr = np.zeros(ar_argsort.shape[0])
    #print (startArgsSell_ar[:40])
    #print (transArgsBuy_ar[:40])
    #print (transAfterBuy_ar[20:40]) #this and the next line should be overlapping in a way that they are not.
    #print (transArgsSell_ar[20:40])
    #print (times[transAfterBuy_ar[:20]]) #This line and the next line are not the same but they should be.
    #print (times[transArgsSell_ar[:20]])
    #quit()

    validAr[np.logical_and((startArgsSell_ar - transArgsBuy_ar) >= 0, (transArgsSell_ar - transAfterBuy_ar) > 0)] = 1 #Used to be "!= 0" but "> 0" allows for initialy buying multiple shares
    #validAr[np.logical_and((startArgsSell_ar - transArgsBuy_ar) >= 0, (transArgsSell_ar - transAfterBuy_ar) == 0)] = 1


    #print (startArgsSell_ar[validAr==1][:40])
    #print (transArgsBuy_ar[validAr==1][:40])
    #print (times[transAfterBuy_ar[validAr==1][:20]]) #This line and the next line are not the same but they should be.
    #print (times[transArgsSell_ar[validAr==1][:20]])
    #quit()
    validArgs = np.argwhere(validAr == 1)[:, 0]

    profitCumSum = np.cumsum(sellProfits_ar * validAr)
    #profitCumSum = np.cumsum(validAr)

    #plt.plot(profitCumSum)
    #plt.show()

    #print (indicesEnd.shape)
    #quit()
    profitCumSum = profitCumSum[indicesEnd]
    finalProfits = profitCumSum
    #print (profitCumSum[:20])
    finalProfits[1:] = profitCumSum[1:] - profitCumSum[:-1]
    #print (finalProfits[:20])

    #ar = np.sort(finalProfits[finalProfits!=0])
    #plt.scatter(ar, np.cumsum(ar))
    #plt.show()

    #plt.hist(finalProfits[finalProfits!=0], bins=100)
    #plt.show()

    #quit()

    validBuy = repeatingIndividual[validBuy]
    return finalProfits, validBuy

def fullForcedTransaction(isBuy, name, loopNumber, priceCutOff, transList, argsTransact, argsIn, miliseconds):
    if isBuy:
        #Yargs = loadnpz('./resultData/Recur1_tempXY2/Yargs_Bid_' + name + '.npz')
        #output = loadnpz('./resultData/outputPrediction3/Bid_' + name + '.npz')
        #finalData = loadnpz('./resultData/Recur1_tempXY2/X_Bid_' + name + '.npz')

        #argsGood_2 = loadnpz('./resultData/temporaryArgsGood/Bid_' + name + ".npz")
        #argsGood_2_sub =  loadnpz('./resultData/Recur1_tempXY2/argsGood_Bid_' + name + '.npz')
        #argsGood_2 = argsGood_2[argsGood_2_sub]
        #argsGood_1 = loadnpz('./inputData/Processed/inputXY/argsGood_Bid_' + name + '.npz')
        oType = "B"
    else:
        #Yargs = loadnpz('./resultData/Recur1_tempXY2/Yargs_Ask_' + name + '.npz')
        #output = loadnpz('./resultData/outputPrediction3/Ask_' + name + '.npz')
        #finalData = loadnpz('./resultData/Recur1_tempXY2/X_Ask_' + name + '.npz')
        oType = "S"

        #argsGood_1 = loadnpz('./inputData/Processed/inputXY/argsGood_Ask_' + name + '.npz')

    argsGood_1 = loadnpz('./recursive/0/orderData/withCancels/argsGood_' + oType + '_' + name + '.npz')

    output = loadnpz('./recursive/' + str(loopNumber) + '/outputPredictions/' + oType + '_' + name + '.npz')
    output[:, 1][output[:, 0] < 0.5] = -2.0
    output = output[:, 1]

    #YData = loadnpz('./recursive/' + str(loopNumber) + '/orderData/withCancels/Y_' + oType + '_' + name + '.npz')
    Yargs = loadnpz('./recursive/' + str(loopNumber) + '/orderData/withCancels/Trans_' + oType + '_' + name + '.npz')
    finalData = loadnpz('./recursive/' + str(loopNumber) + '/orderData/withCancels/X_' + oType + '_' + name + '.npz')


    #argsGood_2 = loadnpz('./resultData/temporaryArgsGood/Ask_' + name + ".npz")
    #argsGood_2 = loadnpz('./recursive/' + str(loopNumber) + '/orderData/initial/argsGood_' + oType + '_' + name + '.npz')
    #argsGood_2_sub =  loadnpz('./resultData/Recur1_tempXY2/argsGood_Ask_' + name + '.npz')
    argsGood_2 = loadnpz('./recursive/' + str(loopNumber) + '/orderData/withCancels/argsGood_' + oType + '_' + name + '.npz')
    #argsGood_2 = argsGood_2[argsGood_2_sub]

    reverser = (np.zeros(max(np.max(argsGood_1), np.max(argsGood_2)) + 1 ) - 1).astype(int)
    reverser[argsGood_1] = np.arange(argsGood_1.shape[0])
    subArgs = reverser[argsGood_2]
    output = output[subArgs]

    depth = finalData[:, 2]
    Yargs = Yargs[depth == 0]
    output = output[depth == 0]

    #plt.hist(output, bins=100)
    #plt.show()
    #quit()

    #priceCutOff = -0.77 #TODO add if recursiveCancels to here
    #finalData = finalData[output > priceCutOff]
    Yargs = Yargs[output > priceCutOff]
    output = output[output > priceCutOff]


    transList2 = np.copy(transList)
    subset3 = np.array([]).astype(int)
    for a in range(int(np.min(transList[:, 2])), int(np.max(transList[:, 2]) + 1)):
        subset2 = np.argwhere(transList2[:, 2] == a)[:, 0]
        subset2 = subset2[np.argsort(transList2[subset2, 0])]
        start1, end1 = transList2[subset2, 0], transList2[subset2, 1]
        before1 = stefan_firstBefore(end1[:-1], start1)
        after1 = np.arange(start1.shape[0]-1) + 1
        diff1 = before1 - after1
        #plt.plot(end1)
        #plt.show()
        #quit()
        #plt.plot(before1)
        #plt.plot(after1)
        #plt.plot(end1)
        #plt.show()
        #quit()
        after1 = after1[diff1 > 0]
        before1 = before1[diff1 > 0]
        ar1 = np.zeros(subset2.shape[0] + 1)

        after2, after2_counts = np.unique(after1, return_counts=True)
        before2, before2_counts = np.unique(before1, return_counts=True)

        ar1[after2] = after2_counts
        ar1[before2+1] = before2_counts * -1
        ar1 = np.cumsum(ar1)[:-1]
        #plt.plot(ar1)
        #plt.show()
        #quit()
        #if np.sum(ar1) != 0:
        subset2 = subset2[ar1 == 0].astype(int)
        subset3 = np.concatenate((subset3, subset2))
    #print (transList2.shape)
    transList2 = transList2[subset3]
    #print (transList2.shape)
    #quit()





    #print (transList[:, 1].shape)
    #print (transList[:, 1].shape, transList[:, 0].shape, argsTransact.shape, transList[:, 1][argsIn].shape, np.array([Yargs, output]).T.shape)

    finalProfits, validBuy = forcedTransationProfit(transList2[:, 1], transList2[:, 0], argsTransact, transList[:, 1][argsIn], miliseconds, np.array([Yargs, output]).T )


    #finalProfits[finalProfits < -1] = -1
    #print (np.mean(finalProfits))
    #print (np.mean(finalProfits[finalProfits != 0]))
    #quit()
    return finalProfits, validBuy

def saveAllTransactionProfits(loopNumber, filesRun=(0, 15)):


    def saveComponent(nameNum):

        names = giveFNames()
        name = names[nameNum]
        priceBoth = np.array([])
        validBoth = np.array([])
        forcedBoth = np.array([])

        print (name)

        for isBuy in [True, False]:#[True, False]:#[True, False]:
            if isBuy:
                oType = 'B'
            else:
                oType = 'S'

            #if loopNumber == 0:
            #    if isBuy:
            #        transList = loadnpz('./inputData/ITCH_LOB/reformedTransTimes/Bid_' + name + ".npz")
            #    else:
            #        transList = loadnpz('./inputData/ITCH_LOB/reformedTransTimes/Ask_' + name + ".npz")
            #else:
                #transList = loadnpz('./recursive/' + str(loopNumber) + '/transProfits/initial/' + "test_" + oType + '_' + name + '.npz')

            #transList = loadnpz('./recursive/' + str(loopNumber) + '/transProfits/initial/' + oType + '_' + name + '.npz')

            ###
            Yargs = loadnpz('./recursive/' + str(loopNumber) + '/orderData/withCancels/Trans_' + oType + '_' + name + '.npz')
            finalData = loadnpz('./recursive/' + str(loopNumber) + '/orderData/withCancels/X_' + oType + '_' + name + '.npz')
            YData = loadnpz('./recursive/' + str(loopNumber) + '/orderData/withCancels/Y_' + oType + '_' + name + '.npz')
            #outputCancels = loadnpz('./recursive/' + str(loopNumber) + '/outputCancels/' + oType + '_' + name + '.npz')
            #Yargs = Yargs[outputCancels == 0]
            #finalData = finalData[outputCancels == 0]
            #YData = YData[outputCancels == 0]

            Yargs = Yargs[finalData[:, 2] == 0]
            YData = YData[finalData[:, 2] == 0]
            finalData = finalData[finalData[:, 2] == 0]


            #Yargs = Yargs[finalData[:, 1] != -1]
            #YData = YData[finalData[:, 1] != -1]
            #finalData = finalData[finalData[:, 1] != -1]

            #print (np.unique(finalData[:, 1]))
            #quit()

            position = finalData[:, 0]
            price = YData[:, 1]
            transList = np.array([position, Yargs, price]).T





            data = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz')
            miliseconds = data[:, 0].astype(int)
            argsTransact = np.argwhere(np.isin(data[:, 3], np.array(['E', 'F']) ))[:, 0]
            #argsTransact = argsTransact[data[argsTransact, 4] == oType]
            argsTransact = argsTransact[data[argsTransact, 4] != oType]

            pricePriority = transList[:, 2]

            if isBuy: #Jun 21 removed not
                pricePriority = np.max(transList[:, 2]) - transList[:, 2]
            pricePriority[pricePriority >= 100000] = 100000 - 1 #This shouldn't have to be used I beleive, but exists just incase

            prioritiyValue = (miliseconds[transList[:, 1]] * 100000) + pricePriority
            #prioritiyValue = (transList[:, 1] * 100000) + pricePriority
            prioritiyValue = transList[:, 1]
            # note, argsTransact = buys

            _, argsIn = firstAwithBafterC2(prioritiyValue, transList[:, 0], argsTransact)
            argsTransact = argsTransact[:argsIn.shape[0]]

            #plt.plot(argsTransact)
            #plt.plot(transList[:, 0][argsIn])
            #plt.plot(transList[:, 1][argsIn])
            #plt.scatter(transList[:, 0][0::1000], transList[:, 1][0::1000])
            #plt.plot(transList[:, 1][argsIn] - argsTransact)
            #plt.show()
            #quit()


            ###
            #finalData = loadnpz('./recursive/' + str(loopNumber) + '/orderData/withCancels/X_' + oType + '_' + name + '.npz')
            #YData = loadnpz('./recursive/' + str(loopNumber) + '/orderData/withCancels/Y_' + oType + '_' + name + '.npz')
            #depth = finalData[:, 2]
            #YData = YData[depth == 0]
            #finalData = finalData[depth == 0]

            #print (finalData[argsTransact][10000:10000 + 10])
            #quit()
            #'''

            #if isBuy:
            #    data = data[data[:, 4] == "B"]
            #else:
            #    data = data[data[:, 4] == "S"]
            #price = data[:, 2].astype(int)
            #priceUnique = np.unique(price)
            #priceVal = priceUnique[transList[:, 2][argsIn]]
            #print (np.mean(priceVal))
            #quit()

            #for a in range(0, 2):
            #    print (np.max(argsIn))
            #    print (finalData.shape)
            #    plt.hist(finalData[:, a], bins=100, density=True)
            #    plt.hist(finalData[argsIn, a], bins=100, density=True)
            #    plt.show()
            #quit()
            #'''





            #finalProfits, validBuy = fullForcedTransaction(isBuy, name, loopNumber, priceCutOff, transList, argsTransact, argsIn, miliseconds)
            if True:
                size1 = data.shape[0]
                if isBuy:
                    data = data[data[:, 4] == "B"]
                    bestPrice0 = loadnpz('./inputData/ITCH_LOB/bestPrice/Bid_' + name + '.npz')
                else:
                    data = data[data[:, 4] == "S"]
                    bestPrice0 = loadnpz('./inputData/ITCH_LOB/bestPrice/Ask_' + name + '.npz')

                price = data[:, 2].astype(int)
                priceUnique = np.unique(price)

                #data1 = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz')
                #data1 = data1[data1[:, 4] == "B"]
                #priceUniqueBid = np.unique(data1[:, 2].astype(int))
                #data1 = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz')
                #data1 = data1[data1[:, 4] == "S"]
                #priceUniqueAsk = np.unique(data1[:, 2].astype(int))



                priceVal = priceUnique[transList[:, 2][argsIn]]
                #priceVal = transList[:, 2][argsIn]

                forcedVal = np.zeros(priceVal.shape)
                #if isBuy:
                    #priceVal[validBuy] = priceVal[validBuy] - finalProfits
                    #forcedVal[validBuy] = -1 * finalProfits
                #else:
                    #priceVal[validBuy] = priceVal[validBuy] + finalProfits
                    #forcedVal[validBuy] = finalProfits
                #priceVal = transList[:, 2][argsIn]

                #print (priceVal[1000:1010])
                #print (np.mean(priceVal))
                #quit()

                x1 = transList[:, 1][argsIn]

                priceVal = priceVal[x1 < (size1 - 1000)]
                argsTransact = argsTransact[x1 < (size1 - 1000)]
                forcedVal = forcedVal[x1 < (size1 - 1000)]
                x1 = x1[x1 < (size1 - 1000)]

                #x0 = x0[x1 < (size1 - 1000)]

                priceVal = priceVal[argsTransact > 1000]
                forcedVal = forcedVal[argsTransact > 1000]
                x1 = x1[argsTransact > 1000]
                argsTransact = argsTransact[argsTransact > 1000]
                #x1 = x1[x0 > 1000]

                size2 = x1.shape[0]#[-size2//5:]
                #plt.plot(argsTransact[np.argsort(argsTransact)], x1[np.argsort(argsTransact)])
                #plt.plot(argsTransact[np.argsort(argsTransact)], (x1 - argsTransact)[np.argsort(argsTransact)])
                #plt.show()



                #bestBid = loadnpz('./inputData/ITCH_LOB/bestPrice/Bid_' + name + '.npz')
                #bestAsk = loadnpz('./inputData/ITCH_LOB/bestPrice/Ask_' + name + '.npz')
                #if isBuy:
                    #print (np.mean(priceUniqueBid[bestBid[x1]] - priceVal))
                    #print (np.mean(priceUniqueAsk[bestAsk[x1]] - priceVal))
                    #prof1 = np.mean(priceUniqueBid[bestBid[argsTransact]] - priceVal)
                    #print (prof1)

                    #print (np.mean(priceUniqueBid[bestBid[argsTransact]] - priceVal))
                    #print (np.mean(priceUniqueAsk[bestAsk[argsTransact]] - priceVal))
                    #print (np.mean(priceUniqueAsk[bestAsk[argsTransact]] - priceUniqueBid[bestBid[argsTransact]]))

                    #plt.plot(bestBid[argsTransact])
                    #plt.plot(bestAsk[argsTransact])
                    #plt.plot(priceVal)
                    #plt.show()
                #else:
                    #print (np.mean(priceUniqueBid[bestBid[x1]] - priceVal))
                    #print (np.mean(priceUniqueAsk[bestAsk[x1]] - priceVal))
                    #prof2 = np.mean(priceVal - priceUniqueAsk[bestAsk[argsTransact]])
                    #print (prof2)

                    #print (np.mean(priceVal - priceUniqueAsk[bestAsk[argsTransact]]))
                    #print (np.mean(priceVal - priceUniqueBid[bestBid[argsTransact]]))
                    #print (np.mean(priceUniqueAsk[bestAsk[argsTransact]] - priceUniqueBid[bestBid[argsTransact]]))

                transProfitData = np.zeros((argsTransact.shape[0], 3))
                transProfitData[:, 0] = argsTransact
                transProfitData[:, 1] = priceVal
                transProfitData[:, 2] = forcedVal
                argsValidSort = np.argsort(argsTransact)
                transProfitData = transProfitData[argsValidSort]

                #Intentionaly reversed
                if isBuy:
                    oType = 'S'
                else:
                    oType = 'B'

                if True:
                    np.save('./recursive/' + str(loopNumber) + '/transProfits/final/transSalePrices_' + oType + '_' +  name + '.npy', transProfitData)
                    reverse = np.zeros(int(np.max(argsTransact)+1)).astype(int) - 1
                    reverse[transProfitData[:, 0].astype(int)] = np.arange(transProfitData.shape[0])
                    np.save('./recursive/' + str(loopNumber) + '/transProfits/final/ReverseTrans_' + oType + '_' + name + '.npy', reverse)


    #for a in range(0, 15):
    #    saveComponent(a)
    #saveComponent(0)
    #quit()
    #if filesRun == -1:
    #    filesRun = 15
    multiProcessor(saveComponent, filesRun[0], filesRun[1], 4, doPrint = True)



#saveAllTransactionProfits(1)
#quit()

def transTimesConverter():

    names = giveFNames()

    for name in names: #3
        print (name)
        for isBuy in [True, False]:
            if isBuy:
                #orderTransTimes = loadnpz('./inputData/ITCH_LOB/orderTransTimes/Bid_' + name + ".npz", allow_pickle = True)
                LOBhistory = loadnpz('./inputData/ITCH_LOB/BuyLOB/' + name + '.npz', allow_pickle=True)
                oType = 'B'
            else:
                #orderTransTimes = loadnpz('./inputData/ITCH_LOB/orderTransTimes/Ask_' + name + ".npz", allow_pickle = True)
                LOBhistory = loadnpz('./inputData/ITCH_LOB/SellLOB/' + name + '.npz', allow_pickle=True)
                oType = 'S'

            orderTransTimes = loadnpz('./recursive/' + str(0) + '/orderTransTimes/' + oType + '_' + name + ".npz", allow_pickle=True)

            #data = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz', allow_pickle=True)
            #argsBuy = np.argwhere(data[:, 4] == "B")[:, 0]
            #bestPrice0 = loadnpz('./inputData/ITCH_LOB/bestPrice/Bid_' + name + '.npz')
            #print (bestPrice0[LOBhistory[820][0][469]])
            #820 469
            #quit()
            #b1 = 469
            #print (bestPrice0[argsBuy[int(orderTransTimes[820][b1][-1]) ]])
            #quit()


            transList = np.zeros((10000000, 3)).astype(int)
            count = 0

            for a in range(0, len(orderTransTimes)):
                args = LOBhistory[a][0]
                #print ("L")
                #print (len(orderTransTimes[a]))
                #print (len(args))
                #quit()
                for b in range(0, len(orderTransTimes[a])):
                    b1 = len(orderTransTimes[a]) - b - 1
                    if orderTransTimes[a][b1][-1] != -1:
                        #if orderTransTimes[a][b1][-1] < args[b]:
                        #    print ("A")
                            #print (orderTransTimes[a][b][-1], args[b], args[b-1])
                        #    quit()
                        #if args[b] == 427032:
                            #print (a, b1, b)
                            #print (bestPrice0[argsBuy[int(orderTransTimes[a][b1][-1]) ]])
                            #quit()
                        transList[count] = np.array([args[b], orderTransTimes[a][b1][-1], a])
                        count += 1

            transList = transList[:count]

            data = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz', allow_pickle=True)
            argsBuy = np.argwhere(data[:, 4] == oType)[:, 0]
            del data
            #transList[:, 2] = priceUnique[transList[:, 2]]

            #print (transList[:, 0][argsBuy[transList[:, 0]] == 844214])
            #quit()

            transList[:, 0] = argsBuy[transList[:, 0]]
            transList[:, 1] = argsBuy[transList[:, 1]]
            #plt.plot(transList[:, 0] - transList[:, 1])
            #plt.show()
            #print (transList[np.argsort(transList[:, 0])][:10])
            #quit()

            transList = transList.astype(int)

            if isBuy:
                bestPrice0 = loadnpz('./inputData/ITCH_LOB/bestPrice/Bid_' + name + '.npz')
            else:
                bestPrice0 = loadnpz('./inputData/ITCH_LOB/bestPrice/Ask_' + name + '.npz')

            bestPrice = bestPrice0[transList[:, 0]]
            priceDiff = transList[:, 2] - bestPrice
            if isBuy:
                priceDiff = priceDiff * -1

            #plt.hist(priceDiff, bins=100)
            #plt.show()

            #print (transList.shape)
            transList = transList[priceDiff >= 0]
            #print (transList.shape)


            #Todo remove
            '''
            bestPrice = bestPrice0[transList[:, 1]] #1
            priceDiff = transList[:, 2] - bestPrice

            #print (transList[:, 0][priceDiff < -10])
            argsBelow = transList[:, 1][priceDiff < -10]
            if argsBelow.shape[0] > 0:
                print (np.min(np.array([argsBelow, bestPrice0.shape[0] - argsBelow]), axis=0))
                argsBelow = np.min(np.array([argsBelow, bestPrice0.shape[0] - argsBelow]), axis=0) / bestPrice0.shape[0]
                print (np.max(argsBelow))
            else:
                print (0.0)
            #quit()
            #plt.hist(priceDiff, bins=100)
            #plt.plot(bestPrice0[2000:-1000])
            #plt.plot(bestPrice0[2000:-1000][150000:170000]) #- np.mean(bestPrice0)) #2323609
            plt.plot(np.arange(bestPrice0.shape[0])[159600:160000], bestPrice0[159600:160000])
            transList = transList[transList[:, 1] > 159600]
            transList = transList[transList[:, 1] < 160000]
            #plt.scatter(transList[:, 1][np.argsort(transList[:, 1])], priceDiff[np.argsort(transList[:, 1])])
            plt.scatter(transList[:, 1][np.argsort(transList[:, 1])], transList[:, 2][np.argsort(transList[:, 1])])
            plt.show()
            quit()
            '''

            #print (transList.shape)
            #quit()
            if isBuy:
                np.savez_compressed('./inputData/ITCH_LOB/reformedTransTimes/Bid_' + name + ".npz", transList)
            else:
                np.savez_compressed('./inputData/ITCH_LOB/reformedTransTimes/Ask_' + name + ".npz", transList)

def findValidSells(loopNumber, filesRun=(0, 15)):

    def saveComponent(nameNum):
        names = giveFNames()
        name = names[nameNum]
        print (name)
        for isBuy in [True, False]:
            '''
            if isBuy:
                LOB_share = loadnpz('./inputData/ITCH_LOB/BuyLOB/' + name + '.npz', allow_pickle=True)
                oType = "B"
                #bestPrice0 = loadnpz('./inputData/ITCH_LOB/bestPrice/Bid_' + name + '.npz')
            else:
                LOB_share = loadnpz('./inputData/ITCH_LOB/SellLOB/' + name + '.npz', allow_pickle=True)
                oType = "S"
                #bestPrice0 = loadnpz('./inputData/ITCH_LOB/bestPrice/Ask_' + name + '.npz')

            argsGood_1 = loadnpz('./recursive/0/orderData/withCancels/argsGood_' + oType + '_' + name + '.npz')

            if loopNumber > 0:
                output = loadnpz('./recursive/' + str(loopNumber) + '/outputCancels/' + oType + '_' + name + '.npz')

            argsGood_2 = loadnpz('./recursive/' + str(loopNumber) + '/orderData/withCancels/argsGood_' + oType + '_' + name + '.npz')

            reverser = (np.zeros(max(np.max(argsGood_1), np.max(argsGood_2)) + 1 ) - 1).astype(int)
            reverser[argsGood_1] = np.arange(argsGood_1.shape[0])
            subArgs = reverser[argsGood_2]
            '''

            YData = loadnpz('./recursive/' + str(loopNumber) + '/orderData/withCancels/Y_' + oType + '_' + name + '.npz')
            Yargs = loadnpz('./recursive/' + str(loopNumber) + '/orderData/withCancels/Trans_' + oType + '_' + name + '.npz')
            finalData = loadnpz('./recursive/' + str(loopNumber) + '/orderData/withCancels/X_' + oType + '_' + name + '.npz')
            #if loopNumber > 0:
            #    output = output[subArgs]

            #print (np.mean(output))
            #print (finalData[output == 1])
            #print (YData[output == 1])
            #quit()

            #orderTransTimes = loadnpz('./recursive/' + str(loopNumber) + '/orderTransTimes/' + oType + '_' + name + ".npz", allow_pickle=True)

            #data = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz', allow_pickle=True)
            #argsBuy = np.argwhere(data[:, 4] == oType)[:, 0]
            #del data

            depth = finalData[:, 2]
            YData = YData[depth == 0]
            finalData = finalData[depth == 0]
            #if loopNumber > 0:
            #    output = output[depth == 0]
            Yargs = Yargs[depth == 0]

            #if loopNumber > 0:
            #    finalData = finalData[output == 0]
            #    YData = YData[output == 0]
            #    Yargs = Yargs[output == 0]
            #    output = output[output == 0]

            lobPos = YData[:, 0]
            position = finalData[:, 0]
            price = YData[:, 1]
            quePos = YData[:, 2]

            transList = np.array([position, Yargs, price]).T
            print ("U")
            if True:
                np.savez_compressed('./recursive/' + str(loopNumber) + '/transProfits/initial/' + oType + '_' + name + '.npz', transList)

    #if loopNumber > 0:
        #saveComponent(5)
        #quit()
        #saveComponent(0)
        #quit()
        #if filesRun == -1:
        #    filesRun = 15
    #saveComponent(14)
    #quit()
    multiProcessor(saveComponent, filesRun[0], filesRun[1], 4, doPrint = True)


#findValidSells(1)
#quit()

def individualLOB(isBuy):
    import copy
    names = giveFNames()
    for name in names:
        print (name)
        #name = '20200305'
        data = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz', allow_pickle=True)
        if isBuy:
            data = data[data[:, 4] == "B"]
        else:
            data = data[data[:, 4] == "S"]
        type = data[:, 3]
        price = data[:, 2].astype(int)
        share = data[:, 1].astype(int)
        orderNum = data[:, 6].astype(int)

        priceUnique, price_inverse = np.unique(price, return_inverse=True)

        LOBhistory = []
        LOBcurrent = []
        for a in range(0, priceUnique.shape[0]):
            LOBcurrent.append([])
            LOBhistory.append([[ 0 ], [ [] ]]) #Adjusted Apr5  from [[], []]

        #orderUnique, order_inverse, order_counts = np.unique(orderNum, return_inverse=True, return_counts=True)

        lenSum = 0
        print (data.shape[0] // 10000)
        for a in range(0, data.shape[0]):
            price_arg = price_inverse[a]
            len1 = len(LOBcurrent[price_arg])
            lenSum += len1
            if type[a] in ["B", "S"]:
                LOBcurrent[price_arg].append([orderNum[a], share[a], a]) #LOBcurrent[price_arg].append([orderNum[a], share[a]])
            else:
                orderNow = orderNum[a]
                orderArg = np.argwhere(np.array(LOBcurrent[price_arg])[:, 0] == orderNow)[0][0]
                if type[a] in ["D", "F"]:
                    del LOBcurrent[price_arg][orderArg]
                elif type[a] not in ["T", "X"]:
                    LOBcurrent[price_arg][orderArg][1] = LOBcurrent[price_arg][orderArg][1] - share[a]

            LOBhistory[price_arg][0].append(a)
            LOBhistory[price_arg][1].append(copy.copy(LOBcurrent[price_arg]))
            if a % 100000 == 0:
                print (a//100000)
                print (lenSum / 100000)
                lenSum = 0

        if isBuy:
            #np.save('./resultData/temporary/LOB_Buy.npy', LOBhistory)
            np.savez_compressed('./inputData/ITCH_LOB/BuyLOB/' + name + '.npz', LOBhistory)
        else:
            #np.save('./resultData/temporary/LOB_Sell.npy', LOBhistory)
            np.savez_compressed('./inputData/ITCH_LOB/SellLOB/' + name + '.npz', LOBhistory)


def shareLOB(isBuy):
    names = giveFNames()
    for name in names:
        print (name)

        if isBuy:
            LOBhistory = loadnpz('./inputData/ITCH_LOB/BuyLOB/' + name + '.npz', allow_pickle=True)
        else:
            LOBhistory = loadnpz('./inputData/ITCH_LOB/SellLOB/' + name + '.npz', allow_pickle=True)

        LOB_share = []
        tempArgs = []
        for a in range(0, len(LOBhistory)):
            LOBatPrice = LOBhistory[a]
            ar = np.copy(np.array(LOBhistory[a][0]))

            shareAtPrice = [[0, 0]]
            for b in range(0, len(LOBatPrice[0])):
                #print (np.array(LOBatPrice[1][b]).shape)
                shares1 = np.array(LOBatPrice[1][b])
                if shares1.shape[0] == 0:
                    shareSum = 0
                else:
                    shareSum = np.sum(shares1[:, 1])
                shareAtPrice.append([LOBatPrice[0][b], shareSum])

            shareAtPrice = np.array(shareAtPrice)
            #ar = shareAtPrice[:, 0]

            #tempArgs = tempArgs + list(ar[ar < 1000])
            LOB_share.append(shareAtPrice)
        #print (np.unique(tempArgs))
        #print (len(LOB_share))
        if isBuy:
            np.savez_compressed('./inputData/ITCH_LOB/BuyShareLOB/' + name + '.npz', LOB_share)
        else:
            np.savez_compressed('./inputData/ITCH_LOB/SellShareLOB/' + name + '.npz', LOB_share)

def constructBestPrice(isBuy):
    names = giveFNames()#[:-15]
    for name in names:
        #name = '20200305'
        print (name)
        data = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz')
        if isBuy:
            data = data[data[:, 4] == "B"]
            LOB_share = loadnpz('./inputData/ITCH_LOB/BuyShareLOB/' + name + '.npz', allow_pickle=True)
        else:
            data = data[data[:, 4] == "S"]
            LOB_share = loadnpz('./inputData/ITCH_LOB/SellShareLOB/' + name + '.npz', allow_pickle=True)

        lens1 = []
        oncePerM = np.arange(data.shape[0])
        bidMax = np.zeros(oncePerM.shape[0])
        allShares = []
        #max1 = 0
        for a1 in range(0, len(LOB_share)):
            if isBuy:
                a = a1
            else:
                a = len(LOB_share) - 1 - a1
            #print (a)
            ar = np.array(LOB_share[a])[:, 0]
            argsInAr = stefan_nextAfter(oncePerM+1, ar) - 1
            shares = np.array(LOB_share[a])[argsInAr, 1]
            bidMax[shares != 0] = a

        if isBuy:
            np.savez_compressed('./inputData/ITCH_LOB/BestBid/' + name + '.npz', bidMax)
        else:
            np.savez_compressed('./inputData/ITCH_LOB/BestAsk/' + name + '.npz', bidMax)

def saveNearByShare(isBuy):
    names = giveFNames()#[-15:]
    for name in names:
        print (name)
        #name = '20200305'
        data = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz')
        if isBuy:
            data = data[data[:, 4] == "B"]
            LOB_share = loadnpz('./inputData/ITCH_LOB/BuyShareLOB/' + name + '.npz', allow_pickle=True)
            bestPrice = loadnpz('./inputData/ITCH_LOB/BestBid/' + name + '.npz')
            shift = 20
        else:
            data = data[data[:, 4] == "S"]
            LOB_share = loadnpz('./inputData/ITCH_LOB/SellShareLOB/' + name + '.npz', allow_pickle=True)
            bestPrice = loadnpz('./inputData/ITCH_LOB/BestAsk/' + name + '.npz')
            shift = -20

        oncePerM = np.arange(data.shape[0])
        #oncePerM = np.arange(1000) + (data.shape[0] // 2)
        bestPrice = bestPrice[oncePerM]
        allShares = []
        #max1 = 0
        #rangeSize = abs(shift) * 2
        allShares = np.zeros((oncePerM.shape[0], (np.abs(shift) * 2) + 1))
        for a in range(0, len(LOB_share)): #1879
            if a % 100 == 0:
                print (a)
            #a from bestPrice - 40 to bestPrice
            args = stefan_squeeze(np.argwhere( np.abs(a - bestPrice + shift) <= np.abs(shift) )).astype(int)
            a2 = (a - bestPrice[args] + shift + np.abs(shift)).astype(int)
            ar = np.array(LOB_share[a])[:, 0]
            argsInAr = stefan_nextAfter(args+1, ar) - 1 #TODO correct -1 to mean not existing.
            shares = np.array(LOB_share[a])[argsInAr, 1]
            allShares[args, a2] = shares
            #sharesChange = shares[1:] - shares[:-1]
            #argsChange = argsInAr[1:] - argsInAr[:-1]

        if isBuy:
            np.savez_compressed('./inputData/ITCH_LOB/nearBestBidShare/' + name + '.npz', allShares)
        else:
            np.savez_compressed('./inputData/ITCH_LOB/nearBestAskShare/' + name + '.npz', allShares)

def saveNearByShareFull():
    names = giveFNames()[-15:]
    for name in names:
        #name = '20200305'
        print (name)
        data = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz')
        bidShares = loadnpz('./inputData/ITCH_LOB/nearBestBidShare/' + name + '.npz')
        askShares = loadnpz('./inputData/ITCH_LOB/nearBestAskShare/' + name + '.npz')
        bidArgs = np.arange(data.shape[0])[data[:, 4] == "B"]
        askArgs = np.arange(data.shape[0])[data[:, 4] == "S"]
        argsAll = np.arange(data.shape[0])

        bidBefore = stefan_nextAfter(argsAll+1, bidArgs) - 1
        askBefore = stefan_nextAfter(argsAll+1, askArgs) - 1

        shares = np.zeros((data.shape[0], 82))
        shares[:, :41] = bidShares[bidBefore]
        shares[:, 41:] = askShares[askBefore]

        bidPriceUnique = np.unique(data[bidArgs, 2].astype(int))
        askPriceUnique = np.unique(data[askArgs, 2].astype(int))
        bestBid = loadnpz('./inputData/ITCH_LOB/BestBid/' + name + '.npz')
        bestAsk = loadnpz('./inputData/ITCH_LOB/BestAsk/' + name + '.npz')

        np.savez_compressed('./inputData/ITCH_LOB/bestPrice/Bid_' + name + '.npz', bestBid[bidBefore].astype(int))
        np.savez_compressed('./inputData/ITCH_LOB/bestPrice/Ask_' + name + '.npz', bestAsk[askBefore].astype(int))

        np.savez_compressed('./inputData/ITCH_LOB/bestPrice/Bid0_' + name + '.npz', bidPriceUnique[bestBid[bidBefore].astype(int)].astype(int))
        np.savez_compressed('./inputData/ITCH_LOB/bestPrice/Ask0_' + name + '.npz', askPriceUnique[bestAsk[askBefore].astype(int)].astype(int))

        spread = askPriceUnique[bestAsk[askBefore].astype(int)].astype(int) - bidPriceUnique[bestBid[bidBefore].astype(int)].astype(int)

        np.savez_compressed('./inputData/ITCH_LOB/nearSharesAll/' + name + '.npz', shares)
        np.savez_compressed('./inputData/ITCH_LOB/spread/' + name + '.npz', spread)

def saveLOBhistWithStarts():
    fnames = giveFNames()[5:]#[:-15]
    for name in fnames:
        print (name)
        for isBuy in [True, False]:
            if isBuy:
                LOBhistory = loadnpz('./inputData/ITCH_LOB/BuyLOB/' + name + '.npz', allow_pickle=True)
            else:
                LOBhistory = loadnpz('./inputData/ITCH_LOB/SellLOB/' + name + '.npz', allow_pickle=True)
            LOBhistory2 = []
            for a in range(1183, len(LOBhistory)):

                print (a)
                LOBflat = np.zeros((0, 3))
                #arSizes = [0] #This is so I can remove the last element of cumsum
                arSizes = []
                arArgs = LOBhistory[a][0]
                for b in range(0, len(arArgs)):
                    arSizes.append(len(LOBhistory[a][1][b]))
                    LOBflat = np.concatenate(( LOBflat, np.array(LOBhistory[a][1][b]).reshape(len(LOBhistory[a][1][b]), 3) ))
                LOBhistory2a = [arArgs, arSizes, LOBflat]
                #arSizes = list(np.cumsum(np.array(arSizes))[:-1])
                LOBhistory2.append(LOBhistory2a)

            if isBuy:
                np.savez_compressed('./inputData/ITCH_LOB/BuyFlatLOB/' + name + '.npz', LOBhistory2)
            else:
                np.savez_compressed('./inputData/ITCH_LOB/SellFlatLOB/' + name + '.npz', LOBhistory2)

def correctLOBSizesMini():
    fnames = giveFNames()
    for name in fnames:
        print (name)
        for isBuy in [True, False]:
            if isBuy:
                LOBhistory = loadnpz('./inputData/ITCH_LOB/BuyFlatLOB/' + name + '.npz', allow_pickle=True)
            else:
                LOBhistory = loadnpz('./inputData/ITCH_LOB/SellFlatLOB/' + name + '.npz', allow_pickle=True)
            for a in range(0, len(LOBhistory)):
                LOBhistory[a][1] = LOBhistory[a][1][1:]
            if isBuy:
                np.savez_compressed('./inputData/ITCH_LOB/BuyFlatLOB/' + name + '.npz', LOBhistory)
            else:
                np.savez_compressed('./inputData/ITCH_LOB/SellFlatLOB/' + name + '.npz', LOBhistory)
    quit()

def saveAllLOBRelated():
    individualLOB(True)
    individualLOB(False)
    shareLOB(True)
    shareLOB(False)
    constructBestPrice(True)
    constructBestPrice(False)
    saveNearByShare(True)
    saveNearByShare(False)
    saveNearByShareFull()
    saveLOBhistWithStarts()

#saveAllLOBRelated()
#quit()


def multiArgSort(ar1, ar2):
    _, ar1_ = np.unique(ar1, return_inverse=True)
    ar2u, ar2_ = np.unique(ar2, return_inverse=True)
    ar3 = (ar1_ * ar2u.shape[0]) + ar2_
    argsort1 = np.argsort(ar3)
    return argsort1

def runNow1():
    saveLayerBinfo_1(1)
    quit()
    #saveLayerBinfo_2(1)
    #quit()
    #analyizeDepthConnection()
    #quit()
    analysisOfB()
    quit()
    analysisOfB2()
    quit()

def findTransactPriority(isBuy, name, transactionArgs, shareShift):
    data = np.load('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npy')
    allPrices = data[:, 2].astype(int)
    if isBuy:
        argsBid = np.squeeze(np.argwhere(data[:, 4] == "B"))
        LOB_share = np.load('./inputData/ITCH_LOB/BuyShareLOB/' + name + '.npy', allow_pickle=True)
    else:
        argsBid = np.squeeze(np.argwhere(data[:, 4] == "S"))
        LOB_share = np.load('./inputData/ITCH_LOB/SellShareLOB/' + name + '.npy', allow_pickle=True)
    priceUniqueBid, priceInverseBid = np.unique(allPrices[argsBid], return_inverse=True)
    tup1 = np.intersect1d(argsBid, transactionArgs, return_indices=True)

    buySubArgs = tup1[1]
    priceBid = priceInverseBid[buySubArgs]

    if shareShift > 0:
        if isBuy:
            priceBid = priceBid - shareShift
        else:
            priceBid = priceBid + shareShift
        priceBid[priceBid>=np.max(priceInverseBid)] = np.max(priceInverseBid) - 1
        priceBid[priceBid<0] = 0

    priceBidUnique = np.unique(priceBid)
    ar_argsort, indicesStart, indicesEnd = fastAllArgwhere(priceBid)

    sharesFull = np.zeros(priceBid.shape[0])
    a1 = 0
    notDone = True
    while notDone:
        a = priceBidUnique[a1]
        argsInPoints = np.sort(ar_argsort[indicesStart[a1]:indicesEnd[a1] + 1])

        ar = np.array(LOB_share[a])[:, 0]
        shares = np.array(LOB_share[a])[:, 1]

        if shareShift > 0:
            ar_input = stefan_firstBefore(buySubArgs[argsInPoints]-1, ar)
            shares = shares[ar_input]
        else:
            shares = shares[:-1][np.isin(ar[1:], buySubArgs[argsInPoints])] #1:, :-1 => the number of shares BEFORE the transaction


        #sharesFull = np.concatenate((sharesFull, shares))
        sharesFull[argsInPoints] = shares

        a1 += 1
        if a1 >= priceBidUnique.shape[0]:
            notDone = False
        elif priceBidUnique[a1] >= len(LOB_share):
            notDone = False



    argsInTrans = tup1[2]
    return sharesFull, argsInTrans
    #return "banana", argsInTrans
    ('hi')

def findNextTrans1(isBuy, name, args1):
    data = np.load('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npy')
    args2 = np.arange(data.shape[0])[data[:, 3] == "E"]
    args3 = np.arange(data.shape[0])[data[:, 3] == "F"]
    args3 = np.concatenate((args2, args3))
    if isBuy:
        args3 = args3[data[args3, 4] == "B"]
    else:
        args3 = args3[data[args3, 4] == "S"]


    args3 = np.sort(args3)
    args3_index = stefan_nextAfter(args1, args3)
    args3 = args3[args3_index]

    return args3

def afterwardTransPointProfitBasicAnalysis():
    names = giveFNames()
    a0 = 0
    coeffs = []
    spreadPlot1 = [[], [], [], []]
    spreadPlot2 = [[], [], [], []]
    for name in names:
        a0 += 1
        print (name)
        for isBuy in [True, False]:
            #isBuy = False

            transProfitData = np.load('./resultData/transProfits/' + str(name) + '.npy')
            validTransArgs = transProfitData[:, 0].astype(int)
            profits = transProfitData[:, 1]
            spread = np.load('./inputData/ITCH_LOB/spread/' + name + '.npy')
            shares = np.load('./inputData/ITCH_LOB/nearSharesAll/' + name + '.npy')
            data = np.load('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npy')
            prices = data[:, 2].astype(int)
            if isBuy:
                argsBid = np.squeeze(np.argwhere(data[:, 4] == "B"))
                bestBid = np.load('./inputData/ITCH_LOB/BestBid/' + name + '.npy')
                #validTransArgs = np.intersect1d(validTransArgs, argsBid)
                pricesO = prices[validTransArgs]
                tup1 = np.intersect1d(argsBid, validTransArgs, return_indices=True)
                argsInTrans = tup1[2]
                spread = spread[validTransArgs[argsInTrans]]
                pricesO = prices[validTransArgs[argsInTrans]]
                shares = shares[validTransArgs[argsInTrans]]

                validTransArgs = findNextTrans1(False, name, validTransArgs[argsInTrans]) #True and False "swapped" on purpose.
            else:
                argsBid = np.squeeze(np.argwhere(data[:, 4] == "S"))
                bestBid = np.load('./inputData/ITCH_LOB/BestAsk/' + name + '.npy')
                #validTransArgs = np.intersect1d(validTransArgs, argsBid)
                pricesO = prices[validTransArgs]
                tup1 = np.intersect1d(argsBid, validTransArgs, return_indices=True)
                argsInTrans = tup1[2]
                pricesO = prices[validTransArgs[argsInTrans]]
                spread = spread[validTransArgs[argsInTrans]]
                shares = shares[validTransArgs[argsInTrans]]

                validTransArgs = findNextTrans1(True, name, validTransArgs[argsInTrans])

            #quit()
            del data

            pricesN = prices[validTransArgs]



            profitEstimate = (pricesN - pricesO)
            if not isBuy:
                profitEstimate = -1 * profitEstimate
            profits = profits[argsInTrans]




            #profits[profits > 1200] = 1200
            #profits[profits <-1200] =-1200
            #profits = profits - np.mean(profits)
            #profits = profits / (np.mean(profits**2.0)**0.5)
            shares[shares> 10000] = 10000
            shares[shares<-10000] =-10000

            #print (profits.shape)
            #print (profitEstimate.shape)

            #print (scipy.stats.pearsonr(profitEstimate, profits))
            #print (np.mean(profits))
            #print (np.mean(profitEstimate))
            profits = profits - np.mean(profits)
            profitEstimate = profitEstimate - np.mean(profitEstimate)
            #print (np.mean(profits**2.0)**0.5)
            #print (np.mean(profitEstimate**2.0)**0.5)

            #spread = spread[validTransArgs[argsInTrans]]

            #quit()

            print ("A")
            #print (scipy.stats.pearsonr(spread, profits))
            #print (scipy.stats.pearsonr(spread, profitEstimate))
            #print (scipy.stats.pearsonr(profitEstimate, profits))
            #print (statCor(spread, profits))
            #print (statCor(spread, profitEstimate))
            print (statCor(shares[:, -40], profitEstimate))
            print (statCor(shares[:, -40], profits))
            print (statCor(profitEstimate, profits))
            #quit()

            #plt.hist(profitEstimate, bins=100)
            #plt.show()
            #plt.hist(profits, bins=100)
            #plt.show()
            #print (scipy.stats.pearsonr(pricesN - pricesO, profits))
            #quit()
            #quit()

            '''
            bestBid1 = bestBid[1:] - bestBid[:-1]

            shares = np.load('./inputData/ITCH_LOB/nearSharesAll/' + name + '.npy')
            spread = np.load('./inputData/ITCH_LOB/spread/' + name + '.npy')



            bestBid1 = bestBid1[argsInTrans-2]


            #sharesFull, argsInTrans = findTransactPriority(isBuy, name, validTransArgs, 0)
            #shares = shares[validTransArgs[argsInTrans]-1, 40:-40]
            #shares = shares[validTransArgs[argsInTrans]-1, 33-8:-33+8]
            print ("B")
            print (shares.shape)
            print (argsInTrans.shape)
            shares = shares[validTransArgs[argsInTrans]-1]
            print (shares.shape)
            spread = spread[validTransArgs[argsInTrans]]


            profits = profits[argsInTrans]

            profits[profits > 1200] = 1200
            profits[profits <-1200] =-1200

            #print (scipy.stats.pearsonr(bestBid1, profits))
            #quit()


            shares1 = np.mean(shares[:, 41-8:41-4], axis=1)
            shares2 = np.mean(shares[:, -(41-4):-(41-8)], axis=1)
            shares3 = np.mean(shares[:, 41-16:41-8], axis=1)
            shares4 = np.mean(shares[:, -(41-8):-(41-16)], axis=1)
            shares5 = np.mean(shares[:, :41-16], axis=1)
            shares6 = np.mean(shares[:, -(41-16):], axis=1)
            shares = shares[:, 41-7:-(41-7)]

            shares[:, 2] = shares1
            shares[:, -3] = shares2
            shares[:, 1] = shares3
            shares[:, -2] = shares4
            shares[:, 0] = shares5
            shares[:, -1] = shares6



            #plt.plot(shares[0, :])
            #plt.show()
            #quit()
            #shares[:, 4] = np.mean(shares[:, :4], axis=1)
            #shares[:,-4] = np.mean(shares[:,-4:], axis=1)
            #shares = shares[:, 3:-3]
            #quit()

            for a in range(0, shares.shape[1]):
                shares[a] = shares[a] - np.mean(shares[a])
                shares[a] = shares[a] / (np.mean(shares[a] ** 2.0) ** 0.5)

            spread[spread < 0] = 0
            spread[spread > 1000] = 1000
            spread = spread - np.mean(spread)
            spread = spread / (np.mean(spread**2.0)**0.5)

            if isBuy:
                shares = shares[:, -1::-1]

            #plt.hist(profits, bins=100)
            #plt.show()
            #quit()

            #profits = profits[spread > 150]
            #shares = shares[spread > 150]
            length1 = spread.shape[0] // 4
            #print (scipy.stats.pearsonr(spread, profits))
            if isBuy:
                spreadPlot1[0].append(scipy.stats.pearsonr(spread[:length1], profits[:length1])[0])
                spreadPlot1[1].append(scipy.stats.pearsonr(spread[length1:length1*2], profits[length1:length1*2])[0])
                spreadPlot1[2].append(scipy.stats.pearsonr(spread[2*length1:length1*3], profits[2*length1:length1*3])[0])
                spreadPlot1[3].append(scipy.stats.pearsonr(spread[3*length1:length1*4], profits[3*length1:length1*4])[0])
            else:
                spreadPlot2[0].append(scipy.stats.pearsonr(spread[:length1], profits[:length1])[0])
                spreadPlot2[1].append(scipy.stats.pearsonr(spread[length1:length1*2], profits[length1:length1*2])[0])
                spreadPlot2[2].append(scipy.stats.pearsonr(spread[2*length1:length1*3], profits[2*length1:length1*3])[0])
                spreadPlot2[3].append(scipy.stats.pearsonr(spread[3*length1:length1*4], profits[3*length1:length1*4])[0])

            #quit()

            #coeffs.append(np.array([coeff1, coeff2, coeff3, coeff4, coeff5, coeff6, coeff7, coeff8]))
            #length1 = shares.shape[0] #// 2
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression().fit(shares[:], profits[:])
            print (np.array(reg.coef_))
            #quit()
            coeffs.append(np.array(reg.coef_))

            #plt.plot(np.array(reg.coef_))
            #plt.show()
            '''


        #plt.plot(np.array(coeffs).T)
        #plt.show()
        #print (spreadPlot1)
        #plt.plot(np.array(spreadPlot1))
        #plt.show()
        #plt.plot(np.array(spreadPlot2))
        #plt.show()
        #quit()
        #if a0 == 2:
        #    quit()

    coeffs = np.array(coeffs).T

def saveProfitCoefficients():
    names = giveFNames()
    a0 = 0
    coeffs = []
    expsFull = []
    #spreadPlot1 = [[], [], [], []]
    #spreadPlot2 = [[], [], [], []]
    for name in names:
        a0 += 1
        print (name)
        for isBuy in [True, False]:
            #name = "20200225"
            #isBuy = False

            #transProfitData = np.load('./resultData/transProfits/' + str(name) + '.npy')
            transProfitData = np.load('./resultData/transProfitsWithForced/' + str(name) + '.npy')
            validTransArgs = transProfitData[:, 0].astype(int)
            profits = transProfitData[:, 1]
            #data = np.load('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npy')
            data = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz')

            prices = data[:, 2].astype(int)
            if isBuy:
                argsBid = np.squeeze(np.argwhere(data[:, 4] == "B"))
                #bestBid = np.load('./inputData/ITCH_LOB/BestBid/' + name + '.npy')
            else:
                argsBid = np.squeeze(np.argwhere(data[:, 4] == "S"))
                #bestBid = np.load('./inputData/ITCH_LOB/BestAsk/' + name + '.npy')

            #bestBidShift = bestBid[1:] - bestBid[:-1]
            del data
            tup1 = np.intersect1d(argsBid, validTransArgs, return_indices=True)
            argsInTrans = tup1[2]

            #bestBidShift = bestBidShift[tup1[1] - 2]

            spread = np.load('./inputData/ITCH_LOB/spread/' + name + '.npy')
            #shares = np.load('./inputData/ITCH_LOB/nearSharesAll/' + name + '.npy')
            #spread = loadnpz('./inputData/ITCH_LOB/spread/' + name + '.npz')
            shares = loadnpz('./inputData/ITCH_LOB/nearSharesAll/' + name + '.npz')

            shares = shares[validTransArgs[argsInTrans]-1]
            spread = spread[validTransArgs[argsInTrans]-1]
            profits = profits[argsInTrans]
            profits[profits > 1200] = 1200
            profits[profits <-1200] =-1200
            #profits = profits - np.mean(profits)
            profits = profits / 200
            #profits = profits / (np.mean(profits**2.0)**0.5)
            #print ((np.mean(profits**2.0)**0.5))
            #quit()

            #print (statCor(bestBidShift, profits))
            #quit()

            spread[spread>500] = 500
            spread1 = np.zeros((spread.shape[0], 3))
            spread1[spread == 100, 0] = 1
            spread1[spread == 200, 1] = 1
            spread1[spread > 200, 2] = 1

            shares[shares> 10000] = 10000
            shares[shares<-10000] =-10000


            shares1 = np.mean(shares[:, 41-8:41-4], axis=1)
            shares2 = np.mean(shares[:, -(41-4):-(41-8)], axis=1)
            shares3 = np.mean(shares[:, 41-16:41-8], axis=1)
            shares4 = np.mean(shares[:, -(41-8):-(41-16)], axis=1)
            shares5 = np.mean(shares[:, :41-16], axis=1)
            shares6 = np.mean(shares[:, -(41-16):], axis=1)

            shares = shares[:, 41-7:-(41-7)]

            shares[:, 2] = shares1
            shares[:, -3] = shares2
            shares[:, 1] = shares3
            shares[:, -2] = shares4
            shares[:, 0] = shares5
            shares[:, -1] = shares6

            if isBuy:
                shares = shares[:, -1::-1]

            sharesTemp = np.zeros((shares.shape[0], shares.shape[1]+3))
            sharesTemp[:, :-3] = shares
            sharesTemp[:, -3:] = spread1
            shares = sharesTemp

            for a in range(0, shares.shape[1] - 3):
                #shares[:, a] = shares[:, a] - np.mean(shares[:, a])
                #print (np.mean(shares[:, a] ** 2.0) ** 0.5)
                shares[:, a] = shares[:, a] / 1000.0
                #shares[:, a] = shares[:, a] / (np.mean(shares[:, a] ** 2.0) ** 0.5)
            #quit()
            #from sklearn.linear_model import LinearRegression
            #reg = LinearRegression().fit(shares, profits)
            from sklearn.linear_model import Ridge
            reg = Ridge(alpha=100.0).fit(shares, profits)
            ar1 = np.array(reg.coef_)
            ar2 = np.zeros(ar1.shape[0]+1)
            ar2[:-1] = ar1
            ar2[-1] = reg.intercept_
            coeffs.append(ar2)

        plt.plot(np.array(coeffs).T)
        plt.plot(np.array(expsFull).T)
        plt.show()

def saveLinearProfitEstimates():
    coeffs = np.load("./temporary/executionProfitCoefs.npy")
    coeff = np.mean(coeffs, axis=0)

    finalProfits = np.zeros([])
    names = giveFNames()
    for name in names:
        print (name)
        size1 = np.load('./resultData/transProfits/' + str(name) + '.npy').shape[0]
        bothProfits = np.zeros(size1)

        for isBuy in [True, False]:
            #name = "20200225"
            #isBuy = False
            transProfitData = np.load('./resultData/transProfits/' + str(name) + '.npy')
            validTransArgs = transProfitData[:, 0].astype(int)
            profits = transProfitData[:, 1]

            data = np.load('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npy')
            if isBuy:
                argsBid = np.squeeze(np.argwhere(data[:, 4] == "B"))
            else:
                argsBid = np.squeeze(np.argwhere(data[:, 4] == "S"))
            del data
            tup1 = np.intersect1d(argsBid, validTransArgs, return_indices=True)
            argsInTrans = tup1[2]

            spread = np.load('./inputData/ITCH_LOB/spread/' + name + '.npy')
            shares = np.load('./inputData/ITCH_LOB/nearSharesAll/' + name + '.npy')

            shares = shares[validTransArgs[argsInTrans]-1]
            spread = spread[validTransArgs[argsInTrans]-1]
            profits = profits[argsInTrans]
            profits[profits > 1200] = 1200
            profits[profits <-1200] =-1200
            profits = profits / 200

            spread[spread>500] = 500
            spread1 = np.zeros((spread.shape[0], 3))
            spread1[spread == 100, 0] = 1
            spread1[spread == 200, 1] = 1
            spread1[spread > 200, 2] = 1

            shares[shares> 10000] = 10000
            shares[shares<-10000] =-10000

            shares1 = np.mean(shares[:, 41-8:41-4], axis=1)
            shares2 = np.mean(shares[:, -(41-4):-(41-8)], axis=1)
            shares3 = np.mean(shares[:, 41-16:41-8], axis=1)
            shares4 = np.mean(shares[:, -(41-8):-(41-16)], axis=1)
            shares5 = np.mean(shares[:, :41-16], axis=1)
            shares6 = np.mean(shares[:, -(41-16):], axis=1)

            shares = shares[:, 41-7:-(41-7)]

            shares[:, 2] = shares1
            shares[:, -3] = shares2
            shares[:, 1] = shares3
            shares[:, -2] = shares4
            shares[:, 0] = shares5
            shares[:, -1] = shares6

            if isBuy:
                shares = shares[:, -1::-1]

            sharesTemp = np.zeros((shares.shape[0], shares.shape[1]+3))
            sharesTemp[:, :-3] = shares
            sharesTemp[:, -3:] = spread1
            shares = sharesTemp

            for a in range(0, shares.shape[1] - 3):
                shares[:, a] = shares[:, a] / 1000.0

            expsProfit = np.zeros(shares.shape[0]) + coeff[-1]
            for a in range(0, shares.shape[1]):
                expsProfit = expsProfit + (shares[:, a] * coeff[a])

            bothProfits[argsInTrans] = expsProfit

        #finalProfits = np.concatenate((finalProfits, bothProfits))
        profits = np.load('./resultData/transProfits/' + str(name) + '.npy')[:, 1]
        print (statCor(bothProfits, profits))
        print (np.mean(bothProfits))
        print (np.mean(profits))

        np.save("./temporary/linearExpectedProfits/" + name + ".npy", bothProfits)

def saveProfitXY():
    names = giveFNames()
    a0 = 0
    Xdata = np.zeros((0, 18))
    profitFull = np.array([])
    transSharesFull = np.array([])
    #spreadPlot1 = [[], [], [], []]
    #spreadPlot2 = [[], [], [], []]
    for name in names:
        a0 += 1
        print (name)
        for isBuy in [True, False]:
            #name = "20200225"
            #isBuy = False

            transProfitData = np.load('./resultData/transProfits/' + str(name) + '.npy')
            validTransArgs = transProfitData[:, 0].astype(int)
            profits = transProfitData[:, 1]
            #data = np.load('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz')
            data = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz')

            #sharesTrans = data[validTransArgs][:, 1].astype(int)
            #plt.hist(sharesTrans, bins=100, range=(0, 1000))
            #plt.show()
            #quit()
            prices = data[:, 2].astype(int)
            if isBuy:
                argsBid = np.squeeze(np.argwhere(data[:, 4] == "B"))
                #bestBid = np.load('./inputData/ITCH_LOB/BestBid/' + name + '.npy')
            else:
                argsBid = np.squeeze(np.argwhere(data[:, 4] == "S"))
                #bestBid = np.load('./inputData/ITCH_LOB/BestAsk/' + name + '.npy')

            validTransArgsBid = np.intersect1d(argsBid, validTransArgs)
            sharesTrans = data[validTransArgsBid, 1].astype(int)
            transSharesFull = np.concatenate((transSharesFull, sharesTrans))

            '''

            #bestBidShift = bestBid[1:] - bestBid[:-1]
            del data
            tup1 = np.intersect1d(argsBid, validTransArgs, return_indices=True)
            argsInTrans = tup1[2]

            #plt.hist(data[valid])

            #bestBidShift = bestBidShift[tup1[1] - 2]

            spread = np.load('./inputData/ITCH_LOB/spread/' + name + '.npy')
            shares = np.load('./inputData/ITCH_LOB/nearSharesAll/' + name + '.npy')

            shares = shares[validTransArgs[argsInTrans]-1]
            spread = spread[validTransArgs[argsInTrans]-1]
            profits = profits[argsInTrans]
            profits[profits > 1200] = 1200
            profits[profits <-1200] =-1200
            #profits = profits - np.mean(profits)
            profits = profits / 200
            #profits = profits / (np.mean(profits**2.0)**0.5)
            #print ((np.mean(profits**2.0)**0.5))
            #quit()

            #print (statCor(bestBidShift, profits))
            #quit()

            spread[spread>500] = 500
            spread1 = np.zeros((spread.shape[0], 3))
            spread1[spread == 100, 0] = 1
            spread1[spread == 200, 1] = 1
            spread1[spread > 200, 2] = 1

            shares[shares> 10000] = 10000
            shares[shares<-10000] =-10000


            shares1 = np.mean(shares[:, 41-8:41-4], axis=1)
            shares2 = np.mean(shares[:, -(41-4):-(41-8)], axis=1)
            shares3 = np.mean(shares[:, 41-16:41-8], axis=1)
            shares4 = np.mean(shares[:, -(41-8):-(41-16)], axis=1)
            shares5 = np.mean(shares[:, :41-16], axis=1)
            shares6 = np.mean(shares[:, -(41-16):], axis=1)

            shares = shares[:, 41-7:-(41-7)]

            shares[:, 2] = shares1
            shares[:, -3] = shares2
            shares[:, 1] = shares3
            shares[:, -2] = shares4
            shares[:, 0] = shares5
            shares[:, -1] = shares6

            if isBuy:
                shares = shares[:, -1::-1]

            sharesTemp = np.zeros((shares.shape[0], shares.shape[1]+4))
            sharesTemp[:, :-4] = shares
            sharesTemp[:, -4:-1] = spread1
            if isBuy:
                sharesTemp[:, -1] = 0
            else:
                sharesTemp[:, -1] = 1
            shares = sharesTemp

            for a in range(0, shares.shape[1] - 4):
                #shares[:, a] = shares[:, a] - np.mean(shares[:, a])
                #print (np.mean(shares[:, a] ** 2.0) ** 0.5)
                shares[:, a] = shares[:, a] / 1000.0
                #shares[:, a] = shares[:, a] / (np.mean(shares[:, a] ** 2.0) ** 0.5)
            #quit()
            #from sklearn.linear_model import LinearRegression
            #reg = LinearRegression().fit(shares, profits)

            Xdata = np.concatenate((Xdata, shares))
            profitFull = np.concatenate((profitFull, profits))
            '''
    np.save("./temporary/ExchangeProfitWdata.npy", transSharesFull)
    #np.save("./temporary/ExchangeProfitXdata.npy", Xdata)
    #np.save("./temporary/ExchangeProfitYdata.npy", profitFull)
    ('hi')

class SuperSimpleNet(nn.Module):
    def __init__(self, Ni, Nh, No):
        super(SuperSimpleNet, self).__init__()
        self.lin1 = nn.Linear(Ni, Nh)
        self.lin2 = nn.Linear(Nh, No)
        self.lin3 = nn.Linear(No, 1)
        self.lin4 = nn.Linear(1, 1)
        self.nonlin = torch.tanh

    def forward(self, x):
        x = self.lin1(x)
        x = self.nonlin(x)
        x = self.lin2(x)

        x = self.nonlin(x)
        x = self.lin3(x)

        x = self.nonlin(x)
        x = self.lin4(x)
        return x

def giveModelParams(model):
    params1 = []
    for p in model.parameters():
        d1 = torch.clone(p.data)
        s1 = d1.shape
        s2 = 1
        for z in range(0, len(s1)):
            s2 = s2 * s1[z]
        params1 = params1 + list(d1.reshape((s2,)))
    params1 = torch.tensor(params1).data.numpy()
    return params1

def trainBasicProfitPredictor():
    Xdata = np.load("./temporary/ExchangeProfitXdata.npy") ** 0.8
    profitFull = np.load("./temporary/ExchangeProfitYdata.npy")
    Weights = np.load("./temporary/ExchangeProfitWdata.npy")
    Weights = Weights / 200.0

    shift = 10000
    for a in range(0, Xdata.shape[1] - 1):
        arAvg = np.cumsum(Xdata[:, a])[:-1]
        arAvg = (arAvg[shift:] - arAvg[:-shift]) / shift
        Xdata[shift+1:, a] = Xdata[shift+1:, a] - arAvg

    Xdata, profitFull, Weights = Xdata[shift+1:], profitFull[shift+1:], Weights[shift+1:]

    #for a in range(0, Xdata.shape[1] - 1):
    #    plt.plot(np.cumsum(Xdata[:, a]))
    #    plt.show()
    #print ()
    #quit()

    Xdata = np.concatenate((Xdata.T, Weights.reshape((1, Weights.shape[0])))).T
    shuffleArgs = np.random.choice(profitFull.shape[0], profitFull.shape[0], replace=False)
    Xdata = Xdata[shuffleArgs]
    profitFull = profitFull[shuffleArgs]
    Weights = Weights[shuffleArgs]
    #plt.hist(Weights, bins=100, range=(0, 1000))
    #plt.show()

    #from sklearn.linear_model import Ridge
    #reg = Ridge(alpha=100.0).fit(Xdata, profitFull)
    #predProfit = reg.predict(Xdata)

    #print (statCor(predProfit, profitFull))
    #quit()
    Xdata = torch.tensor(Xdata).float()
    profitFull = torch.tensor(profitFull).float().reshape((profitFull.shape[0], 1))
    #predProfit = torch.tensor(predProfit).float().reshape((profitFull.shape[0], 1))
    Weights = torch.tensor(Weights).float().reshape((profitFull.shape[0], 1))


    #loss = torch.mean((predProfit ** 2.0) - (2.0 * predProfit * profitFull))
    #print (loss)
    #quit()

    model = SuperSimpleNet(Xdata.shape[1], 10, 5)
    #model = torch.load('./temporary/basicModel2.pt')
    #model = torch.load('./temporary/basicModel2_3.pt')
    #model = torch.load('./temporary/basicModel3_1.pt')
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-1) #0.5
    M = 10000
    #iterNum = 301
    iterNum = Xdata.shape[0] // M

    #-0.044574298
    losses = []
    losses2 = []
    epochs = 100
    for a in range(0, epochs):
        for b in range(0, iterNum):
            #print (b)
            X = Xdata[b*M:(b+1)*M]
            Y = profitFull[b*M:(b+1)*M]
            W = Weights[b*M:(b+1)*M]

            YPred = model(X)
            loss = torch.mean( ((YPred ** 2.0) - (2.0 * YPred * Y)) * W )
            losses.append(loss.data.numpy())
            #losses2.append(loss2.data.numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (b % 300 == 0) and (b > 0):
                print (a)
                print (np.mean(losses[-300:]))
                torch.save(model, './temporary/basicModel4.pt')

                YPred2 = model(Xdata[300*10000:])
                print ( torch.mean( ((YPred2 ** 2.0) - (2.0 * YPred2 * profitFull[300*10000:])) * Weights[300*10000:] ))

def checkPower():
    def doPower(x, a, b):
        return a * (x ** b)
    Xdata = np.load("./temporary/ExchangeProfitXdata.npy")
    profitFull = np.load("./temporary/ExchangeProfitYdata.npy")
    Weights = np.load("./temporary/ExchangeProfitWdata.npy")
    Weights = Weights / 200.0
    from scipy.optimize import curve_fit
    for a in range(0, Xdata.shape[1] - 4):
        print (curve_fit(doPower, Xdata[:, a], profitFull)[0])
    quit()

def saveBasicPrediction1():
    #model = torch.load('./temporary/basicModel1.pt')
    #model = torch.load('./temporary/basicModel2.pt')
    model = torch.load('./temporary/basicModel4.pt')
    Xdata = np.load("./temporary/ExchangeProfitXdata.npy") ** 0.8

    shift = 10000
    for a in range(0, Xdata.shape[1] - 1):
        arAvg = np.cumsum(Xdata[:, a])[:-1]
        arAvg = (arAvg[shift:] - arAvg[:-shift]) / shift
        Xdata[shift+1:, a] = Xdata[shift+1:, a] - arAvg

    Weights = np.load("./temporary/ExchangeProfitWdata.npy")
    Weights = Weights / 200.0

    Xdata, Weights = Xdata[shift+1:], Weights[shift+1:]

    Xdata = np.concatenate((Xdata.T, Weights.reshape((1, Weights.shape[0])))).T
    Xdata = torch.tensor(Xdata).float()
    fullPredY = model(Xdata).data.numpy()[:, 0]
    Xdata = Xdata.data.numpy()

    #for a in range(0, Xdata.shape[1]):
    #    plt.plot(np.cumsum(Xdata[:, a] - np.mean(Xdata[:, a])))
    #    plt.plot(np.cumsum(np.abs(Xdata[:, a] - np.mean(Xdata[:, a]))))
    #    plt.show()
    #quit()

    #plt.hist(fullPredY, bins=100)
    #plt.hist(profitFull/6, bins=100)
    #plt.show()
    np.save('./temporary/basicPredY2.npy', fullPredY)

    profitFull = np.load("./temporary/ExchangeProfitYdata.npy")
    print (np.mean(fullPredY))
    print (np.mean(profitFull))
    print (np.mean(fullPredY ** 2.0) ** 0.5)
    print (np.mean(profitFull ** 2.0) ** 0.5)
    quit()

def compressFilesInDirectory(folder):
    fnames = os.listdir(folder)
    if '.DS_Store' in fnames:
        fnames.remove('.DS_Store')
    for name in fnames:
        print (name)
        nameFull = folder + name
        data = np.load(nameFull)
        nameFull = nameFull[:-1] + "z"
        np.savez_compressed(nameFull, data)

def determineSellerLoss(name, model, subset, isBuy):
    data = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz')
    bestBid = loadnpz('./inputData/ITCH_LOB/BestBid/' + name + '.npz')

def makeChoices(probs):
    nums = np.random.uniform(size=probs.shape[0]).repeat(probs.shape[1]).reshape(probs.shape)
    nums = nums - probs
    nums = np.sign(nums)
    numsDiff = nums[:, 1:] - nums[:, :-1]
    numsDiff = ((numsDiff * -1) / 2).astype(int)
    nums = ((nums + 1) / 2).astype(int)
    nums[:, :-1] = numsDiff
    counts = np.arange(probs.shape[1]).repeat(probs.shape[0]).reshape(probs.shape[1], probs.shape[0]).T
    choices = np.sum(counts * nums, axis=1)
    return choices

def saveXCompressed():
    #name = '20200117'
    #isBuy = True
    def saveComponent(nameNum):
        names = giveFNames()
        name = names[nameNum]
        print (name)
        data = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz')
        _, milisecondArgs = np.unique(data[:, 0][-1::-1], return_index=True)
        milisecondArgs = data.shape[0] - 1 - milisecondArgs

        for isBuy in [True, False]:
            extraCount = 0
            if isBuy:
                #data = data[data[:, 4] == "B"]
                argsBuy = np.argwhere(data[:, 4] == "B")[:, 0] #New Jun 15
                LOBhistory = loadnpz('./inputData/ITCH_LOB/BuyLOB/' + name + '.npz', allow_pickle=True)
                bestPrice = loadnpz('./inputData/ITCH_LOB/bestPrice/Bid_' + name + '.npz')
                shift = 10
            else:
                #data = data[data[:, 4] == "S"]
                argsBuy = np.argwhere(data[:, 4] == "S")[:, 0] #New Jun 15
                LOBhistory = loadnpz('./inputData/ITCH_LOB/SellLOB/' + name + '.npz', allow_pickle=True)
                bestPrice = loadnpz('./inputData/ITCH_LOB/bestPrice/Ask_' + name + '.npz')
                shift = -10

            allShares = loadnpz('./inputData/ITCH_LOB/NearBestOrders/' + name + '.npz') #new Aug 25

            oncePerM = np.arange(data.shape[0])
            bestPrice = bestPrice[oncePerM]

            finalData = np.zeros((100000000*3, 3)).astype(int)
            Yfinal = np.zeros((100000000*3, 3)).astype(int)
            #global price, arg in listing, que position from bottum
            countsDone = 0
            M = 100
            count = 0
            count1 = 0
            for a in range(0, len(LOBhistory)):  #Start at 0
                #print (a)
                #print (countsDone)

                #args = stefan_squeeze(np.argwhere( np.abs(a - bestPrice + shift) <= np.abs(shift) )).astype(int)
                args = stefan_squeeze(np.argwhere( np.abs(a - bestPrice[milisecondArgs] + shift) <= (np.abs(shift) + 1) )).astype(int) #+1 includes the orders just below best price
                args = milisecondArgs[args]

                if args.shape[0] > 0:
                    a2 = (a - bestPrice[args] + shift + np.abs(shift)).astype(int)
                    relPrice1 = a - bestPrice[args]
                    if isBuy:
                        relPrice1 = relPrice1 * -1
                    #ar = np.array(LOBhistory[a][0])
                    ar = argsBuy[np.array(LOBhistory[a][0])] #Jun 16
                    argsInAr = stefan_nextAfter(args+1, ar) - 1 #TODO correct -1 to mean not existing.
                    unique1, lasts1 = np.unique(argsInAr[-1::-1], return_index=True)
                    lasts1 = argsInAr.shape[0] - lasts1 #Not -1 because we want one after last instance
                    lasts1 = np.concatenate((np.array([0]), lasts1))

                    for b in range(0, unique1.shape[0]):
                        size1 = lasts1[b+1] - lasts1[b]
                        if len(LOBhistory[a][1][unique1[b]]) == 0:
                            local_XTimes = args[lasts1[b]:lasts1[b+1]]#np.zeros(size1).astype(int) + ar[unique1[b]] #Is this correct? I think not.
                            local_RelativePrice = relPrice1[lasts1[b]:lasts1[b+1]]#a2[lasts1[b]:lasts1[b+1]]#np.zeros(size1).astype(int) + a2[b]
                            local_Depth = np.zeros(size1).astype(int)
                            all_local = np.array([local_XTimes, local_RelativePrice, local_Depth]).T
                            finalData[countsDone:countsDone + local_Depth.shape[0]] = all_local

                            local_Xarg = np.zeros(size1).astype(int) + unique1[b] #argsBuy[LOBhistory[a][0][unique1[b]]]
                            local_price = np.zeros(size1).astype(int) + a
                            local_quePos = np.zeros(size1).astype(int)
                            local_Y = np.array([local_Xarg, local_price, local_quePos]).T
                            Yfinal[countsDone:countsDone + local_Depth.shape[0]] = local_Y

                            countsDone += local_Depth.shape[0]

                        else:
                            size2 = len(LOBhistory[a][1][unique1[b]]) + 1

                            local_XTimes = args[lasts1[b]:lasts1[b+1]] #np.zeros(len(LOBhistory[a][1][unique1[b]]) + 1).astype(int) + ar[unique1[b]] #Is this correct? I think not.

                            local_RelativePrice = relPrice1[lasts1[b]:lasts1[b+1]]#a2[lasts1[b]:lasts1[b+1]] #np.zeros(size2).astype(int) + a2[b]
                            local_Depth = np.zeros(size2).astype(int)
                            depth2 = np.array(LOBhistory[a][1][unique1[b]])[:, 1]
                            depth2 = np.cumsum(depth2[-1::-1])
                            local_Depth[1:] = depth2

                            local_Depth = local_Depth.repeat(size1)
                            local_RelativePrice = local_RelativePrice.repeat(size2).reshape((size1, size2)).T.reshape((size1 * size2, ))
                            local_XTimes = local_XTimes.repeat(size2).reshape((size1, size2)).T.reshape((size1 * size2, ))
                            all_local = np.array([local_XTimes, local_RelativePrice, local_Depth]).T
                            finalData[countsDone:countsDone + local_Depth.shape[0]] = all_local

                            local_Xarg = np.zeros(size1 * size2).astype(int) + unique1[b] #argsBuy[LOBhistory[a][0][unique1[b]]]
                            local_price = np.zeros(size1 * size2).astype(int) + a
                            local_quePos = np.arange(size2)[-1::-1].astype(int).repeat(size1)
                            local_Y = np.array([local_Xarg, local_price, local_quePos]).T
                            Yfinal[countsDone:countsDone + local_Depth.shape[0]] = local_Y

                            countsDone += local_Depth.shape[0]

            finalData = finalData[:countsDone]
            Yfinal = Yfinal[:countsDone]

            #print (np.max(Yfinal[:, 0]))

            if True:
                if isBuy:
                    np.savez_compressed('./inputData/ITCH_LOB/allOrderXY/X_Bid_' + name + '.npz', finalData)
                    np.savez_compressed('./inputData/ITCH_LOB/allOrderXY/Y_Bid_' + name + '.npz', Yfinal)
                else:
                    np.savez_compressed('./inputData/ITCH_LOB/allOrderXY/X_Ask_' + name + '.npz', finalData)
                    np.savez_compressed('./inputData/ITCH_LOB/allOrderXY/Y_Ask_' + name + '.npz', Yfinal)

    multiProcessor(saveComponent, 0, 52, 4, doPrint = True)
    #saveComponent(0)


#print ("R")
#saveXCompressed()
#quit()

def formLOBrecentShares():

    #isBuy = True
    #for name in names:
    def saveComponent(nameNum):
        names = giveFNames()
        name = names[nameNum]
        print (name)
        allSharesFull = []
        for isBuy in [True, False]:
            if isBuy:
                LOBhistory = loadnpz('./inputData/ITCH_LOB/BuyLOB/' + name + '.npz', allow_pickle=True)
            else:
                LOBhistory = loadnpz('./inputData/ITCH_LOB/SellLOB/' + name + '.npz', allow_pickle=True)
            M = 10
            sizes = []
            LOBNewFull = []
            FullNums = []
            for Num in range(0, len(LOBhistory)):
                LOBNew = [LOBhistory[Num][0], np.zeros((len(LOBhistory[Num][0]), M, 2)).astype(int)]
                tempNums = []
                for a in range(0, len(LOBhistory[Num][1])):
                    num = len(LOBhistory[Num][1][a])
                    tempNums.append(num)
                    vals = np.zeros((M, 2)).astype(int)
                    maxBack = min(len(LOBhistory[Num][1][a][-M:]), M)

                    if num != 0:
                        vals2 = np.array(LOBhistory[Num][1][a])

                        if num >= M:
                            vals3 = vals2[:-2]
                            vals3 = np.concatenate((vals3[vals3[:, 1] <= 10], vals3[vals3[:, 1] > 10]))
                            vals2[:-2] = vals3

                        vals[-maxBack:, :] =  vals2[-M:, :2] #sept 18 #vals2[-M:, :]
                        vals[0, 1] = np.sum( vals2[:-(M-1), 1] )

                    LOBNew[1][a] = vals
                LOBNewFull.append(LOBNew)
                FullNums.append(tempNums)
            del LOBhistory

            data = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz')

            if isBuy:
                data = data[data[:, 4] == "B"]
                #LOB_share = loadnpz('./inputData/ITCH_LOB/BuyShareLOB/' + name + '.npz', allow_pickle=True)
                bestPrice = loadnpz('./inputData/ITCH_LOB/BestBid/' + name + '.npz')
                shift = 20
            else:
                data = data[data[:, 4] == "S"]
                #LOB_share = loadnpz('./inputData/ITCH_LOB/SellShareLOB/' + name + '.npz', allow_pickle=True)
                bestPrice = loadnpz('./inputData/ITCH_LOB/BestAsk/' + name + '.npz')
                shift = -20

            oncePerM = np.arange(data.shape[0])
            bestPrice = bestPrice[oncePerM]
            allShares = np.zeros((oncePerM.shape[0], (np.abs(shift) * 2) + 1, M))
            for a in range(0, len(LOBNewFull)): #1879
                if a % 100 == 0:
                    print (a)
                #a from bestPrice - 40 to bestPrice
                args = stefan_squeeze(np.argwhere( np.abs(a - bestPrice + shift) <= np.abs(shift) )).astype(int)
                a2 = (a - bestPrice[args] + shift + np.abs(shift)).astype(int)
                ar = np.array(LOBNewFull[a][0])
                argsInAr = stefan_nextAfter(args+1, ar) - 1 #TODO correct -1 to mean not existing.

                if argsInAr.shape[0] > 0:
                    sharesNew = np.array(LOBNewFull[a][1])[argsInAr]
                    numsNow = np.array(FullNums[a])[argsInAr]
                    allShares[args, a2] = sharesNew[:, :, 1]
            allSharesFull.append(np.copy(allShares))

            del data
            del bestPrice
            del LOBNewFull
            del allShares
            del FullNums

        #print ("A")
        data = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz')
        dataSize = data.shape[0]
        bidArgs = np.argwhere(data[:, 4] == "B")[:, 0]
        askArgs = np.argwhere(data[:, 4] == "S")[:, 0]
        del data
        #print ("B")
        beforeBids = stefan_firstBefore(np.arange(dataSize), bidArgs)
        beforeAsks = stefan_firstBefore(np.arange(dataSize), askArgs)
        #print ("C")
        shiftSize = (np.abs(shift) * 2) + 1
        allShares = np.zeros((dataSize, shiftSize * 2, M)).astype(int)
        #print ("D")
        allShares[:, :shiftSize, :] = allSharesFull[0][beforeBids, :, :]
        del allSharesFull[0]
        allShares[:, shiftSize:, :] = allSharesFull[0][beforeAsks, :, :]
        #print ("E")
        np.savez_compressed('./inputData/ITCH_LOB/NearBestOrders/' + name + '.npz', allShares)
        del allShares
        del allSharesFull

    multiProcessor(saveComponent, 0, 52, 4, doPrint = True)

#formLOBrecentShares()
#quit()

def findMinorExecute(name, isBuy):

    if isBuy:
        bestPrice0 = loadnpz('./inputData/ITCH_LOB/bestPrice/Bid_' + name + '.npz')
        oType = 'B'
    else:
        bestPrice0 = loadnpz('./inputData/ITCH_LOB/bestPrice/Ask_' + name + '.npz')
        oType = 'S'

    data = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz', allow_pickle=True)
    argsBuy = np.argwhere(data[:, 4] == oType)[:, 0]
    data = data[argsBuy]
    #argsGood = argsBuy[np.isin(data[:, 3], np.array(['E', 'F']))]
    argsGood = np.arange(data.shape[0])[np.isin(data[:, 3], np.array(['E', 'F']))]
    argsGood = argsBuy[argsGood]

    prices = bestPrice0[argsGood].astype(int)

    if isBuy:
        prices = prices + 1
    else:
        prices = prices - 1

    return prices, argsGood



def individualTrajectories():

    #N = 0
    #for c in range(0, 100):
    #    vals = np.load('./temporary/valuePaths2/' + str(c) + '.npy')
    #    N += vals.shape[0]
    #print (N / 100)
    #print (vals.shape[0])
    #quit()


    '''
    oType = 'B'
    name = giveFNames()[0]
    cors = []
    for c in range(0, 100):
        #print (c)
        valsAll = np.load('./temporary/valuePaths2/' + str(c) + '.npy')
        #x = np.arange(valsAll.shape[0]) / valsAll.shape[0]

        #cor1 = np.polyfit(x, valsAll[:, 0], 1)[0]
        #cors.append(cor1)

        y = valsAll[:, 0]

        vals = []
        for b in range(0, 7):
            diff = y[0::(2**b)]
            diff = diff[1:] - diff[:-1]
            val = (np.mean(diff**2.0)/(2**b))**0.5
            vals.append(val)

        cors.append(vals)
        #print (y.shape)
        #quit()

    cors = np.array(cors)
    cors_mean = np.mean(cors, axis=0)
    cors_std = (np.mean(cors ** 2.0, axis=0)  - (cors_mean ** 2.0)) ** 0.5
    plt.plot(cors_mean)
    plt.plot(cors_std)
    plt.show()

    #plt.hist(cors)
    #plt.show()
    #print (np.mean(np.array(cors)))
    #print ((np.mean(np.array(cors)**2.0)/100) ** 0.5)
    quit()
    #'''

    oType = 'S'
    name = giveFNames()[10] #0
    loopNumber = 1
    data = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz', allow_pickle=True)
    data = data[data[:, 4] == oType]
    LOB_share = loadnpz('./inputData/ITCH_LOB/SellLOB/' + name + '.npz', allow_pickle=True)

    #finalData_0 = loadnpz('./inputData/Processed/inputXY/X_Ask_' + name + '.npz')
    #YData_0 = loadnpz('./inputData/Processed/inputXY/Y_Ask_' + name + '.npz')
    finalData_0 = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')
    YData_0 = loadnpz('./recursive/0/orderData/withCancels/Y_' + oType + '_' + name + '.npz')

    for c in range(40, 100):
        print ("C -- -- -- ", c)
        choice = np.random.choice(finalData_0.shape[0], 1)[0]

        finalDataNow = finalData_0[choice]
        YDataNow = YData_0[choice]

        a = YDataNow[1]
        quePos = YDataNow[2]
        bStart = YDataNow[0] + 1

        outputs = loadnpz('./recursive/' + str(1) + '/outputPredictions/' + oType + '_' + name + '.npz')[:, 1]
        #outputs = loadnpz('./recursive/' + str(3) + '/outputPredictions/' + oType + '_' + name + '.npz')

        argsInside = np.arange(finalData_0.shape[0])[YData_0[:, 1] == a]
        #argsGood_1 = loadnpz('./inputData/Processed/inputXY/argsGood_Bid_' + name + '.npz')
        #reverser = (np.zeros(np.max(argsGood_1) + 1 ) - 1).astype(int)
        #reverser[argsGood_1] = np.arange(argsGood_1.shape[0])
        #argsInside = reverser[argsInside]

        finalData = finalData_0[YData_0[:, 1] == a]
        YData = YData_0[YData_0[:, 1] == a]
        #for a in range(0, len(LOB_share)):
        #a =
        valsAll = np.array([])

        quePosses = np.array([])
        b = bStart
        notDone = True
        while ((b < len(LOB_share[a][0])-1 ) and notDone):
            #print ("A")
            #print (quePos)
            #print (LOB_share[a][1][b])
            changeFull = data[LOB_share[a][0][b+1]]
            orderNum = int(changeFull[6])
            orderType = changeFull[3]
            #timeNow = int(changeFull[0])

            argsInside2 = argsInside[np.logical_and(YData[:, 0] == b, YData[:, 2] == quePos)]
            #print (argsInside[np.logical_and(finalData[:, 0] == b)])
            #print (argsInside[np.logical_and(finalData[:, 0] == b, YData[:, 2] == quePos)].shape)
            #quit()
            times = finalData[np.logical_and(YData[:, 0] == b, YData[:, 2] == quePos)][:, 0]
            vals = outputs[argsInside2][np.argsort(times)]

            #if vals.shape[0] == 0:
            #    print (YData[:, 2][YData[:, 0] == b])
            #    if YData[:, 2][YData[:, 0] == b].shape[0] != 0:
            #        quit()
            vals = vals#[:1] #Remove!
            #if vals.shape[0] == 1:
                #quePosses.append(quePos)
            quePosses = np.concatenate((quePosses, quePos + np.zeros(vals.shape[0])))

            valsAll = np.concatenate((valsAll, vals))

            if (orderType == "D") or (orderType == "F"):
                #print ("U")
                arg = np.argwhere(np.array(LOB_share[a][1][b])[:, 0] == orderNum)[0, 0]# + 1
                #print (arg)
                if arg < quePos:
                    quePos -= 1

            if orderType == "E":
                arg = np.argwhere(np.array(LOB_share[a][1][b])[:, 0] == orderNum)[0, 0] #+ 1
                if arg >= quePos:
                    notDone = False
                    #diff = valsAll[1:] - valsAll[:-1]
                    #print (scipy.stats.pearsonr(diff[:-1], diff[1:]))
                    #plt.plot(-1 * np.array(quePosses) / np.max(np.array(quePosses)))
                    #plt.plot(valsAll)
                    #plt.show()
                    #quit()
                    print ("Done1")
            b += 1
        print ("Done2")
        diff = valsAll[1:] - valsAll[:-1]
        print (scipy.stats.pearsonr(diff[:-1], diff[1:]))
        plt.plot(-1 * np.array(quePosses) / np.max(np.array(quePosses)))
        plt.plot(valsAll)
        plt.show()

        np.save('./temporary/valuePaths2/' + str(c) + '.npy', np.array([valsAll, quePosses]).T )


#individualTrajectories()
#quit()


def findLOBpossibleTrans(loopNumber, executeExtra=False, filesRun=(0, 15)):
    import time
    def saveComponent(nameNum):
        names = giveFNames()
        name = names[nameNum]
        print (name)
        for isBuy in [True, False]:

            #time1 = time.time()
            done1Count = 0
            if isBuy:
                LOB_share = loadnpz('./inputData/ITCH_LOB/BuyLOB/' + name + '.npz', allow_pickle=True)
                bestPrice0 = loadnpz('./inputData/ITCH_LOB/bestPrice/Bid_' + name + '.npz')
                bestPriceAlternate = loadnpz('./inputData/ITCH_LOB/bestPrice/Ask_' + name + '.npz').astype(int)
                oType = "B"
            else:
                LOB_share = loadnpz('./inputData/ITCH_LOB/SellLOB/' + name + '.npz', allow_pickle=True)
                bestPrice0 = loadnpz('./inputData/ITCH_LOB/bestPrice/Ask_' + name + '.npz')
                bestPriceAlternate = loadnpz('./inputData/ITCH_LOB/bestPrice/Bid_' + name + '.npz').astype(int)
                oType = "S"
            if loopNumber > 0:
                cancelations = loadnpz('./recursive/' + str(loopNumber) + '/choosenCancels/major/' + oType + '_' + name + '.npz', allow_pickle=True)
                empty_cancels = loadnpz('./recursive/' + str(loopNumber) + '/choosenCancels/empty/' + oType + '_' + name + '.npz', allow_pickle=True)

            data = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz', allow_pickle=True)
            if isBuy:
                price = data[data[:, 4] == "B", 2].astype(int)
                priceAlt = data[data[:, 4] == "S", 2].astype(int)
            else:
                price = data[data[:, 4] == "S", 2].astype(int)
                priceAlt = data[data[:, 4] == "B", 2].astype(int)
            priceUnique = np.unique(price)
            priceUniqueAlt = np.unique(priceAlt)
            #bestPriceVals = priceUnique[bestPrice0]
            bestPriceAlt = priceUniqueAlt[bestPriceAlternate]

            _, milisecondArgs = np.unique(data[:, 0][-1::-1], return_index=True)
            milisecondArgs = data.shape[0] - 1 - milisecondArgs
            milisecondReverse = np.zeros(data.shape[0])
            milisecondReverse[milisecondArgs] = 1

            size1 = data.shape[0]
            argsBuy = np.argwhere(data[:, 4] == oType)[:, 0]
            bestPrice = bestPrice0[data[:, 4] == oType]
            data = data[data[:, 4] == oType]
            newSaleFull = []
            doneTemp = False
            #prevBelowExecute = False

            exeArgs, exePrice = findMinorExecute(name, isBuy)

            times1 = 0
            times2 = 0
            times3 = 0
            times4 = 0
            times5 = 0

            #print (time.time() - time1)
            #time1 = time.time()

            for a in range(0, len(LOB_share)):
                time1_1 = time.time()
                time3_1 = time.time()
                priceDolars = priceUnique[a]
                if isBuy:
                    exeArgs1 = exeArgs[exePrice <= a]
                else:
                    exeArgs1 = exeArgs[exePrice >= a]
                #exeArgs1 = exeArgs[exePrice == a]
                timePoint = -1
                newSaleTime = [np.array([-1]).astype(int)]
                passList = np.array([-1]).astype(int)

                time3_2 = time.time() - time3_1
                times3 += time3_2

                prevBelowExecute1 = False

                for b1 in range(1, len(LOB_share[a][0])):
                    time2_1 = time.time()

                    b = len(LOB_share[a][0]) - b1 - 1
                    changeFull = data[LOB_share[a][0][b+1]]
                    orderNum = int(changeFull[6])
                    orderType = changeFull[3]
                    timeNow = int(changeFull[0])



                    if (orderType == "B") or (orderType == "S"):
                        newList = passList[:-1]
                    else:
                        newSaleTime[b1-1]
                        if (orderType == "C") or (orderType == "E"):
                            newList = np.copy(passList)
                            if orderType == 'E':
                                done1 = True
                                newList[0] = argsBuy[LOB_share[a][0][b+1]]
                        elif (orderType == "D") or (orderType == "F"):
                            arg = np.argwhere(np.array(LOB_share[a][1][b])[:, 0] == orderNum)[0, 0] + 1 #+1 cuz in front of all orders is position 0.
                            newList = np.zeros(passList.shape[0]+1)
                            if orderType == "D":
                                newList[arg:] = passList[arg-1:]
                                newList[:arg] = passList[:arg]
                            else:
                                done1 = True

                                if arg == np.array(LOB_share[a][1][b]).shape[0]: #Remember, the +1 in the definition of arg makes this work.
                                    newList[:] = argsBuy[LOB_share[a][0][b + 1]]
                                else:
                                    newList[:arg] = argsBuy[LOB_share[a][0][b + 1]]
                                    newList[arg:] = passList[arg-1:]

                                #newList[:(arg+1)] = argsBuy[LOB_share[a][0][b + 1]]
                                #newList[(arg+1):] = passList[arg:]

                    newSaleTime.append(np.copy(newList))


                    time4 = time.time()

                    cancelationTime = size1 * 2 #basicaly infinity
                    executionTime = size1 * 2 #basicaly infinity
                    if len(LOB_share[a][1][b]) == 0:
                        if LOB_share[a][0][b] != LOB_share[a][0][b+1]: #There is a case of both zero. This results in the automatic start at 0 plus a genuine start at 0 (the first order).
                            start1, end1 = argsBuy[LOB_share[a][0][b]], argsBuy[LOB_share[a][0][b+1]]
                            #start1, end1 = LOB_share[a][0][b], LOB_share[a][0][b+1]
                            exeArgs2 = exeArgs1[exeArgs1 >= start1]
                            exeArgs2 = exeArgs2[exeArgs2 < end1]

                            if exeArgs2.shape[0] != 0:
                                executionTime = np.min(exeArgs2)

                            if loopNumber > 0:#recursiveCancels:
                                if empty_cancels[a][b] != -1:
                                    cancelationTime = empty_cancels[a][b]


                            time5 = time.time()

                            zeroCancels = milisecondReverse[start1:end1]
                            if 1 in zeroCancels:
                                cancelationTime2 = np.min(np.argwhere(zeroCancels == 1)[:, 0]) + start1 #< -1
                                cancelationTime = min(cancelationTime, cancelationTime2)

                            priceBelowAlt = bestPriceAlt[start1:end1] - priceDolars
                            if not isBuy:
                                priceBelowAlt = priceBelowAlt * -1
                            if np.min(priceBelowAlt) <= 0:
                                executionTime2 = np.min(np.argwhere(priceBelowAlt <= 0)[:, 0]) + start1
                                executionTime = min(executionTime, executionTime2)




                            if min(executionTime, cancelationTime) != size1 * 2:
                                if executionTime <= cancelationTime:
                                    newList = np.array([executionTime])
                                    #newList = np.array([-2])
                                else:
                                    newList = np.array([-1])

                            times5 += time.time() - time5




                    if loopNumber > 0:#recursiveCancels:
                        if len(cancelations[a]) > 0:
                            if len(cancelations[a][b]) > 0:
                                newList[cancelations[a][b]] = -1

                    passList = newList


                    '''
                    for d in range(0, len(newSaleTime[-1])):
                        if newSaleTime[-1][d] != -1:
                            if newSaleTime[-1][d] < LOB_share[a][0][b]:
                                print (a)
                                print (newSaleTime[-1])
                                print (LOB_share[a][0][b])
                                print ("Error!")
                                quit()
                    '''
                    times4 += time.time() - time4
                    time2_2 = time.time() - time2_1
                    times2 += time2_2
                newSaleFull.append(newSaleTime)
                time1_2 = time.time() - time1_1
                times1 += time1_2

                #print ("A")
                #print (times1)
                #print (times2)
                #print (times3)
                #print (times4)
                #print (times5)
            #print ("U")

            #print (time.time() - time1)
            #quit()
            if True:
                np.savez_compressed('./recursive/' + str(loopNumber) + '/orderTransTimes/' + oType + '_' + name + '.npz', newSaleFull)
            #quit()

    #saveComponent(0)
    #quit()
    multiProcessor(saveComponent, filesRun[0], filesRun[1], 4, doPrint = True)


#findLOBpossibleTrans(0)
#quit()


def findFullMinorExecute():

    def saveComponent(nameNum):
        names = giveFNames()
        name = names[nameNum]
        print (name)


        for isBuy in [True, False]:
            priceEx, argsEx = findMinorExecute(name, isBuy)
            if isBuy:
                YData = loadnpz('./inputData/ITCH_LOB/allOrderXY/Y_Bid_' + name + '.npz')#[10000000:10010000]
                finalData = loadnpz('./inputData/ITCH_LOB/allOrderXY/X_Bid_' + name + '.npz')
                LOB_share = loadnpz('./inputData/ITCH_LOB/BuyLOB/' + name + '.npz', allow_pickle=True)
                oType = 'B'
                bestPrice = loadnpz('./inputData/ITCH_LOB/bestPrice/Bid_' + name + '.npz')
                bestPriceAlternate = loadnpz('./inputData/ITCH_LOB/bestPrice/Ask_' + name + '.npz').astype(int)
            else:
                YData = loadnpz('./inputData/ITCH_LOB/allOrderXY/Y_Ask_' + name + '.npz')
                finalData = loadnpz('./inputData/ITCH_LOB/allOrderXY/X_Ask_' + name + '.npz')
                LOB_share = loadnpz('./inputData/ITCH_LOB/SellLOB/' + name + '.npz', allow_pickle=True)
                oType = 'S'
                bestPrice = loadnpz('./inputData/ITCH_LOB/bestPrice/Ask_' + name + '.npz')
                bestPriceAlternate = loadnpz('./inputData/ITCH_LOB/bestPrice/Bid_' + name + '.npz').astype(int)

            #print (loadnpz('./inputData/Processed/minorCancelExec/Bid_' + name + ".npz").shape)
            #print (finalData.shape)
            #quit()

            data = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz', allow_pickle=True)
            if isBuy:
                price = data[data[:, 4] == "B", 2].astype(int)
                priceAlt = data[data[:, 4] == "S", 2].astype(int)
            else:
                price = data[data[:, 4] == "S", 2].astype(int)
                priceAlt = data[data[:, 4] == "B", 2].astype(int)
            priceUnique = np.unique(price)
            priceUniqueAlt = np.unique(priceAlt)
            #bestPriceVals = priceUnique[bestPrice0]
            bestPriceAlt = priceUniqueAlt[bestPriceAlternate]

            _, milisecondArgs = np.unique(data[:, 0][-1::-1], return_index=True)
            milisecondArgs = data.shape[0] - 1 - milisecondArgs

            cancels_mil = np.zeros(data.shape[0])
            cancels_mil[milisecondArgs] = 1
            cancels = cancels_mil

            size1 = data.shape[0]
            argsBuy = np.argwhere(data[:, 4] == oType)[:, 0]

            lobPos = YData[:, 0]
            position = finalData[:, 0]
            price = YData[:, 1]
            #quePos = YData[:, 2]

            if isBuy:
                neg1 = -1

            #results = np.zeros((0, 2)).astype(int)
            results = np.zeros((100000000, 2)).astype(int)
            count1 = 0

            priceU = np.unique(price) #Just adding a little speed up
            price_argsort, indicesStart, indicesEnd = fastAllArgwhere(price)



            times1 = 0
            times2 = 0
            times3 = 0
            times4 = 0
            times5 = 0
            #print (len(LOB_share))
            for a in range(0, len(LOB_share)):
                #for a in range(852, 853):
                #print ("A", a)
                if a in priceU: #Just adding a little speed up
                    priceDolars = priceUnique[a]

                    arg1 = np.argwhere(priceU == a)[0][0]
                    start_1, end_1 = indicesStart[arg1], indicesEnd[arg1] + 1
                    argsGood1 = price_argsort[start_1:end_1]
                    #argsGood1 = np.argwhere(price == a)[:, 0]

                    time1_1 = time.time()

                    if argsGood1.shape[0] != 0:

                        time3_1 = time.time()

                        if isBuy:
                            argsEx1 = argsEx[priceEx <= a]
                        else:
                            argsEx1 = argsEx[priceEx >= a]
                        #argsEx1 = argsEx[priceEx == a]

                        time3_2 = time.time() - time3_1
                        times3 += time3_2

                        '''
                        cancels = (a - bestPrice) * neg1
                        #cancels[cancels > -2] = 0
                        #cancels[cancels <= -2] = 1
                        cancels[cancels > -1] = 0
                        cancels[cancels <= -1] = 1
                        cancels_mil = np.zeros(cancels.shape[0])
                        cancels_mil[milisecondArgs] = 1
                        cancels = cancels * cancels_mil
                        cancels = cancels.astype(int)
                        '''

                        priceBelowAlt = bestPriceAlt - priceDolars
                        if not isBuy:
                            priceBelowAlt = priceBelowAlt * -1
                        if np.min(priceBelowAlt) < 0:
                            argsEx1_fromBelow = np.argwhere(priceBelowAlt <= 0)[:, 0]
                            argsEx1 = np.concatenate((argsEx1, argsEx1_fromBelow))
                            argsEx1 = np.sort(np.unique(argsEx1))

                        #argsEx2_temp = argsEx[priceEx == a - 1]
                        #if argsEx2_temp.shape[0] != 0:
                        #    if np.mean(cancels[argsEx2_temp]) != 1:
                        #        print (np.mean(cancels[argsEx2_temp]))
                        #        quit()

                        #price_argsort, indicesStart, indicesEnd = fastAllArgwhere(price)

                        lobPos_1 = lobPos[argsGood1]

                        #lobPos_1_U = np.unique(lobPos_1)
                        #lobPos_argsort, lob_indicesStart, lob_indicesEnd = fastAllArgwhere(lobPos_1)


                        time4_1 = time.time()

                        goodB = []
                        starts1 = []
                        ends1 = []
                        for b in range(0, len(LOB_share[a][0]) - 1):
                            if len(LOB_share[a][1][b]) == 0:
                                goodB.append(b)
                                start1, end1 = argsBuy[LOB_share[a][0][b]], argsBuy[LOB_share[a][0][b+1]]
                                starts1.append(start1)
                                ends1.append(end1)
                        starts1 = np.array(starts1)
                        ends1 = np.array(ends1)
                        goodB = np.array(goodB)
                        argsExIndex1 = stefan_nextAfter(starts1*2, (argsEx1*2)+ 1)
                        argsExIndex2 = stefan_firstBefore(ends1*2, (argsEx1*2)+ 1)


                        argsGood1 = argsGood1[np.isin(lobPos_1, goodB)]
                        lobPos_1 = lobPos[argsGood1]

                        time4_2 = time.time() - time4_1
                        times4 += time4_2


                        #for b in range(0, len(LOB_share[a][0]) - 1):
                        for b0 in range(0, goodB.shape[0]):
                            b = goodB[b0]
                            time2_1 = time.time()
                            if len(LOB_share[a][1][b]) == 0:
                                start1, end1 = argsBuy[LOB_share[a][0][b]], argsBuy[LOB_share[a][0][b+1]]
                                if start1 != end1:
                                    argsGood2 = argsGood1[lobPos_1 == b]
                                    #print (lobPos[argsGood1], b)
                                    #print ("R", argsGood2.shape)

                                    if argsGood2.shape[0] != 0:
                                        argsGood2 = argsGood2[np.argsort(position[argsGood2])]
                                        argsAfter1 = np.array([])
                                        argsAfter2 = np.array([])

                                        #argsEx2 = argsEx1[np.logical_and(argsEx1 >= start1, argsEx1 < end1)]
                                        argsEx2 = argsEx1[argsExIndex1[b0]:argsExIndex2[b0]+1]

                                        #if argsEx2.shape[0] != 0:
                                        #    if np.mean(np.abs(argsEx2 - argsEx2_c)) != 0:
                                        #        print (np.mean(np.abs(argsEx2 - argsEx2_c)))
                                        #        quit()
                                        if argsEx2.shape[0] != 0:
                                            time5_1 = time.time()

                                            pos1 = position[argsGood2]
                                            pos1 = pos1[pos1 <= np.max(argsEx2)]
                                            argsAfter1 = stefan_nextAfter(pos1, argsEx2)
                                            argsAfter1 = argsEx2[argsAfter1]

                                            time5_2 = time.time() - time5_1
                                            times5 += time5_2



                                        if 1 in cancels[start1:end1]:
                                            cancels1 = cancels[start1:end1]
                                            cancels1 = np.argwhere(cancels1 == 1)[:, 0] + start1
                                            pos2 = position[argsGood2]
                                            pos2 = pos2[pos2 <= np.max(cancels1)]
                                            argsAfter2 = stefan_nextAfter(pos2, cancels1)
                                            argsAfter2 = cancels1[argsAfter2]

                                        max1 = max(argsAfter1.shape[0], argsAfter2.shape[0])
                                        if max1 != 0:
                                            argsAfter = np.zeros(max1)
                                            min1 = min(argsAfter1.shape[0], argsAfter2.shape[0])
                                            if min1 != 0:
                                                argsChoice = np.argmin(np.array([argsAfter1[:min1] * 2, (argsAfter2[:min1]*2) + 1 ]), axis=0)
                                                argsAfter[:min1] = np.array([argsAfter1[:min1], np.zeros(min1) - 1 ])[argsChoice, np.arange(min1)]
                                            #argsAfter[:min1] = np.min(np.array([argsAfter1[:min1], argsAfter2[:min1]]), axis=0)
                                            if argsAfter1.shape[0] > argsAfter2.shape[0]:
                                                argsAfter[min1:] = argsAfter1[min1:]
                                            if argsAfter1.shape[0] < argsAfter2.shape[0]:
                                                argsAfter[min1:] = -1 #argsAfter2[min1:]

                                            #print (argsGood2[:max1].shape, argsAfter.shape)
                                            result1 = np.array([argsGood2[:max1], argsAfter]).T

                                            #results = np.concatenate((results, result1))
                                            count_size = result1.shape[0]
                                            results[count1:count1+count_size] = result1
                                            count1 += count_size
                        time2_2 = time.time() - time2_1
                        times2 += time2_2
                    time1_2 =  time.time() - time1_1
                    times1 += time1_2

            results = results[:count1]

                    #print ('A')
                    #print (times1)
                    #print (times2)
                    #print (times3)
                    #print (times4)
                    #print (times5)

            #print (np.mean(results))
            #np.save('./temporary/cancelResults/2.npy', results)
            #quit()

            if True:
                if isBuy:
                    np.savez_compressed('./inputData/Processed/minorCancelExec/Bid_' + name + ".npz", results)
                else:
                    np.savez_compressed('./inputData/Processed/minorCancelExec/Ask_' + name + ".npz", results)
            print ("U")

    #saveComponent(0)
    #quit()
    #multiProcessor(saveComponent, 0, 15, 4, doPrint = True)
    multiProcessor(saveComponent, filesRun[0], filesRun[1], 2, doPrint = True)

#findFullMinorExecute()
#quit()

def saveCancelationInput(loopNumber, filesRun=(0, 15)):


    def saveComponent(nameNum):
        #print ("A")
        names = giveFNames()
        name = names[nameNum]
        print (name)
        for isBuy in [True, False]:
            time1 = time.time()
            if isBuy:
                if loopNumber == 0:
                    YData = loadnpz('./inputData/ITCH_LOB/allOrderXY/Y_Bid_' + name + '.npz')#[10000000:10010000]
                    finalData = loadnpz('./inputData/ITCH_LOB/allOrderXY/X_Bid_' + name + '.npz')
                minorCancelExec = loadnpz('./inputData/Processed/minorCancelExec/Bid_' + name + ".npz")
                oType = "B"
            else:
                if loopNumber == 0:
                    YData = loadnpz('./inputData/ITCH_LOB/allOrderXY/Y_Ask_' + name + '.npz')
                    finalData = loadnpz('./inputData/ITCH_LOB/allOrderXY/X_Ask_' + name + '.npz')
                minorCancelExec = loadnpz('./inputData/Processed/minorCancelExec/Ask_' + name + ".npz")
                oType = "S"

            if loopNumber > 0:
                finalData = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')
                YData = loadnpz('./recursive/0/orderData/withCancels/Y_' + oType + '_' + name + '.npz')
                argsGood = loadnpz('./recursive/0/orderData/withCancels/argsGood_' + oType + '_' + name + '.npz')
            else:
                size2 = finalData.shape[0]
                argsGood = np.arange(finalData.shape[0])[finalData[:, 1] < 18]
                argsGood = argsGood[finalData[argsGood, 1] != -1]
                finalData = finalData[argsGood]
                YData = YData[argsGood]

            #print (time.time() - time1)
            #time1 = time.time()


            max1 = np.max(YData[:, 0]) + 1
            max2 = np.max(YData[:, 1]) + 1
            vals = YData[:, 0] + (max1 * (YData[:, 1] + (max2 * YData[:, 2])))
            vals_unique, vals_index, vals_inverse = np.unique(vals, return_index=True, return_inverse=True)


            #print (time.time() - time1)
            #time1 = time.time()

            size2 = int(max(np.max(argsGood), np.max(minorCancelExec[:, 0])) + 1)
            reverser = np.zeros(size2) - 1
            reverser[argsGood] = np.arange(argsGood.shape[0])
            minorCancelExec[:, 0] = reverser[minorCancelExec[:, 0].astype(int)]
            minorCancelExec = minorCancelExec[minorCancelExec[:, 0] != -1]

            orderTransTimes = loadnpz('./recursive/' + str(loopNumber) + '/orderTransTimes/' + oType + '_' + name + ".npz", allow_pickle=True)
            #orderTransTimes = loadnpz('./recursive/' + str(1) + '/orderTransTimes/' + oType + '_' + name + ".npz", allow_pickle=True)

            #print (time.time() - time1)
            #time1 = time.time()

            #Yargs = np.zeros(YData.shape[0]).astype(int) - 1
            Yargs = np.zeros(vals_unique.shape[0]).astype(int) - 1
            for a in range(0, vals_unique.shape[0]):
                Yargs[a] = orderTransTimes[YData[vals_index[a], 1]][-YData[vals_index[a], 0] - 1][YData[vals_index[a], 2]]
            Yargs = Yargs[vals_inverse]

            #print (np.argwhere(Yargs == -1).shape)
            #print (np.argwhere(Yargs == -2).shape)
            #print (np.argwhere(Yargs >= 0).shape)
            #quit()

            diff = Yargs[Yargs != -1] - finalData[Yargs != -1][:, 0]
            #print (diff[diff < 0].shape)


            #print (time.time() - time1)
            #time1 = time.time()

            #for a in range(0, YData.shape[0]):
            #    Yargs[a] = orderTransTimes[YData[a, 1]][-YData[a, 0] - 1][YData[a, 2]]

            #list1 = list(np.arange(YData.shape[0]).astype(int))
            #Yargs2 = [orderTransTimes[YData[a, 1]][-YData[a, 0] - 1][YData[a, 2]] for a in list1]
            #print (time.time() - time1)
            #quit()
            #print (Yargs[Yargs != -1].shape)

            Yargs[minorCancelExec[:, 0].astype(int)] = minorCancelExec[:, 1]

            diff = Yargs[Yargs != -1] - finalData[Yargs != -1][:, 0]
            #print (diff[diff < 0].shape)
            #quit()

            #print (np.mean(Yargs))
            #quit()
            #print (Yargs[Yargs != -1].shape)

            goodSubArgs = np.arange(Yargs.shape[0])[Yargs != -1]
            argsGood = argsGood[goodSubArgs]
            finalData = finalData[goodSubArgs]
            YData = YData[goodSubArgs]
            Yargs = np.copy(Yargs[goodSubArgs]).astype(int)

            #print (Yargs.shape)

            #print (YData.shape)
            #quit()
            #a = 14700
            #print (orderTransTimes[YData[a, 1]][-YData[a, 0] - 1][YData[a, 2]])

            #print (time.time() - time1)
            #print ("U")
            #quit()

            #print (time.time() - time1)
            #time1 = time.time()


            #quit()
            if True:
                np.savez_compressed('./recursive/' + str(loopNumber) + '/orderData/initial/Y_' + oType + '_' + name + '.npz', YData)
                np.savez_compressed('./recursive/' + str(loopNumber) + '/orderData/initial/Trans_' + oType + '_' + name + '.npz', Yargs)
                np.savez_compressed('./recursive/' + str(loopNumber) + '/orderData/initial/X_' + oType + '_' + name + '.npz', finalData)
                #if loopNumber == 0:
                #    np.savez_compressed('./recursive/' + str(loopNumber) + '/orderData/initial/argsGood_' + oType + '_' + name + '.npz', goodSubArgs)
                #else:
                np.savez_compressed('./recursive/' + str(loopNumber) + '/orderData/initial/argsGood_' + oType + '_' + name + '.npz', argsGood)#updated with loaded finalData withCancels

                if loopNumber == 0:
                    np.savez_compressed('./recursive/' + str(loopNumber) + '/orderData/withCancels/Y_' + oType + '_' + name + '.npz', YData)
                    np.savez_compressed('./recursive/' + str(loopNumber) + '/orderData/withCancels/Trans_' + oType + '_' + name + '.npz', Yargs)
                    np.savez_compressed('./recursive/' + str(loopNumber) + '/orderData/withCancels/X_' + oType + '_' + name + '.npz', finalData)
                    #np.savez_compressed('./recursive/' + str(loopNumber) + '/orderData/withCancels/argsGood_' + oType + '_' + name + '.npz', goodSubArgs)
                    np.savez_compressed('./recursive/' + str(loopNumber) + '/orderData/withCancels/argsGood_' + oType + '_' + name + '.npz', argsGood)
                print ('U')



    #multiProcessor(saveComponent, 1, 8, 4, doPrint = True)
    #saveComponent(0)
    #quit()
    multiProcessor(saveComponent, filesRun[0], filesRun[1], 4, doPrint = True)


#saveCancelationInput(0)
#quit()

def applyMinorCancelation(loopNumber, filesRun=(0, 15)):

    def saveComponent(nameNum):
        names = giveFNames()
        name = names[nameNum]
        print (name)

        for isBuy in [True, False]:#[True, False]:

            if isBuy:
                LOB_share = loadnpz('./inputData/ITCH_LOB/BuyLOB/' + name + '.npz', allow_pickle=True)
                bestPrice = loadnpz('./inputData/ITCH_LOB/bestPrice/Bid_' + name + '.npz')

                shift = 10
                oType = "B"
            else:
                LOB_share = loadnpz('./inputData/ITCH_LOB/SellLOB/' + name + '.npz', allow_pickle=True)
                bestPrice = loadnpz('./inputData/ITCH_LOB/bestPrice/Ask_' + name + '.npz')

                shift = -10
                oType = "S"

            finalData_1 = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')
            YData_1 = loadnpz('./recursive/0/orderData/withCancels/Y_' + oType + '_' + name + '.npz')
            Yargs_1 = loadnpz('./recursive/0/orderData/withCancels/Trans_' + oType + '_' + name + '.npz')

            YData_2 = loadnpz('./recursive/' + str(loopNumber) + '/orderData/initial/Y_' + oType + '_' + name + '.npz')
            Yargs_2 = loadnpz('./recursive/' + str(loopNumber) + '/orderData/initial/Trans_' + oType + '_' + name + '.npz')
            finalData_2 = loadnpz('./recursive/' + str(loopNumber) + '/orderData/initial/X_' + oType + '_' + name + '.npz')

            argsGood_2 = loadnpz('./recursive/' + str(loopNumber) + '/orderData/initial/argsGood_' + oType + '_' + name + '.npz')


            output = loadnpz('./recursive/' + str(loopNumber) + '/outputCancels/' + oType + '_' + name + '.npz')

            data = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz', allow_pickle=True)
            argsBuy = np.argwhere(data[:, 4] == oType)[:, 0]
            del data

            finalData_1 = finalData_1[output == 1]
            YData_1 = YData_1[output == 1]

            price_1 = YData_1[:, 1]
            price_2 = YData_2[:, 1]
            lobPos_1 = YData_1[:, 0]
            lobPos_2 = YData_2[:, 0]
            quePos_1 = YData_1[:, 2]
            quePos_2 = YData_2[:, 2]
            pos_1 = finalData_1[:, 0]
            pos_2 = finalData_2[:, 0]

            price_1_U = np.unique(price_1)
            price_2_U = np.unique(price_2)
            price_1_argsort, starts1, ends1 = fastAllArgwhere(price_1)
            price_2_argsort, starts2, ends2 = fastAllArgwhere(price_2)

            times1 = 0
            times2 = 0
            times3 = 0
            times4 = 0


            '''
            lobPos_1 = lobPos[argsGood1]

            #lobPos_1_U = np.unique(lobPos_1)
            #lobPos_argsort, lob_indicesStart, lob_indicesEnd = fastAllArgwhere(lobPos_1)


            time4_1 = time.time()

            goodB = []
            starts1 = []
            ends1 = []
            for b in range(0, len(LOB_share[a][0]) - 1):
                if len(LOB_share[a][1][b]) == 0:
                    goodB.append(b)
                    start1, end1 = argsBuy[LOB_share[a][0][b]], argsBuy[LOB_share[a][0][b+1]]
                    starts1.append(start1)
                    ends1.append(end1)
            starts1 = np.array(starts1)
            ends1 = np.array(ends1)
            goodB = np.array(goodB)
            argsExIndex1 = stefan_nextAfter(starts1*2, (argsEx1*2)+ 1)
            argsExIndex2 = stefan_firstBefore(ends1*2, (argsEx1*2)+ 1)


            argsGood1 = argsGood1[np.isin(lobPos_1, goodB)]
            lobPos_1 = lobPos[argsGood1]

            time4_2 = time.time() - time4_1
            times4 += time4_2


            #for b in range(0, len(LOB_share[a][0]) - 1):
            for b0 in range(0, goodB.shape[0]):
                b = goodB[b0]
                time2_1 = time.time()
                if len(LOB_share[a][1][b]) == 0:
                    start1, end1 = argsBuy[LOB_share[a][0][b]], argsBuy[LOB_share[a][0][b+1]]
                    if start1 != end1:
                        argsGood2 = argsGood1[lobPos_1 == b]
                        #print (lobPos[argsGood1], b)
                        #print ("R", argsGood2.shape)
            '''

            canceledArgs = []
            for a in range(0, len(LOB_share)): #make 0 #990
                time1 = time.time()
                if a in price_1_U:#price_1:
                    if a in price_2_U:#price_2:
                        price_arg1 = np.argwhere(price_1_U == a)[0][0]
                        price_arg2 = np.argwhere(price_2_U == a)[0][0]
                        start1, end1 = starts1[price_arg1], ends1[price_arg1] + 1
                        start2, end2 = starts2[price_arg2], ends2[price_arg2] + 1
                        argsGood_1_1 = price_1_argsort[start1:end1]
                        argsGood_2_1 = price_2_argsort[start2:end2]
                        #argsGood_1_1 = np.argwhere(price_1 == a)[:, 0]
                        #argsGood_2_1 = np.argwhere(price_2 == a)[:, 0]
                        if (argsGood_1_1.shape[0] != 0) and (argsGood_2_1.shape[0] != 0):
                            UlobPos_1 = np.unique(lobPos_1[argsGood_1_1])
                            UlobPos_2 = np.unique(lobPos_2[argsGood_2_1])

                            #New Sep 23
                            UlobPos_intersect = np.intersect1d(UlobPos_1, UlobPos_2)
                            argsGood_1_1 = argsGood_1_1[np.isin(lobPos_1[argsGood_1_1], UlobPos_intersect)]
                            argsGood_2_1 = argsGood_2_1[np.isin(lobPos_2[argsGood_2_1], UlobPos_intersect)]

                            for b in range(0, len(LOB_share[a][0])-1): #make 0 #22
                                time2 = time.time()
                                if b in UlobPos_1:
                                    if b in UlobPos_2:

                                        time4 = time.time()
                                        argsGood_1_2 = argsGood_1_1[lobPos_1[argsGood_1_1] == b]
                                        argsGood_2_2 = argsGood_2_1[lobPos_2[argsGood_2_1] == b]
                                        times4 += time.time() - time4

                                        #if (a == 990) and (b == 2):
                                        #    print ("Good-1")
                                        #if (a == 990) and (b == 3):
                                        #    print ("Bad")
                                        #    quit()


                                        #print (LOB_share[906][1][94])
                                        #print (LOB_share[904][1][442])
                                        #print (LOB_share[903][1][370])
                                        #print (LOB_share[879][1][1367])
                                        #print (LOB_share[879][1][67])
                                        #quit()

                                        time3 = time.time()
                                        if len(LOB_share[a][1][b]) != 0:
                                            #if (a == 990) and (b == 22):
                                            #    print ("Good0")
                                            #time_1 = time.time()
                                            for c in range(0, len(LOB_share[a][1][b])+1):
                                                if c in quePos_1[argsGood_1_2]:
                                                    if c in quePos_2[argsGood_2_2]:
                                                        argsGood_2_3 = argsGood_2_2[quePos_2[argsGood_2_2] == c]
                                                        if argsGood_2_3.shape[0] != 0:
                                                            maxCancel = np.max(pos_1[argsGood_1_2[quePos_1[argsGood_1_2] == c]])
                                                            if np.min(pos_2[argsGood_2_3]) <= maxCancel:
                                                                argsGood_2_3 = argsGood_2_3[pos_2[argsGood_2_3] <= maxCancel]
                                                                Yargs_2[argsGood_2_3] = -1
                                        #else:
                                        times3 += time.time() - time3

                                        if False:#len(LOB_share[a][1][b]) == 0:
                                            #if (a == 990) and (b == 22):
                                            #    print ("Good1")


                                            #22 990
                                            argsGood_1_2 = argsGood_1_2[np.argsort(pos_1[argsGood_1_2])]
                                            argsGood_2_2 = argsGood_2_2[np.argsort(pos_2[argsGood_2_2])]

                                            pos_1_2 = pos_1[argsGood_1_2]
                                            pos_2_2 = pos_2[argsGood_2_2]
                                            maxCancel = np.max(pos_1_2)
                                            if np.min(pos_2_2) <= maxCancel:

                                                #if (a == 990) and (b == 22):
                                                #    print ("Good2")

                                                argsGood_2_3 = argsGood_2_2[pos_2_2 <= maxCancel]
                                                Yargs2_1 = Yargs_2[argsGood_2_3]

                                                if np.min(Yargs2_1) < LOB_share[a][0][b+1]:

                                                    #if (a == 990) and (b == 22):
                                                    #    print ("Good3")

                                                    pos_2_2 = pos_2[argsGood_2_3]
                                                    #next1 = stefan_nextAfter(pos_2_2, pos_1_2)
                                                    next1 = stefan_nextAfter(pos_2_2 * 2, (pos_1_2 * 2)+ 1)
                                                    argsGood_2_3 = argsGood_2_3[next1 != -1]
                                                    if argsGood_2_3.shape[0] != 0:

                                                        #if (a == 990) and (b == 22):
                                                            #print ("Good4")
                                                            #if 1551 in pos_1_2:
                                                            #    print ("A")
                                                            #else:
                                                            #    print ("B")
                                                            #arg2 = np.argwhere(pos_2_2 == 1551)[0][0]

                                                            #print (np.mean(np.abs(pos_1_2 - np.sort(pos_1_2))))
                                                            #print (1551 in pos_1_2)
                                                            #print (next1[arg2])
                                                            #print (pos_1_2[next1[arg2]])
                                                            #quit()

                                                        #[[1551   16    0]
                                                        # [1563   16    0]]
                                                        #[[ 22 990   0]
                                                        # [ 42 991   0]]
                                                        next1 = next1[next1 != -1]
                                                        after1 = pos_1_2[next1]
                                                        Yargs2_1 = Yargs_2[argsGood_2_3]
                                                        #argsChoice = np.argmin(np.array( [(after1*2) + 1, Yargs2_1 * 2]))
                                                        argsChoice = np.argmin(np.array( [(after1*2) + 1, Yargs2_1 * 2]), axis=0)

                                                        #if (a == 990) and (b == 22):
                                                            #arg1 = np.argwhere(pos_2[argsGood_2_3] == 1551)[0][0]
                                                            #print (after1[arg1])
                                                            #print (argsChoice[arg1])
                                                            #print ("arg1")
                                                            #print (arg1)
                                                            #quit()
                                                        argsGood_2_3 = argsGood_2_3[argsChoice == 0]
                                                        Yargs_2[argsGood_2_3] = -1


                                                        #print (YData_2[argsGood_2_3])
                                                        #if (a == 990) and (b == 22):
                                                            #print (np.array( [(after1*2) + 1, Yargs2_1 * 2]))
                                                            #print (argsChoice)
                                                            #print (argsGood_2_3)
                                                            #print ("Good6")
                                                            #quit()

                                                else:

                                                    #if (a == 990) and (b == 22):
                                                    #    print ("Good5")

                                                    Yargs_2[argsGood_2_3] = -1


                                times2 += time.time() - time2
                times1 += time.time() - time1

                #print ("A")
                #print (times1)
                #print (times2)
                #print (times3)
                #print (times4)
            #quit()
                                        #if (a == 990) and (b == 22):
                                        #    print ("Good7")
                                            #quit()

            finalData_2 = finalData_2[Yargs_2 != -1]
            YData_2 = YData_2[Yargs_2 != -1]
            argsGood_2 = argsGood_2[Yargs_2 != -1]
            Yargs_2 = Yargs_2[Yargs_2 != -1]
            #print (Yargs_2.shape)

            #quit()
            np.savez_compressed('./recursive/' + str(loopNumber) + '/orderData/withCancels/Y_' + oType + '_' + name + '.npz', YData_2)
            np.savez_compressed('./recursive/' + str(loopNumber) + '/orderData/withCancels/Trans_' + oType + '_' + name + '.npz', Yargs_2)
            np.savez_compressed('./recursive/' + str(loopNumber) + '/orderData/withCancels/X_' + oType + '_' + name + '.npz', finalData_2)
            np.savez_compressed('./recursive/' + str(loopNumber) + '/orderData/withCancels/argsGood_' + oType + '_' + name + '.npz', argsGood_2)
            print ("U")

    if loopNumber > 0:
        #print ("T")
        #saveComponent(0)
        #quit()
        #for a in range(8, 9):
        #    saveComponent(a)
        #quit()
        multiProcessor(saveComponent, 32, filesRun[1], 4, doPrint = True)
        #multiProcessor(saveComponent, filesRun[0], filesRun[1], 4, doPrint = True)

#applyMinorCancelation(1, filesRun=(1, 15))
#quit()

def findSubsetProfit(loopNumber, filesRun=(0, 15)):
    def saveComponent(nameNum):
        names = giveFNames()
        #for name in names:
        name = names[nameNum]
        print (name)

        for isBuy in [True, False]:#[True, False]:#[True, False]:

            if isBuy:
                oType = "B"
            else:
                oType = "S"

            YData = loadnpz('./recursive/' + str(loopNumber) + '/orderData/withCancels/Y_' + oType + '_' + name + '.npz')
            Yargs = loadnpz('./recursive/' + str(loopNumber) + '/orderData/withCancels/Trans_' + oType + '_' + name + '.npz')
            finalData = loadnpz('./recursive/' + str(loopNumber) + '/orderData/withCancels/X_' + oType + '_' + name + '.npz')
            argsGood = loadnpz('./recursive/' + str(loopNumber) + '/orderData/withCancels/argsGood_' + oType + '_' + name + '.npz')

            namePart = ''
            if isBuy:
                namePart = namePart + "Bid_"
            else:
                namePart = namePart + "Ask_"
            namePart = namePart + name

            data = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz')
            if isBuy:
                argsBuy = np.argwhere(data[:, 4] == "B")[:, 0]
            else:
                argsBuy = np.argwhere(data[:, 4] == "S")[:, 0]
            prices = np.unique(data[argsBuy, 2].astype(int))
            originalPrice = prices[YData[:, 1]]

            argsExecute = np.argwhere(np.isin(data[:, 3], np.array(['E', 'F'])))[:, 0]
            if isBuy:
                argsExecute = argsExecute[data[argsExecute, 4] == 'B']
            else:
                argsExecute = argsExecute[data[argsExecute, 4] == 'S']
            reverser = np.zeros(data.shape[0]).astype(int)
            reverser[argsExecute] = 1
            reverser = np.cumsum(reverser)
            reverser[argsExecute] = reverser[argsExecute] - 1
            argsExecute = np.concatenate((argsExecute, np.array([data.shape[0]]))) #things after last argsExecute is out of array so gets kicked out of "known profit Args"
            argsExecute[reverser[Yargs]]

            Yargs = argsExecute[reverser[Yargs]] #This sounds up. This allows the trans profits data to be used. This is equivelent to simply delaying trying to sell (after a buy) until the next buy trans


            #transProfits = np.load('./recursive/' + str(loopNumber) + '/transProfits/final/transSalePrices_' + name + '.npy')
            #reverseTrans = np.load('./recursive/' + str(loopNumber) + '/transProfits/final/ReverseTrans_' + name + '.npy')

            transProfits = np.load('./recursive/' + str(loopNumber) + '/transProfits/final/transSalePrices_' + oType + '_' + name + '.npy')
            reverseTrans = np.load('./recursive/' + str(loopNumber) + '/transProfits/final/ReverseTrans_' + oType + '_' + name + '.npy')
            #transProfits = np.load('./recursive/' + str(0) + '/transProfits/final/transSalePrices_' + oType + '_' + name + '.npy')
            #reverseTrans = np.load('./recursive/' + str(0) + '/transProfits/final/ReverseTrans_' + oType + '_' + name + '.npy')

            #transProfits = np.load('./recursive/' + str(0) + '/transProfits/final/transSalePrices_' + oType + '_' + name + '.npy')
            #reverseTrans = np.load('./recursive/' + str(0) + '/transProfits/final/ReverseTrans_' + oType + '_' + name + '.npy')
            #transProfits = np.load('./recursive/' + str(0) + '/transProfits/final/transSalePrices_' + name + '.npy')
            #reverseTrans = np.load('./recursive/' + str(0) + '/transProfits/final/ReverseTrans_' + name + '.npy')

            knownProfitArgs = np.argwhere(np.isin(Yargs, transProfits[:, 0]))[:, 0]
            #print (knownProfitArgs.shape)
            #quit()
            Yargs = Yargs[knownProfitArgs]
            finalData = finalData[knownProfitArgs]
            YData = YData[knownProfitArgs] #New Aug 25
            originalPrice = originalPrice[knownProfitArgs]

            #print (originalPrice[:10])
            #print (np.mean(originalPrice))
            #originalPrice = data[Yargs.astype(int), 2].astype(int)
            #print (originalPrice[:10])
            #print (np.mean(originalPrice))
            #quit()

            Yargs_copy = np.copy(Yargs)

            Yargs = reverseTrans[Yargs.astype(int)]


            '''
            argsGood1 = argsGood[knownProfitArgs]
            output2 = loadnpz('./recursive/1/samples/prob_' + oType + '_' + name + '.npz')
            argsGood0 = loadnpz('./recursive/0/orderData/withCancels/argsGood_' + oType + '_' + name + '.npz')
            argsGood0 = argsGood0[output2[:, 0].astype(int)]
            validSubArgs = np.argwhere(np.isin(argsGood1, argsGood0))[:, 0]
            validYargs = Yargs.astype(int)[validSubArgs]
            #print (argsGood0.shape)
            #print (output2.shape)
            #quit()
            transProfits0 = np.load('./recursive/' + str(0) + '/transProfits/final/transSalePrices_' + oType + '_' + name + '.npy')
            transProfits1 = np.load('./recursive/' + str(1) + '/transProfits/final/transSalePrices_' + oType + '_' + name + '.npy')
            print ("U")
            print (np.mean(transProfits0[:, 1]))
            print (np.mean(transProfits1[:, 1]))
            print (np.mean(transProfits0[validYargs, 1]))
            print (np.mean(transProfits1[validYargs, 1]))
            '''


            YargsPrice = transProfits[Yargs.astype(int), 1]
            if isBuy:
                YargsPrice = YargsPrice - originalPrice
            else:
                YargsPrice = originalPrice - YargsPrice


            #print (YData.shape)
            #print (finalData.shape)

            #print ("Profits")
            #print (np.mean(YargsPrice))

            '''

            plt.hist(YargsPrice, bins=100)
            plt.show()

            YargsPrice = transProfits[Yargs.astype(int), 1] + transProfits[Yargs.astype(int), 2]
            print (np.mean(transProfits[Yargs.astype(int), 2]))
            if isBuy:
                YargsPrice = YargsPrice - originalPrice
            else:
                YargsPrice = originalPrice - YargsPrice
            print (np.mean(YargsPrice))
            '''
            #quit()


            #print ("A")
            #print (Yargs.shape)
            #print ("Profit 1", np.mean(Yargs))
            #print ("Profit 2", np.mean(Yargs[np.argsort(Yargs_copy)][1000:-1000]))
            #print ("Profit 3", np.mean(Yargs[np.argsort(Yargs_copy)][10000:-10000]))
            #quit()
            #print (np.mean(YargsPrice))
            #print (np.mean(loadnpz('./recursive/' + str(loopNumber) + '/profitData/Trans_' + oType + '_' + name + '.npz')))

            #quit()
            #plt.hist(Yargs, bins=100)
            #plt.scatter(Yargs_copy[0::1000], Yargs[0::1000])
            #plt.show()
            #quit()

            #print (np.mean(YargsPrice))
            #YargsPrice2 = loadnpz('./recursive/' + str(loopNumber) + '/profitData/Trans_' + oType + '_' + name + '.npz')
            #print (np.mean(YargsPrice2))
            #print (np.mean(np.abs(YargsPrice2 - YargsPrice)))
            #quit()

            if True:
                X_argsort = np.argsort(finalData[:, 0])
                #print (finalData.shape)
                max1 = np.max(finalData[:, 0])
                #print (finalData.shape)
                #print (np.argwhere(finalData[:, 0] < (max1 // 10)).shape)
                #print (np.argwhere(finalData[:, 0] > (max1 * 0.95)).shape)
                #quit()
                X_argsort = X_argsort[(X_argsort.shape[0] // 10):-(X_argsort.shape[0] // 20)]
                #print (X_argsort.shape)
                np.savez_compressed('./recursive/' + str(loopNumber) + '/profitData/Y_' + oType + '_' + name + '.npz', YData[X_argsort])
                np.savez_compressed('./recursive/' + str(loopNumber) + '/profitData/Trans_' + oType + '_' + name + '.npz', YargsPrice[X_argsort])
                np.savez_compressed('./recursive/' + str(loopNumber) + '/profitData/X_' + oType + '_' + name + '.npz', finalData[X_argsort])
                np.savez_compressed('./recursive/' + str(loopNumber) + '/profitData/argsGood_' + oType + '_' + name + '.npz', argsGood[knownProfitArgs][X_argsort])


    #for a in range(0, 15):
    #    saveComponent(a)
    #quit()
    #for a in range(0, 15):
    #    saveComponent(a)
    #quit()
    #saveComponent(0)
    #quit()
    #if filesRun == -1:
    #    filesRun = 15
    #multiProcessor(saveComponent, 20, filesRun[1], 4, doPrint = True)
    multiProcessor(saveComponent, filesRun[0], filesRun[1], 4, doPrint = True)
    #quit()

#findSubsetProfit(0)
#quit()


def XpriceToOrder(X, isBuy):

    buyAr = np.zeros(X.shape[0]).astype(int)
    if isBuy:
        buyAr = buyAr + 1

    order = np.zeros((X.shape[0], 19*2, 2))
    relativePrice = X[:, 1] + 1
    relativePrice = relativePrice * (1 - (buyAr*2))
    relativePrice = relativePrice - buyAr
    relativePrice = relativePrice + 18 + 1
    order[np.arange(relativePrice.shape[0]), relativePrice, 0] = 1
    order[np.arange(relativePrice.shape[0]), relativePrice, 1] = (X[:, 2] / 200.0)
    order = np.swapaxes(order, 1, 2)
    return order


def loopZeroDoXsubset(filesRun=(0, 15)):
    def saveComponent(nameNum):
        names = giveFNames()
        name = names[nameNum]

        print (name)

        for isBuy in [True, False]:##[True, False]:

            if isBuy:
                oType = "B"
            else:
                oType = "S"

            #M = 10000000
            #M = 2000000 #For now, Jun 30. For now, Jun 29 (1000000)

            #M = 500000
            #N = 100000
            M = 150000
            N = 50000

            finalData = loadnpz('./recursive/0/profitData/X_' + oType + '_' + name + '.npz')

            argsGood = loadnpz('./recursive/0/profitData/argsGood_' + oType + '_' + name + '.npz')

            profitTrans = loadnpz('./recursive/' + str(loopNumber) + '/orderData/withCancels/Trans_' + oType + '_' + name + '.npz')
            argsGood0 = loadnpz('./recursive/' + str(loopNumber) + '/orderData/withCancels/argsGood_' + oType + '_' + name + '.npz')
            reverser = np.zeros(int(np.max(argsGood0)+1))
            reverser[argsGood0] = profitTrans
            profitTrans = reverser[argsGood]

            #argsValid = np.argwhere((profitTrans - finalData[:, 0]) < 50000)[:, 0]
            argsValid = np.argwhere((profitTrans - finalData[:, 0]) < 10000)[:, 0]
            #print (argsValid.shape)
            #print (profitTrans.shape)

            #quit()
            #argsValid = np.argwhere(finalData[:, 1] < 2)[:, 0]
            argsChoice = np.random.choice(argsValid, size=M)
            #argsChoice = np.random.choice(finalData.shape[0], size=M)



            argsGood = argsGood[argsChoice]

            buyAr = np.zeros(M).astype(int)
            if isBuy:
                buyAr = buyAr + 1

            X = finalData[argsChoice]

            spread = loadnpz('./inputData/ITCH_LOB/spread/' + name + '.npz')
            spread = spread[X[:, 0]] / 200.0

            #print (X)
            #quit()
            #print (finalData.shape)
            del finalData
            allShares = loadnpz('./inputData/ITCH_LOB/NearBestOrders/' + name + '.npz') / 200.0 #[:, :, :, 1] / 200.0
            #print (np.mean(allShares))
            #quit()
            midpoint = allShares.shape[1] // 2
            allShares = allShares[:, midpoint-20:midpoint+20]

            allShares = allShares[X[:, 0]]



            allSharesSum = np.mean(np.mean(allShares, axis=2), axis=1) * 2.0
            allSharesSum = np.cumsum(allSharesSum)
            Q = 1000
            #allShares = allShares[N:]
            allSharesSum = (allSharesSum[Q:] - allSharesSum[:-Q]) / Q
            allSharesSum = np.concatenate( ( (np.zeros(Q) + allSharesSum[0]).astype(int), allSharesSum))


            allSharesSum = allSharesSum.repeat(allShares.shape[1] * allShares.shape[2]).reshape(allShares.shape)

            #allShares = allShares / allSharesSum #sept 19

            #print (allSharesSum[1100:1102])
            #quit()


            #allShares = np.sum(allShares, axis=2)
            #allShares = np.max(allShares, axis=1)
            #diff = allShares - (X[:, 2] / 200.0)
            #print (np.argwhere(diff < 0).shape)
            #quit()

            order = XpriceToOrder(X, isBuy)

            if isBuy:
                endingName1 = '_Bid_' + name
            else:
                endingName1 = '_Ask_' + name

            totalNum = M // N
            print (totalNum)
            for num in range(0, totalNum):
                args = np.arange(N) + (num * N)
                #args = randomized[args]
                #args = argsFull[args] #not needed
                endingName2 = endingName1 + '_' + str(num)

                print (argsGood[args].shape)

                np.savez_compressed('./recursive/0/neuralInputs/1/Shares' + endingName2 + '.npz', allShares[args])
                np.savez_compressed('./recursive/0/neuralInputs/1/order' + endingName2 + '.npz', order[args])
                np.savez_compressed('./recursive/0/neuralInputs/1/spread' + endingName2 + '.npz', spread[args])
                np.savez_compressed('./recursive/0/neuralInputs/1/positions' + endingName2 + '.npz', X[:, 0][args])
                np.savez_compressed('./recursive/0/neuralInputs/1/argsGood' + endingName2 + '.npz', argsGood[args])

    #saveComponent(0)
    #quit()
    multiProcessor(saveComponent, filesRun[0], filesRun[1], 4, doPrint = True)
    #multiProcessor(saveComponent, 0, 4, 4, doPrint = True)


def doXsubset(loopNumber, filesRun=(0, 15), doX = False):
    if (loopNumber == 0) and doX:
        loopZeroDoXsubset(filesRun=filesRun)

    def saveComponent(nameNum):
        names = giveFNames()
        name = names[nameNum]

        print (name)

        for isBuy in [True, False]:##[True, False]:

            #M = 500000
            #N = 100000
            M = 150000
            N = 50000

            if isBuy:
                oType = "B"
            else:
                oType = "S"

            profit = loadnpz('./recursive/' + str(loopNumber) + '/profitData/Trans_' + oType + '_' + name + '.npz') / 100


            '''
            X = loadnpz('./recursive/' + str(loopNumber) + '/profitData/X_' + oType + '_' + name + '.npz')[:, 0]
            X_argsort = np.argsort(X)

            #argsKeep = np.argwhere
            profit1 = np.cumsum(profit[X_argsort])
            L = 10000
            profit1 = (profit1[L:] - profit1[:-L]) / L
            profit[X_argsort[:-L]] = profit[X_argsort[:-L]] - profit1
            profit[X_argsort[-L:]] = profit[X_argsort[-L:]] - profit1[-1]
            #profit[X_argsort[:-L]] = profit1
            #profit[X_argsort[-L:]] = profit1[-1]
            #'''

            #quit()
            argsGood = loadnpz('./recursive/' + str(loopNumber) + '/profitData/argsGood_' + oType + '_' + name + '.npz')
            maxArgsGood = loadnpz('./recursive/0/orderData/withCancels/argsGood_' + oType + '_' + name + '.npz')
            max1 = int(np.max(maxArgsGood) + 1)
            del maxArgsGood
            reverser = np.zeros(max1).astype(int) - 1
            reverser[argsGood] = np.arange(argsGood.shape[0])

            if isBuy:
                endingName1 = '_Bid_' + name
            else:
                endingName1 = '_Ask_' + name

            totalNum = M // N
            print (totalNum)
            for num in range(0, totalNum):
                args = np.arange(N) + (num * N)
                #args = argsFull[args] #not needed
                endingName2 = endingName1 + '_' + str(num)
                argsGoodSubset = loadnpz('./recursive/0/neuralInputs/1/argsGood' + endingName2 + '.npz')
                profit2 = np.zeros((argsGoodSubset.shape[0], 2))
                argSubset = reverser[argsGoodSubset]
                profit2[argSubset == -1, 0] = 0
                profit2[argSubset == -1, 1] = 1
                profit2[argSubset != -1, 0] = profit[argSubset[argSubset != -1]]

                #print (np.mean(profit2[:, 1]))
                #quit()
                print (profit2.shape)
                np.savez_compressed('./recursive/' + str(loopNumber) + '/neuralInputs/1/profit' + endingName2 + '.npz', profit2)


    #print ("R")
    #saveComponent(0)
    #quit()
    #multiProcessor(saveComponent, 44, filesRun[1], 8, doPrint = True)

    multiProcessor(saveComponent, filesRun[0], filesRun[1], 4, doPrint = True)
    #quit()


#names = giveFNames()
#print (names[44])
#quit()

#20200129
'''
doXsubset(0, filesRun=(44, 52), doX=True)
quit()
names = giveFNames()

for name in names:
    print ("U -----------------------------------------------------")
    print (name)
    for isBuy in [True, False]:#

        if isBuy:
            oType = "B"
        else:
            oType = "S"

        M = 150000
        N = 50000

        if isBuy:
            endingName1 = '_Bid_' + name
        else:
            endingName1 = '_Ask_' + name

        totalNum = M // N
        print (totalNum)
        for num in range(0, totalNum):
            #args = randomized[args]
            #args = argsFull[args] #not needed
            endingName2 = endingName1 + '_' + str(num)

            spread = loadnpz('./recursive/0/neuralInputs/1/spread' + endingName2 + '.npz')
            print (spread.shape)

quit()
#doXsubset_v2(4, 0.1)
#quit()
'''

#doXsubset(2)
#quit()

def doXsubset3(loopNumber, filesRun=(0, 15), doX=False):
    if filesRun == -1:
        filesRun = 15
    def saveComponent(num):
        print (num)
        names = giveFNames()
        M = 1000000
        N = 10000
        #totalNum = M // N
        count1 = 0
        #print (filesRun[1])
        #print (names[:filesRun[1]])
        #quit()
        #for name in names[:filesRun[1]]:
        for nameNum in range(filesRun[0], filesRun[1]):
            name = names[nameNum]
            print (name)
            for isBuy in [True, False]:
                #print (isBuy)
                #name = '20200117'
                #isBuy = True
                nameNum2 = nameNum * 2
                if isBuy:
                    endingName1 = '_Bid_' + name
                else:
                    endingName1 = '_Ask_' + name
                    nameNum2 = nameNum2  + 1
                endingName2 = endingName1 + '_' + str(num)

                if (loopNumber == 0) and doX:
                    #print ('./recursive/' + str(loopNumber) + '/neuralInputs/1/order' + endingName2 + '.npz')
                    allShares1 = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/1/Shares' + endingName2 + '.npz')
                    order1 = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/1/order' + endingName2 + '.npz')
                    spread1 = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/1/spread' + endingName2 + '.npz')
                    argsGood1 = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/1/argsGood' + endingName2 + '.npz')
                    argsGood1_2 = np.zeros(argsGood1.shape[0]).astype(int) + nameNum2
                    argsGood1 = np.array([argsGood1, argsGood1_2]).T
                profit1 = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/1/profit' + endingName2 + '.npz')

                #print (np.mean(profit1))

                if count1 == 0:
                    if (loopNumber == 0) and doX:
                        allShares = allShares1
                        order = order1
                        spread = spread1
                        argsGood = argsGood1
                    profit = profit1
                else:
                    if (loopNumber == 0) and doX:
                        allShares = np.concatenate((allShares, allShares1))
                        order = np.concatenate((order, order1))
                        spread = np.concatenate((spread, spread1))
                        argsGood = np.concatenate((argsGood, argsGood1))
                    profit = np.concatenate((profit, profit1))
                count1 += 1

        endingName = '_' + str(num)

        len1 = profit.shape[0]
        len2 = (len1 // 10) * 9


        #args0 = loadnpz('./recursive/0/neuralInputs/2/args' + endingName + '.npz')
        #args = args[args < profit.shape[0]] # this helps when only running on subset of files
        #args0 = np.random.permutation(profit.shape[0])
        #np.savez_compressed('./recursive/0/neuralInputs/2/args' + endingName + '.npz', args0)
        args0 = loadnpz('./recursive/0/neuralInputs/2/args' + '_0' + '.npz')
        args1 = args0[args0< len2]
        args2 = args0[args0>=len2]
        argsFull = [args1, args2]

        #'''
        for g in [0, 1]:
            print (g)
            args = argsFull[g]
            if (loopNumber == 0) and doX:
                allShares2 = allShares[args]
                order2 = order[args]
                spread2 = spread[args]
                argsGood2 = argsGood[args]
            profit2 = profit[args]

            if (loopNumber == 0) and doX:
                np.savez_compressed('./recursive/' + str(loopNumber) + '/neuralInputs/2/Shares_g' + str(g) + endingName + '.npz', allShares2)
                np.savez_compressed('./recursive/' + str(loopNumber) + '/neuralInputs/2/order_g' + str(g) + endingName + '.npz', order2)
                np.savez_compressed('./recursive/' + str(loopNumber) + '/neuralInputs/2/spread_g' + str(g) + endingName + '.npz', spread2)
                np.savez_compressed('./recursive/' + str(loopNumber) + '/neuralInputs/2/argsGood_g' + str(g) + endingName + '.npz', argsGood2)

                torch.save(torch.tensor(allShares2).float(), './recursive/' + str(loopNumber) + '/neuralInputs/2/Shares_g' + str(g) + endingName + '.pt')
                torch.save(torch.tensor(order2).float(), './recursive/' + str(loopNumber) + '/neuralInputs/2/order_g' + str(g) + endingName + '.pt')
                torch.save(torch.tensor(spread2).float(), './recursive/' + str(loopNumber) + '/neuralInputs/2/spread_g' + str(g) + endingName + '.pt')

            np.savez_compressed('./recursive/' + str(loopNumber) + '/neuralInputs/2/profit_g' + str(g) + endingName + '.npz', profit2)
        #'''

        '''
        if (loopNumber == 0) and doX:
            args = np.random.permutation(profit.shape[0])
            np.savez_compressed('./recursive/0/neuralInputs/2/args' + endingName + '.npz', args)
            allShares = allShares[args]
            order = order[args]
            spread = spread[args]
            argsGood = argsGood[args]
        else:
            args = loadnpz('./recursive/0/neuralInputs/2/args' + endingName + '.npz')

        profit = profit[args]


        if (loopNumber == 0) and doX:
            np.savez_compressed('./recursive/' + str(loopNumber) + '/neuralInputs/2/Shares' + endingName + '.npz', allShares)
            np.savez_compressed('./recursive/' + str(loopNumber) + '/neuralInputs/2/order' + endingName + '.npz', order)
            np.savez_compressed('./recursive/' + str(loopNumber) + '/neuralInputs/2/spread' + endingName + '.npz', spread)
            np.savez_compressed('./recursive/' + str(loopNumber) + '/neuralInputs/2/argsGood' + endingName + '.npz', argsGood)

            torch.save(torch.tensor(allShares).float(), './recursive/' + str(loopNumber) + '/neuralInputs/2/Shares' + endingName + '.pt')
            torch.save(torch.tensor(order).float(), './recursive/' + str(loopNumber) + '/neuralInputs/2/order' + endingName + '.pt')
            torch.save(torch.tensor(spread).float(), './recursive/' + str(loopNumber) + '/neuralInputs/2/spread' + endingName + '.pt')

        np.savez_compressed('./recursive/' + str(loopNumber) + '/neuralInputs/2/profit' + endingName + '.npz', profit)
        #'''

    print ("Z")
    for a in range(0, 3):
        saveComponent(a)
    #quit()
    #multiProcessor(saveComponent, 0, 20, 5, doPrint = True)
    #multiProcessor(saveComponent, 0, 5, 5, doPrint = True)
    #multiProcessor(saveComponent, filesRun[0], filesRun[1], 5, doPrint = True)
    #quit()




#doXsubset(2)
#quit()
#doXsubset3(1)
#quit()

def doXsubset2(name, isBuy, num, M):


    if isBuy:
        oType = 'B'
    else:
        oType = 'S'

    finalData = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')

    max1 = min(finalData.shape[0], M * (num+1))
    start1 = M * num
    size1 = max1-start1

    argsChoice = np.arange(size1) + start1
    X = finalData[argsChoice]
    del finalData

    spread = loadnpz('./inputData/ITCH_LOB/spread/' + name + '.npz')
    spread = spread[X[:, 0]] / 200.0

    #allShares = loadnpz('./inputData/ITCH_LOB/NearBestOrders/' + name + '.npz') / 200.0
    #midpoint = allShares.shape[1] // 2
    #allShares = allShares[:, midpoint-20:midpoint+20]

    #allShares = np.load('./inputData/ITCH_LOB/NearBestOrders_npy/' + name + '.npy')
    allShares = torch.load('./inputData/ITCH_LOB/NearBestOrders_npy/' + name + '.pt')
    allShares = allShares[torch.tensor(X[:, 0]).long()]

    order = XpriceToOrder(X, isBuy)
    del X

    return allShares, order, spread



def doXsubset4(name, isBuy):


    if isBuy:
        #finalData = loadnpz('./inputData/Processed/inputXY/X_Bid_' + name + '.npz')
        oType = 'B'
    else:
        #finalData = loadnpz('./inputData/Processed/inputXY/X_Ask_' + name + '.npz')
        oType = 'S'

    finalData = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')

    #process = psutil.Process(os.getpid())
    #print(process.memory_info().rss // 1000000 / 1000)

    #Faster alternative that still records everything needed.
    #loadnpz('./recursive/' + str(0) + '/orderData/withCancels/X_' + oType + '_' + name + '.npz', finalData_2)
    #loadnpz('./recursive/' + str(0) + '/orderData/withCancels/argsGood_' + oType + '_' + name + '.npz', argsGood_2)

    #print ("A")
    #print (finalData.shape)
    #print ( M * (num+1))
    #max1 = min(finalData.shape[0], M * (num+1))
    #start1 = M * num
    #size1 = max1-start1
    #print (size1)
    #if size1 < 0:
    #    quit()
    #argsChoice = np.arange(size1) + start1
    argsChoice = np.random.choice(finalData.shape[0], finalData.shape[0]//100, replace=False)
    size1 = argsChoice.shape[0]
    X = finalData[argsChoice]
    del finalData

    buyAr = np.zeros(size1).astype(int)
    if isBuy:
        buyAr = buyAr + 1

    spread = loadnpz('./inputData/ITCH_LOB/spread/' + name + '.npz')
    spread = spread[X[:, 0]] / 200.0

    #process = psutil.Process(os.getpid())
    #print(process.memory_info().rss // 1000000 / 1000)

    allShares = loadnpz('./inputData/ITCH_LOB/NearBestOrders/' + name + '.npz') / 200.0 #[:, :, :, 1] / 200.0
    midpoint = allShares.shape[1] // 2
    allShares = allShares[:, midpoint-20:midpoint+20]

    #process = psutil.Process(os.getpid())
    #print(process.memory_info().rss // 1000000 / 1000)

    allShares = allShares[X[:, 0]]

    order = XpriceToOrder(X, isBuy)

    #del relativePrice
    del X
    del buyAr

    #time.sleep(10)









    #process = psutil.Process(os.getpid())
    #print(process.memory_info().rss // 1000000 / 1000)

    #order_1 = order[:, 0, :]
    #count = np.arange(38).repeat(order.shape[0]).reshape((38, order_1.shape[0])).T
    #orderNum = order_1 * count
    #orderNum = np.sum(orderNum, axis=1)
    #print (np.unique(orderNum, return_counts = True))
    #quit()

    return allShares, order, spread, argsChoice



def doXsubset2_fast(name, isBuy, num, M):


    if isBuy:
        oType = 'B'
    else:
        oType = 'S'

    finalData = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')

    max1 = min(finalData.shape[0], M * (num+1))
    start1 = M * num
    size1 = max1-start1

    argsChoice = np.arange(size1) + start1
    X = finalData[argsChoice]
    del finalData

    spread = loadnpz('./inputData/ITCH_LOB/spread/' + name + '.npz')
    spread = spread[X[:, 0]] / 200.0

    #allShares = loadnpz('./inputData/ITCH_LOB/NearBestOrders/' + name + '.npz') / 200.0
    #midpoint = allShares.shape[1] // 2
    #allShares = allShares[:, midpoint-20:midpoint+20]

    #allShares = np.load('./inputData/ITCH_LOB/NearBestOrders_npy/' + name + '.npy')
    allShares = torch.load('./inputData/ITCH_LOB/NearBestOrders_npy_fast/' + name + '.pt')
    allShares = allShares[torch.tensor(X[:, 0]).long()]

    order = XpriceToOrder(X, isBuy)
    del X

    return allShares, order, spread



def doXsubset2_fast2(name, isBuy, args):


    if isBuy:
        oType = 'B'
    else:
        oType = 'S'

    finalData = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')

    #max1 = min(finalData.shape[0], M * (num+1))
    #start1 = M * num
    #size1 = max1-start1
    #argsChoice = np.arange(size1) + start1
    X = finalData[argsChoice]
    del finalData

    spread = loadnpz('./inputData/ITCH_LOB/spread/' + name + '.npz')
    spread = spread[X[:, 0]] / 200.0

    #allShares = loadnpz('./inputData/ITCH_LOB/NearBestOrders/' + name + '.npz') / 200.0
    #midpoint = allShares.shape[1] // 2
    #allShares = allShares[:, midpoint-20:midpoint+20]

    #allShares = np.load('./inputData/ITCH_LOB/NearBestOrders_npy/' + name + '.npy')
    allShares = torch.load('./inputData/ITCH_LOB/NearBestOrders_npy_fast/' + name + '.pt')
    allShares = allShares[torch.tensor(X[:, 0]).long()]

    order = XpriceToOrder(X, isBuy)
    del X

    return allShares, order, spread

def doXsubset4(name, isBuy):


    if isBuy:
        #finalData = loadnpz('./inputData/Processed/inputXY/X_Bid_' + name + '.npz')
        oType = 'B'
    else:
        #finalData = loadnpz('./inputData/Processed/inputXY/X_Ask_' + name + '.npz')
        oType = 'S'

    finalData = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')

    #process = psutil.Process(os.getpid())
    #print(process.memory_info().rss // 1000000 / 1000)

    #Faster alternative that still records everything needed.
    #loadnpz('./recursive/' + str(0) + '/orderData/withCancels/X_' + oType + '_' + name + '.npz', finalData_2)
    #loadnpz('./recursive/' + str(0) + '/orderData/withCancels/argsGood_' + oType + '_' + name + '.npz', argsGood_2)

    #print ("A")
    #print (finalData.shape)
    #print ( M * (num+1))
    #max1 = min(finalData.shape[0], M * (num+1))
    #start1 = M * num
    #size1 = max1-start1
    #print (size1)
    #if size1 < 0:
    #    quit()
    #argsChoice = np.arange(size1) + start1
    argsChoice = np.random.choice(finalData.shape[0], finalData.shape[0]//100, replace=False)
    size1 = argsChoice.shape[0]
    X = finalData[argsChoice]
    del finalData

    buyAr = np.zeros(size1).astype(int)
    if isBuy:
        buyAr = buyAr + 1

    spread = loadnpz('./inputData/ITCH_LOB/spread/' + name + '.npz')
    spread = spread[X[:, 0]] / 200.0

    #process = psutil.Process(os.getpid())
    #print(process.memory_info().rss // 1000000 / 1000)

    allShares = loadnpz('./inputData/ITCH_LOB/NearBestOrders/' + name + '.npz') / 200.0 #[:, :, :, 1] / 200.0
    midpoint = allShares.shape[1] // 2
    allShares = allShares[:, midpoint-20:midpoint+20]

    #process = psutil.Process(os.getpid())
    #print(process.memory_info().rss // 1000000 / 1000)

    allShares = allShares[X[:, 0]]

    order = XpriceToOrder(X, isBuy)

    #del relativePrice
    del X
    del buyAr

    time.sleep(10)









    #process = psutil.Process(os.getpid())
    #print(process.memory_info().rss // 1000000 / 1000)

    #order_1 = order[:, 0, :]
    #count = np.arange(38).repeat(order.shape[0]).reshape((38, order_1.shape[0])).T
    #orderNum = order_1 * count
    #orderNum = np.sum(orderNum, axis=1)
    #print (np.unique(orderNum, return_counts = True))
    #quit()

    return allShares, order, spread, argsChoice


#doXsubset()
#quit()

def leakyrelu(x):
    x = (torch.relu(x) * 0.9) + (x * 0.1)
    x = x * 2.0
    return x

class LOBSubComp(nn.Module):
    def __init__(self):
        super(LOBSubComp, self).__init__()

        self.conv1 = nn.Conv2d(1, 5, (1, 10)) #, padding=(0, 0)
        #self.conv2 = nn.Conv2d(6, 5, (1, 1))
        #self.conv3 = nn.Conv2d(5, 8, (5, 1))
        #self.conv4 = nn.Conv2d(8, 12, (5, 1), stride=(2, 1))
        #self.conv5 = nn.Conv2d(12, 8, (5, 1) , stride=(2, 1))

        self.conv2 = nn.Conv2d(6, 5, (1, 1))
        self.conv3 = nn.Conv2d(5, 5, (5, 1))
        self.conv4 = nn.Conv2d(5, 5, (5, 1), stride=(2, 1))
        self.conv5 = nn.Conv2d(5, 8, (5, 1) , stride=(2, 1))

        #self.nonlin = torch.tanh
        self.nonlin = leakyrelu

    def forward(self, x, order):
        x = self.nonlin(self.conv1(x))

        #print (x.shape)
        #quit()
        order = order.reshape((x.shape[0], 1, x.shape[2], x.shape[3]))
        x = torch.cat((x, order ), axis=1)

        #print ("Q")
        x = self.nonlin(self.conv2(x))
        x = self.nonlin(self.conv3(x))
        x = self.nonlin(self.conv4(x))
        #print (torch.mean(x))
        x = self.conv5(x)
        #print (torch.mean(x))
        return x



class LOBSubComp3(nn.Module):
    def __init__(self):
        super(LOBSubComp3, self).__init__()

        #self.conv1 = nn.Conv2d(1, 2, (1, 10)) #, padding=(0, 0)
        #self.conv2 = nn.Conv2d(6, 5, (1, 1))
        #self.conv3 = nn.Conv2d(5, 8, (5, 1))
        #self.conv4 = nn.Conv2d(8, 12, (5, 1), stride=(2, 1))
        #self.conv5 = nn.Conv2d(12, 8, (5, 1) , stride=(2, 1))

        #self.conv2 = nn.Conv2d(5, 5, (1, 1))
        self.conv2 = nn.Conv2d(1, 1, (5, 1), stride=(3, 1))
        self.conv3 = nn.Conv2d(1, 2, (5, 1), stride=(2, 1))
        #self.conv4 = nn.Conv2d(5, 8, (5, 1) , stride=(2, 1))

        #self.nonlin = torch.tanh
        self.nonlin = leakyrelu

    def forward(self, x):
        #x = self.nonlin(self.conv1(x))

        #print (x.shape)
        x = torch.mean(x, axis=3).reshape((x.shape[0], x.shape[1], x.shape[2], 1))

        x = self.nonlin(self.conv2(x))
        #print (x.shape)
        #x = self.nonlin(self.conv3(x))
        x = self.conv3(x)
        #print (x.shape)
        #x = self.conv4(x)
        #print (x.shape)
        #quit()
        #x = self.nonlin(self.conv4(x))
        #print (torch.mean(x))
        #x = self.conv5(x)
        #print (torch.mean(x))
        return x



class LOBConvNetComponent(nn.Module):
    def __init__(self):
        super(LOBConvNetComponent, self).__init__()
        '''
        self.conv1_1 = nn.Conv2d(1, 5, (1, 10), stride=(1, 1), padding=(0, 0))
        self.conv1_2 = nn.Conv2d(1, 5, (1, 10), stride=(1, 1), padding=(0, 0))
        self.conv2_1 = nn.Conv2d(5, 8, (5, 1), stride=(1, 1), padding=(0, 0))
        self.conv2_2 = nn.Conv2d(5, 8, (5, 1), stride=(1, 1), padding=(0, 0))
        self.conv3_1 = nn.Conv2d(8, 12, (5, 1), stride=(2, 1), padding=(0, 0))
        self.conv3_2 = nn.Conv2d(8, 12, (5, 1), stride=(2, 1), padding=(0, 0))
        self.conv4_1 = nn.Conv2d(12, 8, (5, 1) , stride=(2, 1), padding=(0, 0)) #10 to 6
        self.conv4_2 = nn.Conv2d(12, 8, (5, 1) , stride=(2, 1), padding=(0, 0))
        '''

        self.subComp1 = LOBSubComp()
        self.subComp2 = LOBSubComp()

    def forward(self, x, order):

        #print ("B")
        #print (torch.mean(x))

        x_1 = x[:, :20, :].reshape((x.shape[0], 1, 20, 10))
        x_2 = x[:, 20:, :].reshape((x.shape[0], 1, 20, 10))

        #print (torch.mean(x_1))

        order_1, order_2 = torch.zeros((order.shape[0], 20)), torch.zeros((order.shape[0], 20))
        order_1[:, 1:], order_2[:, :-1] = order[:, :19], order[:, 19:]

        x_1 = self.subComp1(x_1, order_1)
        x_2 = self.subComp2(x_2, order_2)

        #x_1 = torch.mean(torch.mean(torch.mean(x_1, axis=1), axis=1), axis=1)
        #x_2 = torch.mean(torch.mean(torch.mean(x_2, axis=1), axis=1), axis=1)

        #print (torch.mean(x_1))
        #print (x_1.shape)
        #x = torch.zeros((x_1.shape[0], 2))
        #x[:, 0] = x_1
        #x[:, 1] = x_2

        #x = torch.cat((x_1, x_2), axis=1)[:, :, 0, 0]
        #print (x.shape)
        #print (x_1.shape)
        x = torch.zeros((x_1.shape[0], x_1.shape[1] * 2))
        x[:, :x_1.shape[1]] = x_1[:, :, 0, 0]
        x[:, x_1.shape[1]:] = x_2[:, :, 0, 0]
        #print (x.shape)
        #quit()

        return x


class LOBConvNetComponent3(nn.Module):
    def __init__(self):
        super(LOBConvNetComponent3, self).__init__()
        self.subComp1 = LOBSubComp3()
        self.subComp2 = LOBSubComp3()

    def forward(self, x):
        #print (x.shape)
        x_1 = x[:, :20, :].reshape((x.shape[0], 1, 20, 10))
        x_2 = x[:, 20:, :].reshape((x.shape[0], 1, 20, 10))

        #x_1 = self.subComp1(x_1)
        #x_2 = self.subComp2(x_2)

        #x = torch.cat((x_1, x_2), axis=1)[:, :, 0, 0]
        x = torch.zeros((x.shape[0], 3)).float()
        x[:, 0] = torch.sum(torch.sum(torch.sum(x_1, axis=1), axis=1), axis=1)
        x[:, 1] = torch.sum(torch.sum(torch.sum(x_2, axis=1), axis=1), axis=1)
        return x


class orderProfitModel(nn.Module):
    def __init__(self):
        super(orderProfitModel, self).__init__()
        self.convModel = LOBConvNetComponent()
        self.nonlin = torch.tanh
        #self.nonlin = torch.relu

        #self.conv5_1 = nn.Conv1d(2, 1, 3, padding=1)
        #self.conv5_2 = nn.Conv1d(2, 1, 3, padding=1)

        #startRep = 16
        startRep = 16

        self.linSpread = nn.Linear(1, startRep)
        self.linOrder = nn.Linear(39, 12) #39 or 2

        #N1 = 8
        #N2 = 8
        N1 = 12
        N2 = 12
        self.lin1 = nn.Linear(startRep, N1)
        self.lin2 = nn.Linear(N1, N2)
        self.lin3 = nn.Linear(N2, 1)




    def forward(self, LOB, order, spread):

        #print ("U")
        #time1 = time.time()
        LOB = self.convModel(LOB)

        order_1 = order[:, 0, :]
        order_2 = torch.sum(order[:, 1, :], axis=1)
        order_3 = torch.zeros((order_1.shape[0], order_1.shape[1] + 1))
        order_3[:, :-1] = order_1
        order_3[:, -1] = order_2
        #order_3 = order

        spread = spread.reshape((spread.shape[0], 1))
        spread = self.linSpread(spread)

        order = self.linOrder(order_3)

        x1 = LOB + spread
        x1[:, :order.shape[1]] = x1[:, :order.shape[1]] + order

        x2 = self.nonlin(self.lin1(x1))
        x3 = self.nonlin(self.lin2(x2))
        x4 = self.lin3(x3)
        x4 = x4[:, 0]

        #print ("IN model")
        #process = psutil.Process(os.getpid())
        #print(process.memory_info().rss // 1000000 / 1000)

        return x4

class orderProfitModel3(nn.Module):
    def __init__(self):
        super(orderProfitModel3, self).__init__()
        self.convModel = LOBConvNetComponent3()
        #self.nonlin = torch.tanh
        self.nonlin = leakyrelu

        #self.conv5_1 = nn.Conv1d(2, 1, 3, padding=1)
        #self.conv5_2 = nn.Conv1d(2, 1, 3, padding=1)

        startRep = 3#16

        #self.linSpread = nn.Linear(1, startRep)
        self.linSpread = nn.Linear(6, startRep)

        N1 = 4#16#8
        N2 = 2#16#8
        self.lin1 = nn.Linear(startRep, N1)
        self.lin2 = nn.Linear(N1, N2)
        #self.lin3 = nn.Linear(N2, 2)




    def forward(self, LOB, spread):

        x1 = self.convModel(LOB)
        #x1 = x1
        #x1[:, 2] = (spread * 0.0)
        '''
        spread2 = ((spread * 2))
        spread2[spread2 > 5] = 5
        spread2[spread2 < 1] = 1
        spread2 = spread2.long()
        spread3 = torch.zeros((spread2.shape[0], 6))
        arange1 = torch.tensor(np.arange(spread2.data.numpy().shape[0])).long()
        #spread3[arange1, spread2] = 1
        spread3[:, 0] = spread

        spread = self.linSpread(spread3)
        '''

        #x1 = (LOB) + spread

        x2 = self.nonlin(self.lin1(x1))
        #x3 = self.nonlin(self.lin2(x2))
        x3 = self.lin2(x2)
        #x4 = self.lin3(x3)

        return x3



class orderProfitModel2(nn.Module):
    def __init__(self):
        super(orderProfitModel2, self).__init__()
        self.convModel = LOBConvNetComponent()
        #self.nonlin = torch.tanh
        self.nonlin = leakyrelu

        #self.conv5_1 = nn.Conv1d(2, 1, 3, padding=1)
        #self.conv5_2 = nn.Conv1d(2, 1, 3, padding=1)

        startRep = 16

        #self.linSpread = nn.Linear(1, startRep)
        self.linSpread = nn.Linear(6, startRep)
        self.linOrder = nn.Linear(39, 8) #39 or 2

        N1 = 8#16#8
        N2 = 8#16#8
        self.lin1 = nn.Linear(startRep, N1)
        self.lin2 = nn.Linear(N1, N2)
        self.lin3 = nn.Linear(N2, 2)




    def forward(self, LOB, order, spread):

        #print ("U")
        #time1 = time.time()
        order_1 = order[:, 0, :]

        #print("A")
        #print (torch.mean(LOB))

        #LOB = self.convModel(LOB, order_1)
        #print (LOB[:5])
        #LOB = LOB * 0.0

        #print (torch.mean(LOB))

        order_2 = torch.sum(order[:, 1, :], axis=1)
        order_3 = torch.zeros((order_1.shape[0], order_1.shape[1] + 1))
        order_3[:, :-1] = order_1
        order_3[:, -1] = order_2
        #order_3 = order

        spread2 = ((spread * 2))
        spread2[spread2 > 5] = 5
        spread2[spread2 < 1] = 1
        spread2 = spread2.long()
        spread3 = torch.zeros((spread2.shape[0], 6))
        arange1 = torch.tensor(np.arange(spread2.data.numpy().shape[0])).long()
        spread3[arange1, spread2] = 1
        spread3[:, 0] = spread

        spread = self.linSpread(spread3)
        #spread = spread.reshape((spread.shape[0], 1))
        #spread = self.linSpread(spread)

        order = self.linOrder(order_3)

        #x1 = LOB + spread
        x1 = spread
        #x1[:, :LOB.shape[1]] = x1[:, :LOB.shape[1]] + LOB
        x1[:, :order.shape[1]] = x1[:, :order.shape[1]] + order

        x2 = self.nonlin(self.lin1(x1))
        x3 = self.nonlin(self.lin2(x2))
        x4 = self.lin3(x3)
        #print (x4.shape)
        #quit()
        #x4 = x4[:, 0]

        #print ("IN model")
        #process = psutil.Process(os.getpid())
        #print(process.memory_info().rss // 1000000 / 1000)

        return x4



class orderProfitModel4(nn.Module):
    def __init__(self):
        super(orderProfitModel4, self).__init__()
        self.convModel = LOBConvNetComponent()
        #self.nonlin = torch.tanh
        self.nonlin = leakyrelu

        #self.conv5_1 = nn.Conv1d(2, 1, 3, padding=1)
        #self.conv5_2 = nn.Conv1d(2, 1, 3, padding=1)

        startRep = 16

        #self.linSpread = nn.Linear(1, startRep)
        self.linSpread = nn.Linear(6, startRep)
        self.linOrder = nn.Linear(39, startRep) #39 or 2
        self.linDep = nn.Linear(12, startRep)


        N1 = 16#16#8
        N2 = 16#16#8
        self.lin1 = nn.Linear(startRep, N1)
        #self.lin2 = nn.Linear(N1, N2)
        self.lin3 = nn.Linear(N2, 2)




    def forward(self, LOB, order, spread):

        #print ("U")
        #time1 = time.time()
        order_1 = order[:, 0, :]
        #depth = torch.sum(order[:, 1, :], axis=1)
        #depthShare2 = torch.zeros((spread.shape[0], 12))
        #depthShare2[:, :10] = depthShare
        #depthShare2[:, 10] = depthTotal
        #depthShare2[:, 11] = depth
        order_form_2 = order[:, 1, :12]

        #order_form_2[:, :10] = 0

        #print (order_2.shape)



        #print("A")
        #print (torch.mean(LOB))

        #LOB = self.convModel(LOB, order_1)
        #print (LOB[:5])
        #LOB = LOB * 0.0

        #print (torch.mean(LOB))

        order_2 = torch.sum(order[:, 1, :], axis=1)
        order_3 = torch.zeros((order_1.shape[0], order_1.shape[1] + 1))
        order_3[:, :-1] = order_1
        order_3[:, -1] = order_2
        #order_3 = order

        spread2 = ((spread * 2))
        spread2[spread2 > 5] = 5
        spread2[spread2 < 1] = 1
        spread2 = spread2.long()
        spread3 = torch.zeros((spread2.shape[0], 6))
        arange1 = torch.tensor(np.arange(spread2.data.numpy().shape[0])).long()
        spread3[arange1, spread2] = 1
        spread3[:, 0] = spread

        spread = self.linSpread(spread3)
        #spread = spread.reshape((spread.shape[0], 1))
        #spread = self.linSpread(spread)

        order = self.linOrder(order_3)

        #x1 = LOB + spread
        x1 = spread
        #x1[:, :LOB.shape[1]] = x1[:, :LOB.shape[1]] + LOB
        x1[:, :order.shape[1]] = x1[:, :order.shape[1]] + order

        order_form_2 = self.linDep(order_form_2)
        x1 = x1 + order_form_2

        x2 = self.nonlin(self.lin1(x1))
        #x3 = self.nonlin(self.lin2(x2))
        #x4 = self.lin3(x3)
        x4 = self.lin3(x2)
        #print (x4.shape)
        #quit()
        #x4 = x4[:, 0]

        #print ("IN model")
        #process = psutil.Process(os.getpid())
        #print(process.memory_info().rss // 1000000 / 1000)

        return x4

#findLOBpossibleTrans()
#quit()
#doXsubset()
#quit()

def savePredictions(loopNumber):
    print ("Saving Prediction")

    def saveComponent(nameNum):
        names = giveFNames()
        #model = torch.load('./Models/group2/model1_3.pt')
        model = torch.load('./Models/group2/model' + str(loopNumber) + '_4.pt')
        nameN = len(names)

        #nameN = len(names)
        M = 1000
        N = 100000
        M1 = 1000000 * 2#10
        #M = 10
        #N = 10000
        losses = []


        name = names[nameNum] #Temporary
        #name = names[0] #Temporary

        print (name)
        for isBuy in [True, False]: #[True, False]: Make both true false

            namePart = ''
            if isBuy:
                namePart = namePart + "Bid_"
                oType = "B"
            else:
                namePart = namePart + "Ask_"
                oType = "S"
            namePart = namePart + name

            notDone = True
            outputFull = np.array([])
            num = 0 #Temporary
            #num = 10 * nameNum #Temporary

            while notDone:# and ((num+1) % 10 != 0):
                print (num)
                process = psutil.Process(os.getpid())
                print(process.memory_info().rss // 1000000 / 1000)
                #time1 = time.time()
                allShares, order, spread =  doXsubset2(name, isBuy, num, M1)
                #time2 = time.time()
                #quit()
                if order.shape[0] != M1:
                    notDone = False
                allShares = torch.tensor(allShares).float()
                order = torch.tensor(order).float()
                spread = torch.tensor(spread).float()

                process = psutil.Process(os.getpid())
                print(process.memory_info().rss // 1000000 / 1000)

                lastOne = order.shape[0] // N
                if not notDone:
                    lastOne = lastOne + 1

                for a in range(0,  lastOne):

                    process = psutil.Process(os.getpid())
                    print(process.memory_info().rss // 1000000 / 1000)

                    #print (a)
                    allShares1 = allShares[a*N:(a+1)*N]
                    order1 = order[a*N:(a+1)*N]
                    spread1 = spread[a*N:(a+1)*N]
                    #print (order1.shape)

                    output = model(allShares1, order1, spread1)

                    output = output.data.numpy()
                    outputFull = np.concatenate((outputFull, output))

                    process = psutil.Process(os.getpid())
                    print(process.memory_info().rss // 1000000 / 1000)

                    quit()

                #time3 = time.time()
                #print (time2 - time1)
                #print (time3 - time2)
                #quit()

                del allShares
                del order
                num += 1


                #quit()

            #print ("T1")
            #np.savez_compressed('./resultData/outputPrediction5/' + str(nameNum) + '.npz', outputFull)
            #quit()
            #np.savez_compressed('./resultData/outputPrediction5/' + namePart + '.npz', outputFull)

            #####np.savez_compressed('./recursive/' + str(loopNumber) + '/outputPredictions/' + oType + '_' + name + '.npz', outputFull)


            #np.savez_compressed('./recursive/' + str(loopNumber) + '/outputPredictions/' + oType + str(num) + '_' + name + '.npz')
            #quit()

    saveComponent(0)
    #multiProcessor(saveComponent, 0, 4, 4, doPrint = True)
    #multiProcessor(saveComponent, 0, 15, 5, doPrint = True)
    #multiProcessor(saveComponent, 0, 5, 5, doPrint = True)
    #quit()

#quit()
#savePredictions(2)
#quit()


def savePredictions2(loopNumber):
    print ("Saving Prediction 2")

    def saveComponent_sub(codeNum):

        #print (codeNum)
        #'''
        nameNum = int(codeNum // 2)
        names = giveFNames()
        #model = torch.load('./Models/group2/model1_3.pt')
        #model = torch.load('./Models/group2/model' + str(loopNumber) + '_4.pt') #4

        #name = names[nameNum]
        #allShares = np.load('./inputData/ITCH_LOB/NearBestOrders_npy/' + name + '.npy')
        #print (allShares.shape)
        #time_1 = time.time()
        #allShares = torch.tensor(allShares).float()
        #print (time.time() - time_1)
        #quit()

        #if False:#loopNumber != 0:
        #    model1 = torch.load('./recursive/' + str(loopNumber) + '/Models/' + 'ExecutionProbability' + '/1.pt')#temp 4
        #model2 = torch.load('./recursive/' + str(loopNumber) + '/Models/' + 'expectedProfit3' + '/3.pt')
        #model2 = torch.load('./recursive/' + str(loopNumber) + '/Models/' + 'expectedProfit' + '/4.pt')

        model = torch.load('./recursive/' + str(loopNumber) + '/Models/' + 'predictBoth' + '/1.pt')

        nameN = len(names)

        #nameN = len(names)
        M = 1000
        N = 100000
        M1 = 1000000 *5#* 10#10
        #M1 = 10000000
        #M = 10
        #N = 10000
        losses = []


        name = names[nameNum]

        print (name)


        #'''
        #if nameNum != 0:
        #    np.save('./inputData/ITCH_LOB/NearBestOrders_npy/' + names[nameNum - 1] + '.npy', np.array([0]))
        allShares = loadnpz('./inputData/ITCH_LOB/NearBestOrders/' + name + '.npz') / 200.0 #[:, :, :, 1] / 200.0
        midpoint = allShares.shape[1] // 2
        allShares = allShares[:, midpoint-20:midpoint+20]
        allShares = torch.tensor(allShares).float()
        torch.save(allShares, './inputData/ITCH_LOB/NearBestOrders_npy/' + name + '.pt')
        #np.save('./inputData/ITCH_LOB/NearBestOrders_npy/' + name + '.npy', allShares)
        del allShares
        #'''
        #print ("Did Save")
        #quit()





        if codeNum % 2 == 0:
            isBuy = True
        else:
            isBuy = False

        #print (isBuy)
        #quit()
        #for isBuy in [True, False]: #[True, False]: Make both true false

        namePart = ''
        if isBuy:
            namePart = namePart + "Bid_"
            oType = "B"
        else:
            namePart = namePart + "Ask_"
            oType = "S"
        namePart = namePart + name
        #print (namePart)
        #print (oType)
        #quit()

        notDone = True
        #outputFull = np.array([])

        finalData = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')
        outputFull = np.zeros((finalData.shape[0], 2))
        del finalData
        count1 = 0
        num = 0 #Temporary
        #num = 10 * nameNum #Temporary

        times1 = 0
        times2 = 0
        times3 = 0
        times4 = 0
        times5 = 0
        times6 = 0

        while notDone:# and ((num+1) % 10 != 0):
            time1 = time.time()
            print (num)
            #time1 = time.time()
            allShares, order, spread =  doXsubset2(name, isBuy, num, M1)
            times1 += (time.time() - time1)

            print ('times1', times1)

            time2 = time.time()
            #print ("Order shape", order.shape[0], M1)
            #time2 = time.time()
            #quit()
            if order.shape[0] != M1:
                notDone = False

            with torch.no_grad():
                order = torch.tensor(order).float()
                spread = torch.tensor(spread).float()

                lastOne = order.shape[0] // N
                if not notDone:
                    lastOne = lastOne + 1


                for a in range(0,  lastOne):
                    #print (a)
                    time4 = time.time()
                    allShares1 = allShares[a*N:(a+1)*N]
                    order1 = order[a*N:(a+1)*N]
                    spread1 = spread[a*N:(a+1)*N]

                    #print (order1.shape)

                    #if False:#loopNumber != 0:
                    #    output1 = model1(allShares1, order1, spread1)
                    #    output1 = torch.sigmoid(output1)
                    #    output1 = output1.data.numpy()
                    #else:
                    #    output1 = np.ones(spread1.shape[0])

                    output = model(allShares1, order1, spread1)
                    output[:, 0] = torch.sigmoid(output[:, 0])

                    times4 += (time.time() - time4)
                    time3 = time.time()
                    #output2 =  model2(allShares1, order1, spread1)
                    times3 += (time.time() - time3)

                    time6 = time.time()
                    #output2 = output2.data.numpy()
                    #output = np.array([output1, output2]).T
                    #outputFull = np.concatenate((outputFull, output))
                    count_size = output.shape[0]
                    outputFull[count1:count1 + count_size] = output
                    count1 = count1 + count_size

                    times6 += (time.time() - time6)


                    #output = model(allShares1, order1, spread1)
                    #output = output.data.numpy()
                    #outputFull = np.concatenate((outputFull, output))

                #time3 = time.time()
                #print (time2 - time1)
                #print (time3 - time2)
                #quit()

            del allShares
            del order
            num += 1
            #quit()

            times2 += (time.time() - time2)

            #print ("A")
            #print (times1)
            #print (times2)
            #print (times3)
            #print (times4)
            #print (times5)
            #print (times6)

        np.savez_compressed('./recursive/' + str(loopNumber) + '/outputPredictions/' + oType + '_' + name + '.npz', outputFull)
        #'''

        #np.save('./inputData/ITCH_LOB/NearBestOrders_npy/' + name + '.npy', np.array([0]))
        torch.save(torch.tensor([0]), './inputData/ITCH_LOB/NearBestOrders_npy/' + name + '.pt')

        print ("Done Code: " + str(codeNum))


    def saveComponent(startPoint):

        for a in range(0, 6):
            #if startPoint + (a * 5) < 30:
            #if startPoint + (a * 5):
            #if startPoint + (a * 5) < 16:
            saveComponent_sub(startPoint + (a * 5)) #+ 18
                    #time.sleep(1)


    def saveComponent(startPoint):

        for a in range(0, 15):
            saveComponent_sub(startPoint + (a * 2)) #+ 18

    #0, .. 8
    #13 ..
    #saveComponent_sub(29)
    #quit()

    for a in range(0, 30):
        saveComponent_sub(a)
    quit()

    multiProcessor(saveComponent, 0, 2, 2, doPrint = True)

    #multiProcessor(saveComponent, 0, 5, 5, doPrint = True)

#savePredictions2(0)
#quit()

def savePredictions2_fast(loopNumber, cutOff):
    print ("Saving Prediction 2")

    def saveComponent_sub(codeNum):

        nameNum = int(codeNum // 2)
        names = giveFNames()

        model = torch.load('./recursive/' + str(loopNumber) + '/Models/' + 'predictBoth' + '/6.pt')
        modelFast = torch.load('./recursive/' + str(loopNumber) + '/Models/' + 'fastCancel' + '/4.pt')

        nameN = len(names)
        M = 1000
        N = 100000
        M1 = 1000000 * 5
        losses = []


        name = names[nameNum]

        print (name)


        #'''
        #if nameNum != 0:
        #    np.save('./inputData/ITCH_LOB/NearBestOrders_npy/' + names[nameNum - 1] + '.npy', np.array([0]))
        allShares = loadnpz('./inputData/ITCH_LOB/NearBestOrders/' + name + '.npz') / 200.0 #[:, :, :, 1] / 200.0
        midpoint = allShares.shape[1] // 2
        allShares = allShares[:, midpoint-20:midpoint+20]
        allShares = torch.tensor(allShares).float()
        torch.save(allShares, './inputData/ITCH_LOB/NearBestOrders_npy/' + name + '.pt')
        #np.save('./inputData/ITCH_LOB/NearBestOrders_npy/' + name + '.npy', allShares)
        del allShares
        #'''
        #print ("Did Save")
        #quit()





        if codeNum % 2 == 0:
            isBuy = True
        else:
            isBuy = False


        namePart = ''
        if isBuy:
            namePart = namePart + "Bid_"
            oType = "B"
        else:
            namePart = namePart + "Ask_"
            oType = "S"
        namePart = namePart + name

        notDone = True

        finalData = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')
        outputFull = np.zeros((finalData.shape[0], 2))
        outputFull[:, 0] = 2
        del finalData
        count1 = 0
        num = 0 #Temporary
        #num = 10 * nameNum #Temporary

        times1 = 0
        times2 = 0
        times3 = 0
        times4 = 0
        times5 = 0
        times6 = 0

        while notDone:# and ((num+1) % 10 != 0):
            time1 = time.time()
            print (num)
            #time1 = time.time()
            allShares, order, spread =  doXsubset2(name, isBuy, num, M1)
            times1 += (time.time() - time1)

            print ('times1', times1)

            time2 = time.time()
            #print ("Order shape", order.shape[0], M1)
            #time2 = time.time()
            #quit()
            if order.shape[0] != M1:
                notDone = False

            with torch.no_grad():
                order = torch.tensor(order).float()
                spread = torch.tensor(spread).float()

                spread2 = spread.reshape((spread.shape[0], 1))
                allShares1 = torch.sum(allShares, axis=2)
                #order1 = torch.matmul(order[:, 1, :], torch.tensor(np.arange(38).reshape(38, 1)).float())
                order1 = torch.sum(order[:, 1, :], axis=1).reshape((order.shape[0], 1))
                order2 = order[:, 0, :]
                X = torch.cat((allShares1, order2, order1, spread2), 1)


                lastOne = order.shape[0] // N
                if not notDone:
                    lastOne = lastOne + 1


                for a in range(0,  lastOne):
                    #print (a)
                    time4 = time.time()

                    X1 = X[a*N:(a+1)*N]
                    prediction = modelFast(X1)
                    prediction2 = prediction.data.numpy()
                    #argsTop = np.argsort(prediction2)[(-N//10):]
                    argsTop = np.argwhere(prediction2 > cutOff)[:, 0]
                    argsTop = (a*N) + torch.tensor(argsTop).long()

                    allShares1 = allShares[argsTop]
                    order1 = order[argsTop]
                    spread1 = spread[argsTop]


                    output = model(allShares1, order1, spread1)
                    output[:, 0] = torch.sigmoid(output[:, 0])

                    times4 += (time.time() - time4)
                    time3 = time.time()
                    #output2 =  model2(allShares1, order1, spread1)
                    times3 += (time.time() - time3)

                    time6 = time.time()
                    #output2 = output2.data.numpy()
                    #output = np.array([output1, output2]).T
                    #outputFull = np.concatenate((outputFull, output))
                    count_size = output.shape[0]

                    argsTop = argsTop.data.numpy().astype(int)
                    outputFull[argsTop] = output
                    count1 = count1 + N

                    times6 += (time.time() - time6)


                    #output = model(allShares1, order1, spread1)
                    #output = output.data.numpy()
                    #outputFull = np.concatenate((outputFull, output))

                #time3 = time.time()
                #print (time2 - time1)
                #print (time3 - time2)
                #quit()

            del allShares
            del order
            num += 1
            #quit()

            times2 += (time.time() - time2)

            print ("A")
            print (times1)
            print (times2)
            print (times3)
            print (times4)
            print (times5)
            print (times6)

        np.savez_compressed('./recursive/' + str(loopNumber) + '/outputPredictionsFast/' + oType + '_' + name + '.npz', outputFull)
        #'''

        #np.save('./inputData/ITCH_LOB/NearBestOrders_npy/' + name + '.npy', np.array([0]))
        torch.save(torch.tensor([0]), './inputData/ITCH_LOB/NearBestOrders_npy/' + name + '.pt')

        print ("Done Code: " + str(codeNum))


    def saveComponent(startPoint):

        for a in range(0, 6):
            #if startPoint + (a * 5) < 30:
            #if startPoint + (a * 5):
            #if startPoint + (a * 5) < 16:
            saveComponent_sub(startPoint + (a * 5)) #+ 18
                    #time.sleep(1)

    #0, .. 8
    #13 ..
    #saveComponent_sub(29)
    #quit()

    for a in range(0, 30):
        saveComponent_sub(a)
    quit()

    multiProcessor(saveComponent, 0, 5, 5, doPrint = True)



def savePredictions2_fast_v2(loopNumber, cutOff):
    print ("Saving Prediction 2")

    def saveComponent_sub(codeNum):

        nameNum = int(codeNum // 2)
        names = giveFNames()

        model = torch.load('./recursive/' + str(loopNumber) + '/Models/' + 'predictBoth' + '/3.pt')
        modelFast = torch.load('./recursive/' + str(loopNumber) + '/Models/' + 'fastCancel' + '/3.pt')

        nameN = len(names)
        M = 1000
        N = 100000
        M1 = 1000000 * 5 * 5
        losses = []


        name = names[nameNum]

        print (name)


        #'''
        #if nameNum != 0:
        #    np.save('./inputData/ITCH_LOB/NearBestOrders_npy/' + names[nameNum - 1] + '.npy', np.array([0]))
        allShares = loadnpz('./inputData/ITCH_LOB/NearBestOrders/' + name + '.npz') / 200.0 #[:, :, :, 1] / 200.0
        midpoint = allShares.shape[1] // 2
        allShares = allShares[:, midpoint-20:midpoint+20]
        allShares = torch.tensor(allShares).float()
        allShares = torch.sum(allShares, axis=2)

        torch.save(allShares, './inputData/ITCH_LOB/NearBestOrders_npy_fast/' + name + '.pt')
        #np.save('./inputData/ITCH_LOB/NearBestOrders_npy/' + name + '.npy', allShares)
        del allShares
        #'''
        #print ("Did Save")
        #quit()





        if codeNum % 2 == 0:
            isBuy = True
        else:
            isBuy = False


        namePart = ''
        if isBuy:
            namePart = namePart + "Bid_"
            oType = "B"
        else:
            namePart = namePart + "Ask_"
            oType = "S"
        namePart = namePart + name

        notDone = True

        finalData = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')
        #outputFull = np.zeros((finalData.shape[0], 2))
        #outputFull[:, 0] = 2
        outputArgs = np.zeros(finalData.shape[0])
        del finalData
        count1 = 0
        num = 0 #Temporary
        #num = 10 * nameNum #Temporary

        times1 = 0
        times2 = 0
        times3 = 0
        times4 = 0
        times5 = 0
        times6 = 0

        while notDone:# and ((num+1) % 10 != 0):
            time1 = time.time()
            print (num)
            #time1 = time.time()
            allShares, order, spread =  doXsubset2_fast(name, isBuy, num, M1)
            times1 += (time.time() - time1)

            print ("A")
            print (times1)
            #quit()

            print ('times1', times1)

            time2 = time.time()
            #print ("Order shape", order.shape[0], M1)
            #time2 = time.time()
            #quit()
            if order.shape[0] != M1:
                notDone = False

            with torch.no_grad():
                order = torch.tensor(order).float()
                spread = torch.tensor(spread).float()

                spread = spread.reshape((spread.shape[0], 1))
                #allShares1 = torch.sum(allShares, axis=2)
                #order1 = torch.matmul(order[:, 1, :], torch.tensor(np.arange(38).reshape(38, 1)).float())
                print (order.shape)
                order1 = torch.sum(order[:, 1, :], axis=1).reshape((order.shape[0], 1))
                order2 = order[:, 0, :]
                print (order1.shape)
                print (order2.shape)
                print (spread.shape)
                print (allShares.shape)
                X = torch.cat((allShares, order2, order1, spread), 1)


                lastOne = order.shape[0] // N
                if not notDone:
                    lastOne = lastOne + 1


                for a in range(0,  lastOne):
                    #print (a)
                    X1 = X[a*N:(a+1)*N]
                    prediction = modelFast(X1)
                    prediction2 = prediction.data.numpy()
                    #argsTop = np.argsort(prediction2)[(-N//10):]
                    argsTop = np.argwhere(prediction2 > cutOff)[:, 0]
                    outputArgs[argsTop] = 1


            del allShares
            del order
            num += 1
            #quit()

            times2 += (time.time() - time2)

            print ("A")
            print (times1)
            print (times2)
            #quit()

        np.savez_compressed('./recursive/' + str(loopNumber) + '/outputPredictionsFast_v2/toCheck_' + oType + '_' + name + '.npz', outputArgs)
        #np.savez_compressed('./recursive/' + str(loopNumber) + '/outputPredictionsFast_v2/toCheck_' + oType + '_' + name + '.npz', outputFull)
        #'''

        #np.save('./inputData/ITCH_LOB/NearBestOrders_npy/' + name + '.npy', np.array([0]))
        torch.save(torch.tensor([0]), './inputData/ITCH_LOB/NearBestOrders_npy_fast/' + name + '.pt')

        print ("Done Code: " + str(codeNum))


    def saveComponent(startPoint):

        for a in range(0, 6):
            #if startPoint + (a * 5) < 30:
            #if startPoint + (a * 5):
            #if startPoint + (a * 5) < 16:
            saveComponent_sub(startPoint + (a * 5)) #+ 18
                    #time.sleep(1)

    #0, .. 8
    #13 ..
    #saveComponent_sub(29)
    #quit()

    for a in range(0, 30):
        saveComponent_sub(a)
    quit()

    multiProcessor(saveComponent, 0, 5, 5, doPrint = True)

#savePredictions2_fast_v2(1, -0.75)
#quit()

def savePredictions2_fast_v3(loopNumber, cutOff):
    print ("Saving Prediction 2")

    def saveComponent_sub(codeNum):

        nameNum = int(codeNum // 2)
        names = giveFNames()

        model = torch.load('./recursive/' + str(loopNumber) + '/Models/' + 'predictBoth' + '/3.pt')
        modelFast = torch.load('./recursive/' + str(loopNumber) + '/Models/' + 'fastCancel' + '/3.pt')

        nameN = len(names)
        M = 1000
        N = 100000
        M1 = 1000000 * 5
        losses = []


        name = names[nameNum]

        print (name)


        #'''
        #if nameNum != 0:
        #    np.save('./inputData/ITCH_LOB/NearBestOrders_npy/' + names[nameNum - 1] + '.npy', np.array([0]))
        allShares = loadnpz('./inputData/ITCH_LOB/NearBestOrders/' + name + '.npz') / 200.0 #[:, :, :, 1] / 200.0
        midpoint = allShares.shape[1] // 2
        allShares = allShares[:, midpoint-20:midpoint+20]
        allShares = torch.tensor(allShares).float()
        allShares = torch.sum(allShares, axis=2)

        torch.save(allShares, './inputData/ITCH_LOB/NearBestOrders_npy_fast/' + name + '.pt')
        #np.save('./inputData/ITCH_LOB/NearBestOrders_npy/' + name + '.npy', allShares)
        del allShares
        #'''
        #print ("Did Save")
        #quit()





        if codeNum % 2 == 0:
            isBuy = True
        else:
            isBuy = False


        namePart = ''
        if isBuy:
            namePart = namePart + "Bid_"
            oType = "B"
        else:
            namePart = namePart + "Ask_"
            oType = "S"
        namePart = namePart + name

        notDone = True

        finalData = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')
        outputFull = np.zeros((finalData.shape[0], 2))
        outputFull[:, 0] = 2
        outputArgs = np.zeros(finalData.shape[0])
        del finalData
        count1 = 0
        num = 0 #Temporary
        #num = 10 * nameNum #Temporary

        outputFull = loadnpz('./recursive/' + str(loopNumber) + '/outputPredictionsFast/toCheck_' + oType + '_' + name + '.npz')
        argsChoice1 = np.argwhere(outputFull == 1)[:, 0]

        times1 = 0
        times2 = 0
        times3 = 0
        times4 = 0
        times5 = 0
        times6 = 0

        while notDone:# and ((num+1) % 10 != 0):
            time1 = time.time()
            print (num)
            #time1 = time.time()
            argsChoice2 = argsChoice1[M1 * num: M1 * (num + 1)]
            allShares, order, spread =  doXsubset2_fast2(name, isBuy, argsChoice2)
            times1 += (time.time() - time1)

            print ('times1', times1)

            time2 = time.time()
            #print ("Order shape", order.shape[0], M1)
            #time2 = time.time()
            #quit()
            if order.shape[0] != M1:
                notDone = False

            with torch.no_grad():
                order = torch.tensor(order).float()
                spread = torch.tensor(spread).float()

                lastOne = order.shape[0] // N
                if not notDone:
                    lastOne = lastOne + 1


                for a in range(0,  lastOne):
                    #print (a)
                    time4 = time.time()
                    allShares1 = allShares[a*N:(a+1)*N]
                    order1 = order[a*N:(a+1)*N]
                    spread1 = spread[a*N:(a+1)*N]

                    output = model(allShares1, order1, spread1)
                    output[:, 0] = torch.sigmoid(output[:, 0])

                    times4 += (time.time() - time4)
                    time3 = time.time()
                    #output2 =  model2(allShares1, order1, spread1)
                    times3 += (time.time() - time3)

                    time6 = time.time()
                    #output2 = output2.data.numpy()
                    #output = np.array([output1, output2]).T
                    #outputFull = np.concatenate((outputFull, output))
                    count_size = output.shape[0]
                    outputFull[argsChoice2] = output
                    count1 = count1 + count_size

                    times6 += (time.time() - time6)


                    #output = model(allShares1, order1, spread1)
                    #output = output.data.numpy()
                    #outputFull = np.concatenate((outputFull, output))

                #time3 = time.time()
                #print (time2 - time1)
                #print (time3 - time2)
                #quit()

            del allShares
            del order
            num += 1
            #quit()

            times2 += (time.time() - time2)

            #print ("A")
            #print (times1)
            #print (times2)
            #print (times3)
            #print (times4)
            #print (times5)
            #print (times6)

        np.savez_compressed('./recursive/' + str(loopNumber) + '/outputPredictions_fast/ran_' + oType + '_' + name + '.npz', outputFull)
        #'''

        #np.save('./inputData/ITCH_LOB/NearBestOrders_npy/' + name + '.npy', np.array([0]))
        torch.save(torch.tensor([0]), './inputData/ITCH_LOB/NearBestOrders_npy/' + name + '.pt')

        print ("Done Code: " + str(codeNum))


    def saveComponent(startPoint):

        for a in range(0, 6):
            #if startPoint + (a * 5) < 30:
            #if startPoint + (a * 5):
            #if startPoint + (a * 5) < 16:
            saveComponent_sub(startPoint + (a * 5)) #+ 18
                    #time.sleep(1)

    #0, .. 8
    #13 ..
    #saveComponent_sub(29)
    #quit()

    for a in range(0, 30):
        saveComponent_sub(a)
    quit()

    multiProcessor(saveComponent, 0, 5, 5, doPrint = True)

#savePredictions2_fast_v2(1, -0.75)
#quit()

def savePredictions3(loopNumber, filesRun):
    print ("Saving Prediction 3")

    def saveComponent_sub(codeNum):

        #print (codeNum)
        #'''
        nameNum = int(codeNum // 2)
        names = giveFNames()
        #model = torch.load('./Models/group2/model1_3.pt')
        #model = torch.load('./Models/group2/model' + str(loopNumber) + '_2.pt')
        #if False:#loopNumber != 1:
        #    model1 = torch.load('./recursive/' + str(loopNumber) + '/Models/' + 'ExecutionProbability' + '/4.pt')#temp 4
        #model2 = torch.load('./recursive/' + str(loopNumber) + '/Models/' + 'expectedProfit' + '/4.pt')
        model = torch.load('./recursive/' + str(loopNumber) + '/Models/' + 'predictBoth' + '/6.pt')
        nameN = len(names)

        #nameN = len(names)
        M = 1000
        N = 100000
        #M1 = 1000000 * 2#10
        #M = 10
        #N = 10000
        losses = []


        name = names[nameNum] #Temporary
        #name = names[0] #Temporary

        print (name)
        if codeNum % 2 == 0:
            isBuy = True
        else:
            isBuy = False
        #for isBuy in [True, False]: #[True, False]: Make both true false

        namePart = ''
        if isBuy:
            namePart = namePart + "Bid_"
            oType = "B"
        else:
            namePart = namePart + "Ask_"
            oType = "S"
        namePart = namePart + name

        notDone = True
        #outputFull = np.array([])
        outputFull = np.zeros((0, 2))
        num = 0 #Temporary
        #num = 10 * nameNum #Temporary

        allShares, order, spread, args =  doXsubset4(name, isBuy)

        #print (spread.shape)
        #print (args.shape)
        #time2 = time.time()
        #quit()

        allShares = torch.tensor(allShares).float()
        order = torch.tensor(order).float()
        spread = torch.tensor(spread).float()

        lastOne = ((order.shape[0] - 1) // N) + 1

        for a in range(0,  lastOne):
            #print (a)
            allShares1 = allShares[a*N:(a+1)*N]
            order1 = order[a*N:(a+1)*N]
            spread1 = spread[a*N:(a+1)*N]
            #print (order1.shape)
            #if False:#loopNumber != 1:
            #    output1 = model1(allShares1, order1, spread1)
            #    output1 = torch.sigmoid(output1)
            #    output1 = output1.data.numpy()
            #else:
            #    output1 = np.ones(spread1.shape[0])
            #
            #output2 =  model2(allShares1, order1, spread1)
            #output2 = output2.data.numpy()
            #output = np.array([output1, output2]).T

            output = model(allShares1, order1, spread1)
            output[:, 0] = torch.sigmoid(output[:, 0])
            output = output.data.numpy()
            outputFull = np.concatenate((outputFull, output))

        #print (outputFull.shape)
        outputFull = np.array([args, outputFull[:, 0], outputFull[:, 1]]).T

        #print (outputFull.shape)
        #print (outputFull.shape)
        #quit()
        print ("Done Code: " + str(codeNum))
        np.savez_compressed('./recursive/' + str(loopNumber) + '/samples/prob_' + oType + '_' + name + '.npz', outputFull)
        #'''


    def saveComponent(startPoint):
        for a in range(0, 6):
            if startPoint + (a * 5) < (filesRun[1] * 2):
                saveComponent_sub(startPoint + (a * 5))
            #time.sleep(1)

    for a in range(0, 30):
        saveComponent_sub(a)
    #multiProcessor(saveComponent, 0, 4, 4, doPrint = True)
    #multiProcessor(saveComponent, 0, 15, 5, doPrint = True)
    #multiProcessor(saveComponent, 0, 5, 5, doPrint = True)
    #quit()
    #saveComponent(0)

    #saveComponent_sub(startPoint + (a * 5))

    #multiProcessor(saveComponent, 0, 5, 5, doPrint = True)

#savePredictions3(1, (0, 15))
#quit()


def savePredictions4(loopNumber, filesRun):
    print ("Saving Prediction 4")

    def saveComponent_sub(codeNum, loopNumber):

        #if False:#loopNumber != 0:
        #    model1 = torch.load('./recursive/' + str(loopNumber) + '/Models/' + 'ExecutionProbability' + '/4.pt')#temp 4
        #model2 = torch.load('./recursive/' + str(loopNumber) + '/Models/' + 'expectedProfit' + '/4.pt')
        model = torch.load('./recursive/' + str(loopNumber) + '/Models/' + 'predictBoth' + '/6.pt')

        #nameN = len(names)

        #nameN = len(names)
        M = 1000
        N = 100000

        for c in range(0, 5): #5
        #c = codeNum
            print (c)
            endingName2 = '_' + str(c)

            allShares = torch.load('./recursive/0/neuralInputs/2/Shares_g0' + endingName2 + '.pt')
            order = torch.load('./recursive/0/neuralInputs/2/order_g0' + endingName2 + '.pt')
            spread = torch.load('./recursive/0/neuralInputs/2/spread_g0' + endingName2 + '.pt')
            #profit = loadnpz('./recursive/0/neuralInputs/2/profit_g0' + endingName2 + '.npz')
            profitMean = 0.0#np.mean(profit[:, 0])
            #del profit

            lastOne = ((order.shape[0] - 1) // N) + 1

            outputFull = np.zeros((0, 2))

            for a in range(0,  lastOne):
                #print (a)
                allShares1 = allShares[a*N:(a+1)*N]
                order1 = order[a*N:(a+1)*N]
                spread1 = spread[a*N:(a+1)*N]

                output = model(allShares1, order1, spread1)
                output[:, 0] = torch.sigmoid(output[:, 0])
                output[:, 1] = output[:, 1] + profitMean
                output = output.data.numpy()
                outputFull = np.concatenate((outputFull, output))

            np.savez_compressed('./recursive/' + str(loopNumber) + '/samplesMix/prob_' + str(c) + '.npz', outputFull)
            print ("Done Code: " + str(codeNum))



    #def saveComponent(startPoint):
    #    for a in range(0, 6):
    #        if startPoint + (a * 5) < (filesRun[1] * 2):
    #            saveComponent_sub(startPoint + (a * 5))
    #        #time.sleep(1)

    #for a in range(0, 30):
    #    saveComponent_sub(0)

    saveComponent_sub(0, loopNumber)

def autoTrainSeller(loopNumber):
    print ("Training Model")
    for trainN in range(0, 4): #4
        names = giveFNames()
        print ("trainN ", trainN)

        if trainN == 0:
            model = orderProfitModel()
        else:
            #trainN = 4
            model = torch.load('./Models/group2/model' + str(int(loopNumber+1)) + '_' + str(trainN) + '.pt')

        #model = torch.load('./Models/group2/model' + str(int(loopNumber+1)) + '_' + str(3) + '.pt')


        learningRate = 2 * (10 ** (-1 - trainN))
        optimizer = torch.optim.SGD(model.parameters(), lr = learningRate )# * 0.01


        M = 200
        #N = 10000
        N = 10000
        #M = 10
        #N = 10000
        losses = []
        lossCor = []
        for iter in range(0, 1): #2
            for c in range(0, 5): #5
                #print (c)
                endingName2 = '_' + str(c)# + '.npz'

                allShares = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/2/Shares' + endingName2 + '.npz')#[:100000]
                order = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/2/order' + endingName2 + '.npz')#[:100000]
                spread = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/2/spread' + endingName2 + '.npz')#[:100000]
                profit = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/2/profit' + endingName2 + '.npz')#[:100000]

                #print (np.mean(profit))

                args = np.random.permutation(allShares.shape[0])
                allShares = allShares[args]
                order = order[args]
                spread = spread[args]
                profit = profit[args]

                allShares = torch.tensor(allShares).float()
                order = torch.tensor(order).float()
                spread = torch.tensor(spread).float()
                profit = torch.tensor(profit).float()

                output = model(allShares, order, spread)
                #print (np.mean(output.data.numpy()))
                #quit()

                outputMean = 0.0

                for a in range(0, profit.shape[0] // N):
                    #print ('a', a)
                    allShares1 = allShares[a*N:(a+1)*N]
                    order1 = order[a*N:(a+1)*N]
                    spread1 = spread[a*N:(a+1)*N]
                    profit1 = profit[a*N:(a+1)*N]

                    #np.savez_compressed('./temporary/tempNeuralInput/allShares2.npz', allShares1)
                    #np.savez_compressed('./temporary/tempNeuralInput/order2.npz', order1)
                    #np.savez_compressed('./temporary/tempNeuralInput/spread2.npz', spread1)
                    #np.savez_compressed('./temporary/tempNeuralInput/profit2.npz', profit1)
                    #quit()

                    #print (spread1.shape)
                    output = model(allShares1, order1, spread1)

                    #print (np.mean(output.data.numpy()))
                    #quit()

                    outputMean += np.mean(output.data.numpy())

                    #plt.hist(output.data.numpy(), bins=100)
                    #plt.show()
                    #quit()

                    loss = torch.mean((output - profit1) ** 2.0) - torch.mean((profit1) ** 2.0)
                    losses.append(loss.data.numpy())
                    lossCor.append(scipy.stats.pearsonr(output.data.numpy(), profit1.data.numpy())[0] )

                    #print ("A")
                    #print (scipy.stats.pearsonr(output.data.numpy(), spread1.data.numpy()))
                    #print (scipy.stats.pearsonr(spread1.data.numpy(), profit1.data.numpy()))

                    if (a + 1) % M == 0:
                        lossGroup = np.mean(np.array(losses)[-M:])
                        lossCorGroup = np.mean(np.array(lossCor)[-M:])
                        #print (  a   )
                        print ((a // M) + ( c * (profit.shape[0] //  (N * M)) ))
                        #print (scipy.stats.pearsonr(output.data.numpy(), profit1.data.numpy()))
                        print (lossGroup)
                        print (lossCorGroup)
                        #torch.save(model, './Models/group2/model2_3.pt')

                        #torch.save(model, './Models/group2/model' + str(int(loopNumber+1)) + '_' + str(int(trainN+1)) + '.pt')

                        #plt.hist(output.data.numpy())
                        #plt.show()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                del allShares
                del order
                #del allShares_
                #del order_

#autoTrainSeller(3)
#quit()

def autoTrainSeller_v2(loopNumber, bidCutOff, askCutOff, useCancel=True, predictProb=False):
    print ("Training Model")
    for trainN in range(0, 4): #4
        names = giveFNames()
        print ("trainN ", trainN)

        #namePart = 'deleteCancel'
        #if useCancel:
        #    namePart = 'punishCancel'
        namePart = 'expectedProfit2'
        iterN = 2
        if predictProb:
            namePart = 'ExecutionProbability'
            iterN = 1
        #else:
        #    if trainN == 0:
        #        iterN = 4

        #'''
        if trainN == 0:
            model = orderProfitModel()

            #model = torch.load('./recursive/' + str(int(loopNumber+1)) + '/Models/' + namePart + '/'  + str(1) + '.pt')
        else:
            model = torch.load('./recursive/' + str(loopNumber) + '/Models/' + namePart + '/'  + str(trainN) + '.pt')
        #'''


        #model = torch.load('./Models/group2/model' + str(int(loopNumber+1)) + '_' + str(4) + '.pt')


        learningRate = 2 * (10 ** (-1 - trainN))

        learningRate = learningRate * 0.1

        if predictProb:
            learningRate = learningRate * 10.0## * 100.0#10.0#0.01

        optimizer = torch.optim.SGD(model.parameters(), lr = learningRate )# * 0.01

        #cancelPoint = -0.9

        cancelCutOff = min(bidCutOff, askCutOff)
        #For now. It's to difficult to do the other more accurate way at the moment.

        ep1 = 0.001
        #M = 200
        M = 20
        N = 10000
        #N = 100000
        #learningRate = learningRate * 10.0
        #M = 10
        #N = 10000
        losses = []
        lossCor = []

        times1 = 0
        times2 = 0
        times3 = 0
        times4 = 0
        times5 = 0
        times6 = 0
        times7 = 0
        for iter in range(0, iterN): #2
            for c in range(0, 5): #5
                print (c)
                endingName2 = '_' + str(c)# + '.npz'
                time1 = time.time()
                time2 = time.time()
                #allShares = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/2/Shares' + endingName2 + '.npz')#[:100000]
                #order = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/2/order' + endingName2 + '.npz')#[:100000]
                #spread = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/2/spread' + endingName2 + '.npz')#[:100000]
                allShares = torch.load('./recursive/0/neuralInputs/2/Shares' + endingName2 + '.pt')
                order = torch.load('./recursive/0/neuralInputs/2/order' + endingName2 + '.pt')
                spread = torch.load('./recursive/0/neuralInputs/2/spread' + endingName2 + '.pt')
                profit = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/2/profit' + endingName2 + '.npz')#[:100000]

                times2 += time.time() - time2
                #allShares[:, :, 0] = np.sum(allShares, axis=2)
                #allShares[:, :, 1:] = 0
                #print (allShares.shape)
                #quit()
                #print (np.mean(profit[:, 1]))
                #quit()

                time3 = time.time()

                if (useCancel == False) and (predictProb == False):
                    #print ("LZ")
                    argsNonCancel = np.argwhere(profit[:, 1] == 0)[:, 0]
                    profit = profit[argsNonCancel][:, 0]
                    argsNonCancel = torch.tensor(argsNonCancel).long()
                    allShares = allShares[argsNonCancel]
                    order = order[argsNonCancel]
                    spread = spread[argsNonCancel]
                #else:
                #    profit[:, 0][profit[:, 1] == 1] = cancelCutOff #cancelPoint

                #print (np.mean(profit))

                #args = np.random.permutation(allShares.shape[0]) #Already  randomly permuted in saving
                #allShares = allShares[args]
                #order = order[args]
                #spread = spread[args]
                #profit = profit[args]

                #allShares = torch.tensor(allShares).float()
                #order = torch.tensor(order).float()
                #spread = torch.tensor(spread).float()
                profit = torch.tensor(profit).float()

                times3 += time.time() - time3
                time4 = time.time()

                for a in range(0, profit.shape[0] // N):
                    #print (a, profit.shape[0] // N)

                    allShares1 = allShares[a*N:(a+1)*N]
                    order1 = order[a*N:(a+1)*N]
                    spread1 = spread[a*N:(a+1)*N]
                    profit1 = profit[a*N:(a+1)*N]

                    time5 = time.time()
                    output = model(allShares1, order1, spread1)
                    times5 += time.time() - time5

                    #outputnp = output.data.numpy()
                    #profit1np = profit1.data.numpy()
                    #valArgs = np.argwhere(np.logical_or(profit1np[:, 1] == 0, outputnp > cancelCutOff))[:, 0]
                    #profit1np = profit1np[valArgs]
                    #valArgs = torch.tensor(valArgs)#.astype(long)
                    #output = output[valArgs]
                    #profit1_type = profit1[valArgs][:, 1]
                    #profit1_type = profit1[:, 1]
                    #profit1_type = (profit1_type * 0) + 1.0 #was 4
                    #profit1 = profit1[valArgs][:, 0]
                    #profit1 = profit1[:, 0]

                    time6 = time.time()
                    if predictProb:
                        profit1_type = profit1[:, 1]
                        #output = torch.sigmoid(output) * 0.998)
                        loss =  -1.0 * torch.mean( ((1 - profit1_type) * torch.log(torch.sigmoid(output) + ep1)) + (profit1_type * torch.log(1 - torch.sigmoid(output)  + ep1)) )

                        #lossCor.append(0)
                        lossCor.append(scipy.stats.pearsonr( -1.0 * torch.sigmoid(output).data.numpy(), profit1_type.data.numpy()) [0] )
                    else:
                        loss = torch.mean((output - profit1) ** 2.0) - torch.mean((profit1) ** 2.0)
                        #lossCor.append(scipy.stats.pearsonr(output.data.numpy()[profit1np[:, 1] == 0], profit1.data.numpy()[profit1np[:, 1] == 0]) [0] )
                        lossCor.append(scipy.stats.pearsonr(output.data.numpy(), profit1.data.numpy()) [0] )

                    #loss = torch.mean(((output - profit1) ** 2.0) * profit1_type) - torch.mean((profit1) ** 2.0)
                    losses.append(loss.data.numpy())

                    times6 += time.time() - time6


                    if (a + 1) % M == 0:
                        lossGroup = np.mean(np.array(losses)[-M:])
                        lossCorGroup = np.mean(np.array(lossCor)[-M:])
                        #print ('T')
                        #print (lossGroup)
                        #print (lossCorGroup)
                    #    print (torch.sigmoid(output))

                    time7 = time.time()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    times7 += time.time() - time7

                times4 = time.time() - time4

                #lossGroup = np.mean(np.array(losses)[-M:])
                #lossCorGroup = np.mean(np.array(lossCor)[-M:])
                lossGroup = np.mean(np.array(losses)[-(profit.shape[0] // N):])
                lossCorGroup = np.mean(np.array(lossCor)[-(profit.shape[0] // N):])

                #print ((a // M) + ( c * (profit.shape[0] //  (N * M)) ))
                #print (scipy.stats.pearsonr(output.data.numpy(), profit1.data.numpy()))
                print ('a', c)
                print (lossGroup)
                print (lossCorGroup)


                fullName = './recursive/' + str(loopNumber) + '/Models/' + namePart + '/'  + str(int(trainN+1)) + '.pt' #1_

                #torch.save(model, fullName)

                del allShares
                del order

                times1 += time.time() - time1


                print ('A')
                print (times1)
                print (times2)
                print (times3)
                print (times4)
                print (times5)
                print (times6)
                print (times7)



def autoTrainSeller_v3(loopNumber, bidCutOff, askCutOff, useCancel=True, predictProb=False):
    print ("Training Model")
    for trainN in range(0, 6): #4
        names = giveFNames()
        print ("trainN ", trainN)

        #namePart = 'deleteCancel'
        #if useCancel:
        #    namePart = 'punishCancel'

        #namePart = 'expectedProfit'
        iterN = 1
        #if predictProb:
        #    namePart = 'ExecutionProbability'
        #    iterN = 1

        namePart = 'predictBoth'
        #namePart = 'expectedProfit2'

        iterN = 4

        #if trainN == 0:
        #    iterN = 2
        #if trainN >= 4:
        #    iterN = 4

        #'''
        if trainN == 0:
            model = orderProfitModel4()
            #model = orderProfitModel3()
            #print ("Z")
            #torch.save(model, './temporary/fakeModel.pt')
            #quit()


            #model = torch.load('./recursive/' + str(loopNumber) + '/Models/' + namePart + '/'  + str(1) + '.pt')
            #model = torch.load('./recursive/' + str(loopNumber-1) + '/Models/' + namePart + '/'  + str(1) + '.pt')
        else:
            #trainN = 4
            #model = torch.load('./Models/group2/model' + str(int(loopNumber+1)) + '_' + str(trainN) + '.pt')
            #model = torch.load('./recursive/' + str(int(loopNumber+1)) + '/Models/' + namePart + '/1_'  + str(trainN) + '.pt')
            #model = torch.load('./recursive/' + str(int(loopNumber+1)) + '/Models/' + namePart + '/'  + str(trainN) + '.pt')
            #if trainN == 4:
            #    model = torch.load('./recursive/' + str(loopNumber) + '/Models/' + namePart + '/'  + str(trainN+1) + '.pt')
            model = torch.load('./recursive/' + str(loopNumber) + '/Models/' + namePart + '/'  + str(trainN) + '.pt')
        #'''


        #model = torch.load('./Models/group2/model' + str(int(loopNumber+1)) + '_' + str(4) + '.pt')


        learningRate = 0.01 * (2 ** (-trainN)) * 0.5 #* 0.1

        #learningRate = learningRate * 2

        #if predictProb:
        #    learningRate = learningRate * 10.0## * 100.0#10.0#0.01

        #optimizer = torch.optim.SGD(model.parameters(), lr = learningRate  )# * 0.01
        optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)
        #optimizer = torch.optim.Adam(model.parameters(), lr = learningRate * 0.01)

        #cancelPoint = -0.9

        cancelCutOff = min(bidCutOff, askCutOff)
        #For now. It's to difficult to do the other more accurate way at the moment.

        ep1 = 0.001
        #M = 200
        #M = 20
        M = 10
        N = 10000
        #N = 100000
        #learningRate = learningRate * 10.0
        #M = 10
        #N = 10000
        losses1 = []
        losses2 = []
        losses3 = []
        lossCor1 = []
        lossCor2 = []

        times1 = 0
        times2 = 0
        times3 = 0
        times4 = 0
        times5 = 0
        times6 = 0
        times7 = 0
        for iter in range(0, iterN): #2
            for c in [0, 1, 2]:#range(0, 5): #5
                print (c)
                endingName2 = '_' + str(c)# + '.npz'
                #endingName2 = '_' + str(0)
                time1 = time.time()
                time2 = time.time()
                #allShares = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/2/Shares' + endingName2 + '.npz')#[:100000]
                #order = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/2/order' + endingName2 + '.npz')#[:100000]
                #spread = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/2/spread' + endingName2 + '.npz')#[:100000]
                allShares = torch.load('./recursive/0/neuralInputs/2/Shares_g0' + endingName2 + '.pt') * 0.0
                order = torch.load('./recursive/0/neuralInputs/2/orderExtra_g0' + endingName2 + '.pt')
                spread = torch.load('./recursive/0/neuralInputs/2/spread_g0' + endingName2 + '.pt') * 0.0

                profit = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/2/profit_g0' + endingName2 + '.npz')#[:100000]


                '''
                #orderForm = torch.sum(order[:, 1, 19-3:19+3], axis=1)
                orderForm = torch.sum(order[:, 0, 19-2:19+2], axis=1)
                #print (order[orderForm==1, 0, 19-2:19+2][:50])
                allShares = allShares[orderForm == 1]
                order = order[orderForm == 1]
                spread = spread[orderForm == 1]
                profit = profit[orderForm.data.numpy() == 1]
                '''


                order[:, 1, :10] = order[:, 1, :10] * 0
                order[:, 1, :] = order[:, 1, :] * 0.0#0.1
                #print (order[:10, 1, 10:12])
                order[:, 1, 11] = 0
                #quit()
                #print (order[:3, 1, :12])
                #quit()

                print (profit.shape)
                '''
                #print (spread.shape)
                #print (profit.shape)
                allShares1 = torch.mean(torch.mean(allShares[:, :20, :], axis=1), axis=1)
                #print (allShares1.shape)
                print (scipy.stats.pearsonr(spread.data.numpy(), profit[:, 0]))
                print (scipy.stats.pearsonr(allShares1.data.numpy(), profit[:, 0]))
                #depth = torch.sum(order[:, 1, :], axis=1).data.numpy()
                depth = order[:, 1, 11]
                order = order[:, 0, :].data.numpy()
                midpoint = order.shape[1] // 2


                print (scipy.stats.pearsonr(order[:, midpoint-3], profit[:, 0]))
                print (scipy.stats.pearsonr(depth, profit[:, 0]))

                #print (scipy.stats.pearsonr(depth[order[:, midpoint-3] == 1], profit[:, 0][order[:, midpoint-3] == 1]))
                #print (scipy.stats.pearsonr(depth[order[:, midpoint-2] == 1], profit[:, 0][order[:, midpoint-2] == 1]))
                #print (scipy.stats.pearsonr(depth[order[:, midpoint+1] == 1], profit[:, 0][order[:, midpoint+1] == 1]))
                #print (scipy.stats.pearsonr(depth[order[:, midpoint+2] == 1], profit[:, 0][order[:, midpoint+2] == 1]))
                #print (scipy.stats.pearsonr(order[:, 0, 10].data.numpy(), profit[:, 0]))
                quit()
                #profit = np.abs(profit) * 0.1
                #'''

                #endingName2 = '_' + str(1)
                allShares_test = torch.load('./recursive/0/neuralInputs/2/Shares_g1' + endingName2 + '.pt') * 0.0
                order_test = torch.load('./recursive/0/neuralInputs/2/orderExtra_g1' + endingName2 + '.pt')
                spread_test = torch.load('./recursive/0/neuralInputs/2/spread_g1' + endingName2 + '.pt') * 0.0
                profit_test = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/2/profit_g1' + endingName2 + '.npz')#[:100000]
                #profit_test = np.abs(profit_test) * 0.1
                profit_test = torch.tensor(profit_test).float()

                '''
                #orderForm = torch.sum(order_test[:, 1, 19-3:19+3], axis=1)
                orderForm = torch.sum(order_test[:, 0, 19-2:19+2], axis=1)
                #print (torch.mean(orderForm))
                #quit()
                allShares_test = allShares_test[orderForm == 1][:100000]
                order_test = order_test[orderForm == 1][:100000]
                spread_test = spread_test[orderForm == 1][:100000]
                profit_test = profit_test[orderForm.data.numpy() == 1][:100000]
                '''



                order_test[:, 1, :10] = order_test[:, 1, :10] * 0
                order_test[:, 1, :] = order_test[:, 1, :] * 0.0#0.1

                order_test[:, 1, 11] = 0

                times2 += time.time() - time2

                #allShares = torch.tensor(allShares).float()
                #order = torch.tensor(order).float()
                #spread = torch.tensor(spread).float()
                profit = torch.tensor(profit).float()

                time4 = time.time()

                for a in range(0, profit.shape[0] // N):
                    #print (a, profit.shape[0] // N)

                    allShares1 = allShares[a*N:(a+1)*N]
                    order1 = order[a*N:(a+1)*N]
                    spread1 = spread[a*N:(a+1)*N]
                    profit1 = profit[a*N:(a+1)*N]

                    time5 = time.time()
                    output = model(allShares1, order1, spread1)
                    #output = model(allShares1, spread1)
                    times5 += time.time() - time5

                    time6 = time.time()

                    profit1_type = profit1[:, 1]
                    #output = torch.sigmoid(output) * 0.998)
                    #loss1 = -1.0 * torch.mean( ((1 - profit1_type) * torch.log(torch.sigmoid(output[:, 0]) + ep1)) + (profit1_type * torch.log(1 - torch.sigmoid(output[:, 0])  + ep1)) )
                    #lossCor1.append(scipy.stats.pearsonr( -1.0 * torch.sigmoid(output[:, 0]).data.numpy(), profit1_type.data.numpy()) [0] )

                    profit2, output2 = profit1[profit1_type == 0, 0], output[profit1_type == 0, 1]

                    output2_np = output2.data.numpy()
                    profit2_np = profit2.data.numpy()
                    argsValid = np.argwhere(np.logical_or(output2_np > -2.0, profit2_np > -2.0))[:, 0]
                    argsValid = torch.tensor(argsValid).long()

                    loss2 = torch.mean(   (( output2[argsValid] - profit2[argsValid] ) ** 2.0) - torch.mean(profit2[argsValid] ** 2.0) )
                    lossCor2.append(scipy.stats.pearsonr(output2[argsValid].data.numpy(), profit2[argsValid].data.numpy()) [0] )

                    #loss = torch.mean(((output - profit1) ** 2.0) * profit1_type) - torch.mean((profit1) ** 2.0)
                    #loss = loss1 + loss2
                    loss = loss2

                    #l2_reg = torch.tensor(0.)
                    #for param in model.parameters():
                    #    l2_reg += torch.norm(param)
                    #print (l2_reg)
                    #quit()
                    #loss = loss + (0.001 * l2_reg)


                    #losses1.append(loss1.data.numpy())
                    losses2.append(loss2.data.numpy())
                    losses3.append(loss.data.numpy())



                    times6 += time.time() - time6


                    if (a + 1) % M == 0:
                        lossGroup = np.mean(np.array(losses2)[-M:])
                        lossCorGroup = np.mean(np.array(lossCor2)[-M:])

                        output_test = model(allShares_test, order_test, spread_test)
                        #output_test = model(allShares_test, spread_test)


                        print ('T')
                        print (lossGroup)
                        print (lossCorGroup)
                        print (scipy.stats.pearsonr(output_test[:, 1].data.numpy(), profit_test[:, 0].data.numpy()))
                    #    print (torch.sigmoid(output))

                    time7 = time.time()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    times7 += time.time() - time7

                times4 = time.time() - time4

                #lossGroup = np.mean(np.array(losses)[-M:])
                #lossCorGroup = np.mean(np.array(lossCor)[-M:])
                #lossGroup1 = np.mean(np.array(losses1)[-(profit.shape[0] // N):])
                lossGroup2 = np.mean(np.array(losses2)[-(profit.shape[0] // N):])
                lossGroup3 = np.mean(np.array(losses3)[-(profit.shape[0] // N):])
                #lossCorGroup1 = np.mean(np.array(lossCor1)[-(profit.shape[0] // N):])
                lossCorGroup2 = np.mean(np.array(lossCor2)[-(profit.shape[0] // N):])

                #print ((a // M) + ( c * (profit.shape[0] //  (N * M)) ))
                #print (scipy.stats.pearsonr(output.data.numpy(), profit1.data.numpy()))
                print ('a', c)
                #print (lossGroup1)
                print (lossGroup2)
                print (lossGroup3)
                #print (lossCorGroup1)
                print (lossCorGroup2)

                #quit()


                fullName = './recursive/' + str(loopNumber) + '/Models/' + namePart + '/'  + str(int(trainN+1)) + '.pt' #1_

                torch.save(model, fullName)

                del allShares
                del order

                times1 += time.time() - time1


                #print ('A')
                #print (times1)
                #print (times2)
                #print (times3)
                #print (times4)
                #print (times5)
                #print (times6)
                #print (times7)



def autoTrainSeller_compare(loopNumber, useCancel=True, predictProb=False):
    print ("Training Model")
    for trainN in range(0, 6): #4
        names = giveFNames()
        print ("trainN ", trainN)


        iterN = 2

        namePart = 'predictBoth'
        #namePart = 'expectedProfit2'


        if trainN == 0:
            iterN = 2

        #'''
        if trainN == 0:
            model1 = orderProfitModel2()
            model2 = orderProfitModel2()

            #model = torch.load('./recursive/' + str(loopNumber) + '/Models/' + namePart + '/'  + str(1) + '.pt')
            #model = torch.load('./recursive/' + str(loopNumber-1) + '/Models/' + namePart + '/'  + str(1) + '.pt')
        else:
            #trainN = 4
            #model = torch.load('./Models/group2/model' + str(int(loopNumber+1)) + '_' + str(trainN) + '.pt')
            #model = torch.load('./recursive/' + str(int(loopNumber+1)) + '/Models/' + namePart + '/1_'  + str(trainN) + '.pt')
            #model = torch.load('./recursive/' + str(int(loopNumber+1)) + '/Models/' + namePart + '/'  + str(trainN) + '.pt')
            #if trainN == 1:
            #    model = torch.load('./recursive/' + str(loopNumber) + '/Models/' + namePart + '/'  + str(trainN+1) + '.pt')
            model1 = torch.load('./recursive/' + str(loopNumber) + '/Models/' + namePart + '/1_'  + str(trainN) + '.pt')
            model2 = torch.load('./recursive/' + str(loopNumber) + '/Models/' + namePart + '/2_'  + str(trainN) + '.pt')
        #'''


        #model = torch.load('./Models/group2/model' + str(int(loopNumber+1)) + '_' + str(4) + '.pt')


        #learningRate = 2 * (10 ** (-1 - trainN))

        learningRate = 0.01 * (4 ** (-trainN)) * 0.5

        #learningRate = learningRate * 2

        #if predictProb:
        #    learningRate = learningRate * 10.0## * 100.0#10.0#0.01

        #optimizer = torch.optim.SGD(model.parameters(), lr = learningRate  )# * 0.01
        optimizer1 = torch.optim.Adam(model1.parameters(), lr = learningRate)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr = learningRate)
        #optimizer = torch.optim.Adam(model.parameters(), lr = learningRate * 0.01)

        #cancelPoint = -0.9

        #cancelCutOff = min(bidCutOff, askCutOff)
        #For now. It's to difficult to do the other more accurate way at the moment.

        ep1 = 0.001
        #M = 200
        M = 20
        N = 10000
        #N = 100000
        #learningRate = learningRate * 10.0
        #M = 10
        #N = 10000
        losses1 = []
        losses2 = []
        losses3 = []
        lossCor1 = []
        lossCor2 = []
        lossCor3 = []
        lossCor4 = []

        times1 = 0
        times2 = 0
        times3 = 0
        times4 = 0
        times5 = 0
        times6 = 0
        times7 = 0
        for iter in range(0, iterN): #2
            for c in range(0, 5): #5
                print (c)
                endingName2 = '_' + str(c)# + '.npz'
                time1 = time.time()
                time2 = time.time()
                #allShares = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/2/Shares' + endingName2 + '.npz')#[:100000]
                #order = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/2/order' + endingName2 + '.npz')#[:100000]
                #spread = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/2/spread' + endingName2 + '.npz')#[:100000]
                allShares = torch.load('./recursive/0/neuralInputs/2/Shares' + endingName2 + '.pt')
                order = torch.load('./recursive/0/neuralInputs/2/order' + endingName2 + '.pt')
                spread = torch.load('./recursive/0/neuralInputs/2/spread' + endingName2 + '.pt')


                profit = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/2/profit_noAdd' + endingName2 + '.npz')#[:100000]
                #output = loadnpz('./recursive/' + str(loopNumber) + '/samplesMix/prob' + endingName2 + '.npz')[:, 1]
                #profit[:, 0] = output

                profit = torch.tensor(profit).float()

                #times3 += time.time() - time3
                time4 = time.time()

                for a in range(0, profit.shape[0] // N):
                    #print (a, profit.shape[0] // N)

                    allShares1 = allShares[a*N:(a+1)*N]
                    order1 = order[a*N:(a+1)*N]
                    spread1 = spread[a*N:(a+1)*N]
                    profit1 = profit[a*N:(a+1)*N]

                    time5 = time.time()
                    output1 = model1(allShares1, order1, spread1)
                    output2 = model2(allShares1, order1, spread1)
                    times5 += time.time() - time5

                    time6 = time.time()


                    profit1_type = profit1[:, 1]
                    #loss1 = -1.0 * torch.mean( ((1 - profit1_type) * torch.log(torch.sigmoid(output[:, 0]) + ep1)) + (profit1_type * torch.log(1 - torch.sigmoid(output[:, 0])  + ep1)) )
                    #lossCor1.append(scipy.stats.pearsonr( -1.0 * torch.sigmoid(output[:, 0]).data.numpy(), profit1_type.data.numpy()) [0] )

                    profit2, output1_2, output2_2 = profit1[profit1_type == 0, 0], output1[profit1_type == 0, 1], output2[profit1_type == 0, 1]
                    loss1_2 = torch.mean(   (( output1_2 - profit2 ) ** 2.0) - torch.mean(profit2 ** 2.0) )
                    lossCor1.append(scipy.stats.pearsonr(output1_2.data.numpy(), profit2.data.numpy()) [0] )

                    loss2_2 = torch.mean(   (( output2_2 - profit2 ) ** 2.0) - torch.mean(profit2 ** 2.0) )
                    lossCor2.append(scipy.stats.pearsonr(output2_2.data.numpy(), profit2.data.numpy()) [0] )

                    #loss = torch.mean(((output - profit1) ** 2.0) * profit1_type) - torch.mean((profit1) ** 2.0)
                    #loss = loss2
                    #loss = loss1 + loss2


                    l2_reg_1 = torch.tensor(0.)
                    for param in model1.parameters():
                        l2_reg_1 += torch.norm(param)

                    loss1_2 = loss1_2 + (0.001 * l2_reg_1)

                    l2_reg_2 = torch.tensor(0.)
                    for param in model2.parameters():
                        l2_reg_2 += torch.norm(param)

                    loss2_2 = loss2_2 + (0.001 * l2_reg_2)

                    losses1.append(loss1_2.data.numpy())
                    losses2.append(loss2_2.data.numpy())

                    loss3 = torch.mean( (output1_2 - output2_2) ** 2.0) ** 0.5
                    lossCor3.append(scipy.stats.pearsonr(output1_2.data.numpy(), output2_2.data.numpy()) [0] )


                    lossCor4.append(scipy.stats.pearsonr(((output1_2+output2_2)/2).data.numpy(), profit2.data.numpy()) [0] )

                    losses3.append(loss3.data.numpy())



                    times6 += time.time() - time6


                    if (a + 1) % M == 0:
                        lossGroup1 = np.mean(np.array(losses1)[-M:])
                        lossGroup2 = np.mean(np.array(losses2)[-M:])
                        lossCorGroup1 = np.mean(np.array(lossCor1)[-M:])
                        lossCorGroup2 = np.mean(np.array(lossCor2)[-M:])
                        lossCorGroup4 = np.mean(np.array(lossCor4)[-M:])
                        print ('T')
                        print (lossGroup1)
                        print (lossGroup2)
                        print (lossCorGroup1)
                        print (lossCorGroup2)
                        print (lossCorGroup4)
                    #    print (torch.sigmoid(output))

                    time7 = time.time()
                    optimizer1.zero_grad()
                    loss1_2.backward()
                    optimizer1.step()

                    optimizer2.zero_grad()
                    loss2_2.backward()
                    optimizer2.step()
                    times7 += time.time() - time7

                times4 = time.time() - time4

                #lossGroup = np.mean(np.array(losses)[-M:])
                #lossCorGroup = np.mean(np.array(lossCor)[-M:])
                lossGroup1 = np.mean(np.array(losses1)[-(profit.shape[0] // N):])
                lossGroup2 = np.mean(np.array(losses2)[-(profit.shape[0] // N):])
                lossGroup3 = np.mean(np.array(losses3)[-(profit.shape[0] // N):])
                lossCorGroup1 = np.mean(np.array(lossCor1)[-(profit.shape[0] // N):])
                lossCorGroup2 = np.mean(np.array(lossCor2)[-(profit.shape[0] // N):])
                lossCorGroup3 = np.mean(np.array(lossCor3)[-(profit.shape[0] // N):])
                lossCorGroup4 = np.mean(np.array(lossCor4)[-(profit.shape[0] // N):])

                #print ((a // M) + ( c * (profit.shape[0] //  (N * M)) ))
                #print (scipy.stats.pearsonr(output.data.numpy(), profit1.data.numpy()))
                print ('a', c)
                print (lossGroup1)
                print (lossGroup2)
                print (lossGroup3)
                print (lossCorGroup1)
                print (lossCorGroup2)
                print (lossCorGroup3)
                print (lossCorGroup4)
                #quit()


                fullName = './recursive/' + str(loopNumber) + '/Models/' + namePart + '/1_'  + str(int(trainN+1)) + '.pt' #1_
                torch.save(model1, fullName)
                fullName = './recursive/' + str(loopNumber) + '/Models/' + namePart + '/2_'  + str(int(trainN+1)) + '.pt' #1_
                torch.save(model2, fullName)

                del allShares
                del order

                times1 += time.time() - time1


                #print ('A')
                #print (times1)
                #print (times2)
                #print (times3)
                #print (times4)
                #print (times5)
                #print (times6)
                #print (times7)



def trainCancelPredictor(loopNumber):
    print ("Training Model")
    for trainN in range(0, 4): #4
        names = giveFNames()
        print ("trainN ", trainN)
        namePart = 'fastCancel'

        iterN = 6
        if trainN == 0:
            model = fastCancelModel()
        else:
            model = torch.load('./recursive/' + str(loopNumber) + '/Models/' + namePart + '/'  + str(trainN) + '.pt')

        learningRate = 2 * (10 ** (-1 - trainN))

        #optimizer = torch.optim.SGD(model.parameters(), lr = learningRate  )# * 0.01
        optimizer = torch.optim.Adam(model.parameters(), lr = learningRate * 0.01)
        #optimizer = torch.optim.Adam(model.parameters(), lr = learningRate * 0.01)


        ep1 = 0.001
        #M = 200
        M = 20
        N = 10000
        #N = 100000
        #learningRate = learningRate * 10.0
        #M = 10
        #N = 10000
        losses1 = []
        lossCor1 = []
        lossCor2 = []

        times1 = 0
        times2 = 0
        times3 = 0
        times4 = 0
        times5 = 0
        times6 = 0
        times7 = 0
        for iter in range(0, iterN): #2
            for c in range(0, 5): #5
                print (c)
                endingName2 = '_' + str(c)# + '.npz'
                time1 = time.time()
                time2 = time.time()
                #allShares = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/2/Shares' + endingName2 + '.npz')#[:100000]
                #order = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/2/order' + endingName2 + '.npz')#[:100000]
                #spread = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/2/spread' + endingName2 + '.npz')#[:100000]
                allShares = torch.load('./recursive/0/neuralInputs/2/Shares' + endingName2 + '.pt')
                order = torch.load('./recursive/0/neuralInputs/2/order' + endingName2 + '.pt')
                spread = torch.load('./recursive/0/neuralInputs/2/spread' + endingName2 + '.pt')
                output = loadnpz('./recursive/' + str(loopNumber) + '/samplesMix/prob_' + str(c) + '.npz')
                outputCancels = np.zeros(output.shape[0])
                outputCancels[output[:, 1] < -0.8] = 1
                outputCancels = torch.tensor(outputCancels).float()

                spread = spread.reshape((spread.shape[0], 1))
                allShares = torch.sum(allShares, axis=2)
                order1 = torch.sum(order[:, 1, :], axis=1).reshape((order.shape[0], 1))
                #order1 = torch.matmul(order[:, 1, :], torch.tensor(np.arange(38).reshape(38, 1)).float())

                #order2 = order1.data.numpy().reshape((order1.shape[0],) )
                #print (order2.shape)
                #print (output.shape)
                #print (scipy.stats.pearsonr(output[:, 1], order2))
                #quit()
                order = order[:, 0, :]
                X = torch.cat((allShares, order, order1, spread), 1)

                profit = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/2/profit' + endingName2 + '.npz')#[:100000]


                profit = torch.tensor(profit).float()

                for a in range(0, profit.shape[0] // N):
                    #print (a)
                    #print (a, profit.shape[0] // N)

                    X1 = X[a*N:(a+1)*N]
                    #prediction[a*N:(a+1)*N]
                    outputCancels1 = outputCancels[a*N:(a+1)*N]
                    profit1 = profit[a*N:(a+1)*N]
                    prediction = model(X1)

                    #plt.hist(prediction.data.numpy(), bins=100)
                    #plt.show()
                    #quit()

                    loss = -1.0 * torch.mean( (outputCancels1 * torch.log(prediction + ep1)) + ((1-outputCancels1)  * torch.log(1 - prediction  + ep1)) )

                    scipy.stats.pearsonr( prediction.data.numpy(), outputCancels1.data.numpy() )
                    losses1.append(loss.data.numpy())

                    outputCancels2 = outputCancels1.data.numpy()
                    prediction2 = np.zeros(N)

                    valCut = float(prediction[np.argsort(prediction.data.numpy()) [(-N // 10)]].data.numpy())
                    #print (valCut)
                    #quit()
                    prediction2[np.argsort(prediction.data.numpy()) [(-N // 10):]] = 1
                    sum1 = np.sum(outputCancels2[prediction2 == 0])
                    sum2 = np.sum(outputCancels2)

                    cancelPercent = np.mean(outputCancels2[prediction2 == 1])


                    #plt.hist(prediction.data.numpy()[prediction2 == 1], bins=100)
                    #plt.show()


                    lossCor1.append(sum1 / sum2)
                    lossCor2.append(cancelPercent)

                    if (a + 1) % M == 0:
                        lossGroup = np.mean(np.array(losses1)[-M:])
                        lossCorGroup = np.mean(np.array(lossCor1)[-M:])
                        #print ('T')
                        #print (lossGroup)
                        #print (lossCorGroup)
                    #    print (torch.sigmoid(output))

                    #print ("E")
                    time7 = time.time()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    times7 += time.time() - time7
                    #print ("F")


                print ('a', c)
                lossGroup = np.mean(np.array(losses1)[-M:])
                lossCorGroup = np.mean(np.array(lossCor1)[-M:])
                lossCorGroup2 = np.mean(np.array(lossCor2)[-M:])
                print (lossGroup)
                print (lossCorGroup)
                print (lossCorGroup2)

                #print ("Val Cut", valCut)
                #quit()


                fullName = './recursive/' + str(loopNumber) + '/Models/' + namePart + '/'  + str(int(trainN+1)) + '.pt' #1_

                torch.save(model, fullName)

                del allShares
                del order
    return valCut



#trainCancelPredictor(0)
#quit()


def trainSeller(loopNumber):
    print ("Training Model")
    names = giveFNames()
    model = orderProfitModel()
    #model = torch.load('./Models/group2/model1_2.pt')
    #model = torch.load('./Models/group2/model2_2.pt')
    #model = torch.load('./Models/group2/model2_2.pt')
    #model = torch.load('./Models/group2/model' + str(loopNumber) + '_4.pt')
    optimizer = torch.optim.SGD(model.parameters(), lr = 2e-3 )# * 0.01

    #recursiveCancels = True
    #loopNumber = 1

    #nameN = len(names)
    M = 200
    #N = 10000
    N = 10000
    #M = 10
    #N = 10000
    losses = []
    lossCor = []
    for iter in range(0, 10):
        for c in range(0, 5):
            #print (c)
            endingName2 = '_' + str(c)# + '.npz'

            #noSell = ''# 'noSell_'
            #if recursiveCancels:
            #    allShares = loadnpz('./resultData/Recur1_temporaryNeuralInput2/'  + noSell + 'Shares' + endingName2)
            #    order = loadnpz('./resultData/Recur1_temporaryNeuralInput2/'  + noSell + 'order' + endingName2)
            #    spread = loadnpz('./resultData/Recur1_temporaryNeuralInput2/'  + noSell + 'spread' + endingName2)
            #    profit = loadnpz('./resultData/Recur1_temporaryNeuralInput2/'  + noSell + 'profit' + endingName2)
            #else:
            #    allShares = loadnpz('./inputData/ITCH_LOB/temporaryNeuralInput2/Shares' + endingName2)
            #    order = loadnpz('./inputData/ITCH_LOB/temporaryNeuralInput2/order' + endingName2)
            #    spread = loadnpz('./inputData/ITCH_LOB/temporaryNeuralInput2/spread' + endingName2)
            #    profit = loadnpz('./inputData/ITCH_LOB/temporaryNeuralInput2/profit' + endingName2)

            allShares = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/2/Shares' + endingName2 + '.npz')
            order = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/2/order' + endingName2 + '.npz')
            spread = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/2/spread' + endingName2 + '.npz')
            profit = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/2/profit' + endingName2 + '.npz')

            #print ("Profit Mean")
            #print (np.mean(profit))

            #print (profit.shape)
            #plt.hist(profit, bins=100)
            #plt.show()
            #quit()

            #print (scipy.stats.pearsonr(profit, spread))
            #quit()

            args = np.random.permutation(allShares.shape[0])
            allShares = allShares[args]
            order = order[args]
            spread = spread[args]
            profit = profit[args]

            allShares = torch.tensor(allShares).float()
            order = torch.tensor(order).float()
            spread = torch.tensor(spread).float()
            profit = torch.tensor(profit).float()

            outputMean = 0.0

            for a in range(0, profit.shape[0] // N):
                #print ('a', a)
                allShares1 = allShares[a*N:(a+1)*N]
                order1 = order[a*N:(a+1)*N]
                spread1 = spread[a*N:(a+1)*N]
                profit1 = profit[a*N:(a+1)*N]

                #np.savez_compressed('./temporary/tempNeuralInput/allShares2.npz', allShares1)
                #np.savez_compressed('./temporary/tempNeuralInput/order2.npz', order1)
                #np.savez_compressed('./temporary/tempNeuralInput/spread2.npz', spread1)
                #np.savez_compressed('./temporary/tempNeuralInput/profit2.npz', profit1)
                #quit()

                #print (spread1.shape)
                output = model(allShares1, order1, spread1)

                outputMean += np.mean(output.data.numpy())

                #plt.hist(output.data.numpy(), bins=100)
                #plt.show()
                #quit()

                loss = torch.mean((output - profit1) ** 2.0) - torch.mean((profit1) ** 2.0)
                losses.append(loss.data.numpy())
                lossCor.append(scipy.stats.pearsonr(output.data.numpy(), profit1.data.numpy())[0] )

                #print ("A")
                #print (scipy.stats.pearsonr(output.data.numpy(), spread1.data.numpy()))
                #print (scipy.stats.pearsonr(spread1.data.numpy(), profit1.data.numpy()))

                if (a + 1) % M == 0:
                    lossGroup = np.mean(np.array(losses)[-M:])
                    lossCorGroup = np.mean(np.array(lossCor)[-M:])
                    #print (  a   )
                    print ((a // M) + ( c * (profit.shape[0] //  (N * M)) ))
                    #print (scipy.stats.pearsonr(output.data.numpy(), profit1.data.numpy()))
                    print (lossGroup)
                    print (lossCorGroup)
                    torch.save(model, './Models/group2/model2_3.pt')

                    #plt.hist(output.data.numpy())
                    #plt.show()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #print ("Output Mean")
            #print ( outputMean / (profit.shape[0] // N))

            del allShares
            del order
            #del allShares_
            #del order_

    #print (profit.shape)
    quit()

#trainSeller(1)
#quit()


#def autoCutOff(loopNumber, percentage)

def saveMajorCancelations(loopNumber, filesRun=(0, 15)):
    def saveComponent(nameNum):
        names = giveFNames()
        name = names[nameNum]
        print (name)
        for isBuy in [True, False]:
            if isBuy:
                LOB_share = loadnpz('./inputData/ITCH_LOB/BuyLOB/' + name + '.npz', allow_pickle=True)
                oType = "B"
            else:
                LOB_share = loadnpz('./inputData/ITCH_LOB/SellLOB/' + name + '.npz', allow_pickle=True)
                oType = "S"

            finalData = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')
            YData = loadnpz('./recursive/0/orderData/withCancels/Y_' + oType + '_' + name + '.npz')
            Yargs = loadnpz('./recursive/0/orderData/withCancels/Trans_' + oType + '_' + name + '.npz')


            output = loadnpz('./recursive/' + str(loopNumber) + '/outputCancels/' + oType + '_' + name + '.npz')
            data = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz', allow_pickle=True)
            argsBuy = np.argwhere(data[:, 4] == oType)[:, 0]
            miliseconds = data[:, 0].astype(int)
            del data

            position = finalData[:, 0]
            price = YData[:, 1]
            quePos = YData[:, 2]
            lobPos = YData[:, 0] #

            position = position[output == 1]
            price = price[output == 1]
            quePos = quePos[output == 1]
            lobPos = lobPos[output == 1] #

            priceU = np.unique(price)
            #print (position.shape)

            price_argsort, indicesStart, indicesEnd = fastAllArgwhere(price)

            #times1 = 0
            #times2 = 0
            cancelations = []
            for a in range(0, len(LOB_share)): #800 to 0
                cancelations.append([])
                if a in priceU:
                    #time1_1 = time.time()
                    #pos1_c = position[price == a]
                    #que1_c = quePos[price == a]
                    arg1 = np.argwhere(priceU == a)[0][0]
                    start1, end1 = indicesStart[arg1], indicesEnd[arg1] + 1
                    pos1 = position[price_argsort[start1:end1]]
                    que1 = quePos[price_argsort[start1:end1]]
                    lobPos1 = lobPos[price_argsort[start1:end1]] #
                    posSort = np.argsort(pos1)
                    pos1 = pos1[posSort]
                    que1 = que1[posSort]
                    lobPos1 = lobPos1[posSort] #
                    if pos1.shape[0] > 0:
                        #time2_1 = time.time()
                        args = np.array(LOB_share[a][0])
                        args = argsBuy[args]
                        miliseconds_args = miliseconds[args]

                        if args.shape[0] > 1:
                            #inAr1 = stefan_nextAfter(args[:-1]+1, pos1) #sep 6 added +1
                            inAr1 = stefan_nextAfter(args[:-1], pos1)
                            inAr2 = stefan_firstBefore(args[1:], pos1+1)
                            for b in range(0, inAr1.shape[0]):

                                #With Delay
                                '''
                                inAr_1 = inAr1[b]
                                inAr_2 = inAr2[b] + 1
                                if inAr_2 > inAr_1:
                                    pos2 = pos1[inAr_1:inAr_2]
                                    miliseconds_pos2 = miliseconds[pos2]
                                    #print (miliseconds_pos2.shape)
                                    #print (miliseconds_pos2[[miliseconds_pos2 > miliseconds_args[b]]].shape)
                                    que2 = np.unique(que1[inAr_1:inAr_2][miliseconds_pos2 > miliseconds_args[b]])
                                    cancelations[a].append(que2)
                                else:
                                    cancelations[a].append(np.array([]))
                                '''

                                que2 = np.unique(que1[inAr1[b]:inAr2[b]+1])
                                cancelations[a].append(que2)

                                #que3 = np.unique(que1[lobPos1 == b])
                                #if (que2.shape[0] != 0) or (que3.shape[0] != 0):
                                #    if np.mean(np.abs(que2 - que3)) != 0:
                                #        print ("Done1")
                                #        quit()
                                #        print ('A')
                                #        print (np.mean(np.abs(que2 - que3)))
                        #time2_2 = time.time() - time2_1
                        #times2 += time2_2
                    #time1_2 = time.time() - time1_1
                    #times1 += time1_2
            #quit()
            np.savez_compressed('./recursive/' + str(loopNumber) + '/choosenCancels/major/' + oType + '_' + name + '.npz', cancelations)



    if loopNumber > 0:
        #saveComponent(0)
        #quit()
        #multiProcessor(saveComponent, 0, 15, 4, doPrint = True)
        multiProcessor(saveComponent, filesRun[0], filesRun[1], 4, doPrint = True)


#print (stefan_nextAfter(np.array([0, 1, 2]), np.array([0, 1, 2])))
#quit()

#saveMajorCancelations(2, filesRun=(0, 15))
#quit()

def saveMajorEmptyCancelations(loopNumber, filesRun=(0, 15)):
    def saveComponent(nameNum):
        names = giveFNames()
        name = names[nameNum]
        print (name)
        for isBuy in [True, False]:

            if isBuy:
                #Yargs = loadnpz('./inputData/Processed/inputXY/Trans_Bid_' + name + '.npz')
                #finalData = loadnpz('./inputData/Processed/inputXY/X_Bid_' + name + '.npz')
                #YData = loadnpz('./inputData/Processed/inputXY/Y_Bid_' + name + '.npz')

                LOB_share = loadnpz('./inputData/ITCH_LOB/BuyLOB/' + name + '.npz', allow_pickle=True)
                oType = "B"
                #priceCutOff = bidCutOff
            else:
                #Yargs = loadnpz('./inputData/Processed/inputXY/Trans_Ask_' + name + '.npz')
                #finalData = loadnpz('./inputData/Processed/inputXY/X_Ask_' + name + '.npz')
                #YData = loadnpz('./inputData/Processed/inputXY/Y_Ask_' + name + '.npz')
                LOB_share = loadnpz('./inputData/ITCH_LOB/SellLOB/' + name + '.npz', allow_pickle=True)
                oType = "S"
                #priceCutOff = askCutOff

            finalData = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')
            YData = loadnpz('./recursive/0/orderData/withCancels/Y_' + oType + '_' + name + '.npz')
            Yargs = loadnpz('./recursive/0/orderData/withCancels/Trans_' + oType + '_' + name + '.npz')

            output = loadnpz('./recursive/' + str(loopNumber) + '/outputCancels/' + oType + '_' + name + '.npz')
            #output[:, 1][output[:, 0] < 0.5] = -2.0
            #output = output[:, 1]

            data = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz', allow_pickle=True)
            miliseconds = data[:, 0].astype(int)
            argsBuy = np.argwhere(data[:, 4] == oType)[:, 0]
            del data

            #priceCutOff = -0.39
            #priceCutOff = -0.77
            finalData = finalData[output == 1]
            YData = YData[output == 1]

            lobPos = YData[:, 0]
            price = YData[:, 1]
            goodArgs = np.zeros(lobPos.shape[0])
            for a in range(0, lobPos.shape[0]):
                size1 = len(LOB_share[price[a]][1][lobPos[a]])
                if size1 == 0:
                    goodArgs[a] = 1
            goodArgs = np.argwhere(goodArgs == 1)[:, 0]
            finalData = finalData[goodArgs]
            YData = YData[goodArgs]

            position = finalData[:, 0]
            price = YData[:, 1]
            lobPos = YData[:, 0]

            priceU = np.unique(price)

            price_argsort, indicesStart, indicesEnd = fastAllArgwhere(price)

            cancelations = []
            #print (len(LOB_share))
            count = 0
            for a in range(0, len(LOB_share)): #800 to 0
                #print (a)
                cancelations.append([])
                if a in priceU:

                    arg1 = np.argwhere(priceU == a)[0][0]
                    start1, end1 = indicesStart[arg1], indicesEnd[arg1] + 1
                    pos1 = position[price_argsort[start1:end1]]
                    lob1 = lobPos[price_argsort[start1:end1]]

                    #pos1 = position[price == a]
                    #lob1 = lobPos[price==a]
                    for b in range(0, len(LOB_share[a][0])):
                        if pos1[lob1 == b].shape[0] == 0:
                            cancelations[a].append(-1)
                        else:
                            count += 1

                            pos2 = np.min(pos1[lob1 == b]) #Sep 6
                            cancelations[a].append(pos2) #Sep 6

                            #This adds a time delay of 1 milisecond assuming no LOB update was done in that milisecond
                            #timeStart = miliseconds[argsBuy[LOB_share[a][0][b]]]
                            #pos_1 = pos1[lob1 == b]
                            #posTimes = miliseconds[pos_1]
                            #pos_1 = pos_1[posTimes > timeStart]
                            #if pos_1.shape[0] == 0:
                            #    cancelations[b].append(-1)
                            #else:
                            #    pos2 = np.min(cancelations)
                            #    cancelations[a].append(pos2)


                else:
                    for b in range(0, len(LOB_share[a][0])):
                        cancelations[a].append(-1)

            np.savez_compressed('./recursive/' + str(loopNumber) + '/choosenCancels/empty/' + oType + '_' + name + '.npz', cancelations)
        #quit()

    if loopNumber > 0:
        #saveComponent(0)
        #quit()
        multiProcessor(saveComponent, filesRun[0], filesRun[1], 4, doPrint = True)

#saveMajorEmptyCancelations(1, filesRun=(9, 15))
#quit()

def saveMinorCancelations():
    isBuy = False
    name = '20200117'

    #finalData = loadnpz('./inputData/ITCH_LOB/tempXY/X_Bid_' + name + '.npz')
    #print (finalData.shape)
    #namePart = 'Bid_20200117'

    #output = loadnpz('./resultData/outputPrediction/' + namePart + '.npz')
    #print (output.shape)

    if isBuy:
        #Yargs = loadnpz('./inputData/ITCH_LOB/tempXY/Y_Bid_' + name + '.npz')
        YData = loadnpz('./inputData/ITCH_LOB/allOrderXY/Y_Bid_' + name + '.npz')
        argsGood = loadnpz('./inputData/ITCH_LOB/tempXY/argsGood_Bid_' + name + '.npz')
        YData = YData[argsGood]

        finalData = loadnpz('./inputData/ITCH_LOB/tempXY/X_Bid_' + name + '.npz')
        LOB_share = loadnpz('./inputData/ITCH_LOB/BuyLOB/' + name + '.npz', allow_pickle=True)
        oType = "B"

        bestPrice = loadnpz('./inputData/ITCH_LOB/bestPrice/Bid_' + name + '.npz')
        shift = 10
        output = loadnpz('./resultData/outputPrediction/Bid_' + name + '.npz')
    else:
        #Yargs = loadnpz('./inputData/ITCH_LOB/tempXY/Y_Ask_' + name + '.npz')
        YData = loadnpz('./inputData/ITCH_LOB/allOrderXY/Y_Ask_' + name + '.npz')
        argsGood = loadnpz('./inputData/ITCH_LOB/tempXY/argsGood_Ask_' + name + '.npz')
        YData = YData[argsGood]

        finalData = loadnpz('./inputData/ITCH_LOB/tempXY/X_Ask_' + name + '.npz')
        LOB_share = loadnpz('./inputData/ITCH_LOB/SellLOB/' + name + '.npz', allow_pickle=True)
        oType = "S"

        bestPrice = loadnpz('./inputData/ITCH_LOB/bestPrice/Ask_' + name + '.npz')
        shift = -10
        output = loadnpz('./resultData/outputPrediction/Ask_' + name + '.npz')

    data = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz', allow_pickle=True)
    argsBuy = np.argwhere(data[:, 4] == oType)[:, 0]
    del data

    YData = YData[:output.shape[0]] #TODO REMOVE
    finalData = finalData[:output.shape[0]] #TODO REMOVE


    position = finalData[:, 0]
    price = YData[:, 1]
    quePos = YData[:, 2]

    priceCutOff = -0.3
    position = position[output < priceCutOff]
    price = price[output < priceCutOff]
    quePos = quePos[output < priceCutOff]
    #position = position[output > priceCutOff]
    #price = price[output > priceCutOff]

    cancelations = []
    for a in range(0, len(LOB_share)):
        print (a, len(LOB_share))

        cancelations.append([])
        #if a == 813:
        #print ("U")
        print (a)
        #print (posArgs.shape)
        #print (len(cancelations[a]))
        posArgs = np.argwhere( np.abs(a - bestPrice + shift) <= (np.abs(shift) + 1) )[:, 0].astype(int)

        changeArgs = argsBuy[np.array(LOB_share[a][0])]
        posArgs = posArgs[posArgs < np.max(changeArgs)]
        #print (posArgs.shape)
        inAr = stefan_nextAfter(posArgs, changeArgs) #TODO: check this for missing +1 or something.

        pos1 = position[price == a]
        que1 = quePos[price == a]
        posSort = np.argsort(pos1)
        pos1 = pos1[posSort]
        que1 = que1[posSort]

        ends1 = changeArgs[inAr]
        inAr1 = stefan_nextAfter(posArgs, pos1) #+1 makes strickly after
        inAr2 = stefan_firstBefore(ends1, pos1+1)

        #print (inAr1.shape)
        #print (np.max(inAr1))
        #print (np.max(inAr2))
        #print (np.mean(inAr2 - inAr1))

        time1 = time.time()

        queSplit = np.array([])
        queArgs = [0]
        M = 50
        for b in range(0, que1.shape[0] // M):
            queUnique = np.unique(que1[b*M:(b+1)*M])
            queSplit = np.concatenate((queSplit, queUnique))
            queArgs.append(queUnique.shape[0] + queArgs[-1])
            #queArgs.append(b)
        queArgs = np.array(queArgs)
        queSplit = queSplit.astype(int)

        #print (posArgs.shape)

        for b in range(0, posArgs.shape[0]):
            if inAr2[b] >= inAr1[b]:
                r1, r2 = (inAr1[b] // M) + 1, (inAr2[b] + 1) // M
                if r2 <= r1:
                    que2 = np.unique(que1[inAr1[b]:inAr2[b]+1])
                else:
                    #time1 = time.time()
                    que2_1 = np.unique(que1[inAr1[b]:r1*M])
                    #print (time.time() - time1)
                    que2_2 = np.unique(que1[r2*M:inAr2[b]+1])
                    #print (time.time() - time1)
                    que2_3 = np.unique(queSplit[queArgs[r1]:queArgs[r2]])
                    #print (time.time() - time1)
                    que2 = np.concatenate((np.concatenate((que2_1, que2_2)), que2_3))
                    #print (time.time() - time1)
                    que2 = np.unique(que2)

                    #if (np.mean(np.abs(  que2 -  np.unique(que1[inAr1[b]:inAr2[b]+1])  ))) != 0.0:
                    #    print (np.mean(np.abs(  que2 -  np.unique(que1[inAr1[b]:inAr2[b]+1])  )))
                    #    print (que2, np.unique(que1[inAr1[b]:inAr2[b]+1]) )
                    #    print ("Bad")
                    #    quit()
                    #quit()

                #queNums = np.unique(que1[inAr1[b]:inAr2[b]+1])
                cancelations[a].append(que2)
            else:
                cancelations[a].append([])



        print (time.time() - time1)
        #print (len(cancelations[a]))
        #quit()
        '''
        quit()

        for b in range(0, posArgs.shape[0]):
            if inAr2[b] >= inAr1[b]:
                queNums = np.unique(que1[inAr1[b]:inAr2[b]+1])
                cancelations[a].append(queNums)
            else:
                cancelationsFound.append([])
        '''

    if isBuy:
        np.savez_compressed('./resultData/cancelationsFound/minor_Bid_' + name + '.npz', cancelations)
    else:
        np.savez_compressed('./resultData/cancelationsFound/minor_Ask_' + name + '.npz', cancelations)
    quit()


def findingProfits():
    #print ("10, 10")
    names = giveFNames()
    #name = names[6]
    model = orderProfitModel()
    #model = torch.load('./Models/recLOB/model4_3.pt')
    optimizer = torch.optim.SGD(model.parameters(), lr = 2e-2)# * 0.01

    recursiveCancels = True

    #nameN = len(names)
    M = 19#00
    N = 10000
    #M = 10
    #N = 10000
    losses = []
    lossCor = []
    profitMeans = []
    for c in range(0, 5):
        print (c)
        for name in names:
            #c = 0
            for isBuy in [True, False]:
                if isBuy:
                    endingName1 = '_Bid_' + name
                else:
                    endingName1 = '_Ask_' + name
                endingName2 = endingName1 + '_' + str(c) + '.npz'

                noSell = ''#'noSell_'
                if recursiveCancels:
                    if isBuy:
                        profit = loadnpz('./resultData/Recur1_temporaryNeuralInput/'  + noSell + 'profit' + endingName2)
                    else:
                        profit = np.concatenate((profit, loadnpz('./resultData/Recur1_temporaryNeuralInput/'  + noSell + 'profit' + endingName2)))
                else:
                    if isBuy:
                        profit = loadnpz('./inputData/ITCH_LOB/temporaryNeuralInput/profit' + endingName2)
                    else:
                        profit = np.concatenate((profit, loadnpz('./inputData/ITCH_LOB/temporaryNeuralInput/profit' + endingName2)))

            #print (profit.shape)
            #print (np.mean(profit))
            profitMeans.append(np.mean(profit))
            #plt.hist(profit, bins=100)
            #plt.show()
            #quit()

    profitMeans = np.array(profitMeans)
    print (np.mean(profitMeans))
    plt.hist(profitMeans)
    plt.show()




    quit()


def funPCA():


    '''
    Ar = np.random.normal(size=(10, 10))
    Ar = np.matmul(Ar, Ar.T)

    S, V = np.linalg.eigh(Ar)
    S = np.diag(S)
    print (S, V)
    V = V#.T
    Ar2 = np.matmul(np.matmul(V, S), V.T)
    print (np.mean(np.abs(Ar - Ar2)))
    quit()
    '''

    #U, S, V = np.linalg.svd(Ar)
    #S = np.diag(S)
    #Ar2 = np.matmul(np.matmul(U, S), V)


    '''
    ar1 = np.array([[[1, 2], [2, 3]], [[1, 2], [2, 3]]])
    #ar1_shape = ar1.shape
    #ar1 = ar1.reshape(ar1_shape[0] * ar1_shape)
    ar2 = np.mean(ar1, axis=0)
    ar2_shape = ar2.shape
    ar2 = ar2.reshape((ar2.shape[0] * ar2.shape[1], ))
    print (ar2)
    ar2 = ar2.repeat(2).reshape(ar2.shape[0], 2).T
    ar2 = ar2.reshape((2, ar2_shape[0], ar2_shape[1]))
    print (ar2)
    quit()
    '''

    #.repeat(2)#.reshape(2, ar1.shape[1], ar1.shape[2])
    #print (ar2)
    #quit()

    N = 4#20#50 #100
    M = 50000
    X = np.random.normal(size=N * M)

    vol = np.cos(np.arange(X.shape[0] - 3) / (5 * M)) + 1.0


    mom = np.cos(np.arange(X.shape[0] - 3) / (5 * M)) #2



    #plt.plot(vol)
    ##plt.plot(mom)
    #plt.show()
    #quit()

    volMeasure = ((((X[2:-1] ** 2.0) + (X[1:-2] ** 2.0) + (X[:-3] ** 2.0)) / 3) ** 0.5)
    #X[3:] = X[3:] * 0.5 * (1.0 + vol)
    X[3:] = X[3:] * 1.0 * ( (1.0 - vol) + (vol * volMeasure))
    #X[3:] = X[3:] + (0.1 * mom * X[2:-1])

    #plt.hist(X)
    #plt.show()
    X[X>2] = 2
    X[X<-2] = -2
    X = X + 2
    X = X * 2
    L = 9
    X = np.round(X).astype(int)
    X2 = np.zeros((X.shape[0] - 3, 4, L))
    #X2 = np.zeros((X.shape[0] - 3, 2, L))
    count = np.arange(X.shape[0] - 3)
    X2[count, 0, X[:-3]] = 1
    X2[count, 1, X[1:-2]] = 1
    X2[count, 2, X[2:-1]] = 1
    X2[count, 3, X[3:]] = 1

    #X = X2[:, :3, :]
    #X = X.reshape((X2.shape[0], 3*X2.shape[2]))
    #Y = X2[:, 3, :]

    X2 = X2.reshape((X2.shape[0], 4*X2.shape[2]))


    matrix = np.matmul(X2.T, X2)
    print (matrix.shape)

    #print (Y.shape)
    #print (X.shape)




    quit()

    #X = X2[:, 0, :]
    #Y = X2[:, 1, :]

    from sklearn.neural_network import MLPClassifier
    matrixList = []
    for a in range(0, N):
        print (a)
        #regr = MLPClassifier(hidden_layer_sizes=(5,), activation='identity', max_iter=10000, learning_rate_init=1e-4)
        regr = MLPClassifier(hidden_layer_sizes=(5,), activation='identity', max_iter=200, learning_rate_init=1e-3)
        regr = regr.fit(X[a*M:(a+1)*M], Y[a*M:(a+1)*M])
        weights = regr.coefs_
        matrix = np.matmul(weights[0], weights[1])

        #matrixTotal1 = matrixTotal1 + np.mm(matrix, matrix.T)
        #matrixTotal2 = matrixTotal2 + np.mm(matrix.T, matrix)
        #matrixList1.append(np.mm(matrix, matrix.T))
        #matrixList2.append(np.mm(matrix, matrix.T))
        matrixList.append(matrix)

    matrixList = np.array(matrixList)
    for a in range(0, matrixList.shape[1]):
        for b in range(matrixList.shape[2]):
            matrixList[:, a, b] = matrixList[:, a, b] - np.mean(matrixList[:, a, b])

    matrixTotal1 = np.zeros((3*L, 3*L)) #3
    matrixTotal2 = np.zeros((L, L))
    for a in range(0, matrixList.shape[0]):
        matrixTotal1 = matrixTotal1 +  np.matmul(matrixList[a], matrixList[a].T)
        matrixTotal2 = matrixTotal2 +  np.matmul(matrixList[a].T, matrixList[a])

    matrixTotal1 = matrixTotal1 / N
    matrixTotal2 = matrixTotal2 / N

    S1, V1 = np.linalg.eigh(matrixTotal1)
    #print (S1)
    #S1 = np.diag(S1)

    S2, V2 = np.linalg.eigh(matrixTotal2)
    #S2 = np.diag(S2)

    S_1_full = []
    S_2_full = []
    S_3_full = []
    for a in range(0, matrixList.shape[0]):
        mat1 = np.matmul(matrixList[a], matrixList[a].T)
        S_1 = np.matmul(np.matmul(V1.T, mat1), V1)
        S_1 = S_1[np.arange(S_1.shape[0]), np.arange(S_1.shape[0])]
        S_1 = S_1[-5:]
        mat2 = np.matmul(matrixList[a].T, matrixList[a])
        S_2 = np.matmul(np.matmul(V2.T, mat2), V2)
        S_2 = S_2[np.arange(S_2.shape[0]), np.arange(S_2.shape[0])]
        S_2 = S_2[-5:]

        print (V1.shape)
        print (V2.shape)
        print (matrixList[a].shape)

        S_3 = np.matmul(np.matmul(V1.T, matrixList[a]), V2)
        S_3 = S_3[np.arange(5) - 6, np.arange(5) - 6]
        #plt.imshow(S_3)
        #plt.show()

        print (S_3.shape)

        S_1_full.append(S_1)
        S_2_full.append(S_2)
        S_2_full.append(S_3)

        #print (S_1)
        #plt.plot()
        #plt.imshow(S_1)
        #plt.show()
        #print (S_2)

    S_1_full = np.array(S_1_full)
    S_2_full = np.array(S_2_full)
    S_3_full = np.array(S_3_full)
    print (S_3_full.shape)
    plt.plot(S_3_full)
    plt.show()
    plt.plot(S_1_full)
    #plt.plot(S_3_full)
    plt.show()
    plt.plot(S_2_full)
    #plt.plot(S_3_full)
    plt.show()
    plt.plot(S_3_full)
    plt.show()

    quit()





    #matrixList1 = np.array(matrixList1)
    #matrixList2 = np.array(matrixList2)
    #matrixList1 = matrixList1 - np.repeat()



    #print (X2[0])
    #plt.hist(X, bins=100)
    #plt.show()
    quit()

#funPCA()
#quit()


def rollingPercentageCancel(loopNumber):
    names = giveFNames()
    cutOffs = []
    for a in range(0, 15):
        name = names[a]
        for oType in ['B', 'S']:
            print ("A")
            print (oType)
            #oType = 'B'
            #if oType == 'B':
            #    finalData = loadnpz('./inputData/Processed/inputXY/X_Bid_' + name + '.npz')
            #else:
            #    finalData = loadnpz('./inputData/Processed/inputXY/X_Ask_' + name + '.npz')
            finalData = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')

            X = finalData[:, 0].astype(int)
            del finalData
            output = loadnpz('./recursive/' + str(loopNumber) + '/outputPredictions/' + oType + '_' + name + '.npz')
            output[:, 1][output[:, 0] < 0.5] = -2.0
            output = output[:, 1]

            X_argsort = np.argsort(X)
            output = output[X_argsort]
            X = X[X_argsort]
            M = 1000000
            percentCancel = 0.01
            N = output.shape[0] // M
            output = output[:N*M]
            #print (output[:10])
            output = output.reshape((N, M))
            output = np.sort(output, axis=1)
            #print (output[0][:10])
            cutOff = output[:, int(M * percentCancel)]
            cutOffs.append(cutOff)

            for b in range(0, len(cutOffs)):
                plt.plot(cutOffs[b])
            plt.show()

#rollingPercentageCancel(1)
#quit()



def valsRemoveMean(args, vals):
    sizes = np.arange(vals.shape[0] + 1)
    valsSum = np.cumsum(vals)
    valsSum = np.concatenate((np.array([0]), valsSum))

    _, indices = np.unique(args, return_index=True)
    indices = np.concatenate((indices, np.array([args.shape[0]])))

    valsSum = valsSum[indices[1:]] - valsSum[indices[:-1]]
    sizes = sizes[indices[1:]] - sizes[indices[:-1]]
    means = valsSum / sizes

    vals = vals - means[args]

    return vals
    #print (vals)
    #quit()

def valsRemoveStd(args, vals):
    sizes = np.arange(vals.shape[0] + 1)
    valsSum = np.cumsum(vals ** 2.0)
    valsSum = np.concatenate((np.array([0]), valsSum))

    _, indices = np.unique(args, return_index=True)
    indices = np.concatenate((indices, np.array([args.shape[0]])))

    valsSum = valsSum[indices[1:]] - valsSum[indices[:-1]]
    sizes = sizes[indices[1:]] - sizes[indices[:-1]]
    means = valsSum / sizes

    means[means==0] = 1

    vals = vals / (means[args] ** 0.5)

    return vals

def valsCor(args, vals1, vals2):

    vals1 = valsRemoveMean(args, vals1)
    vals2 = valsRemoveMean(args, vals2)

    vals1 = valsRemoveStd(args, vals1)
    vals2 = valsRemoveStd(args, vals2)

    vals = vals1 * vals2

    sizes = np.arange(vals.shape[0] + 1)
    valsSum = np.cumsum(vals)
    valsSum = np.concatenate((np.array([0]), valsSum))

    _, indices = np.unique(args, return_index=True)
    indices = np.concatenate((indices, np.array([args.shape[0]])))

    valsSum = valsSum[indices[1:]] - valsSum[indices[:-1]]
    sizes = sizes[indices[1:]] - sizes[indices[:-1]]
    means = valsSum / np.mean(sizes)

    #print (means)
    #quit()

    print (np.mean(means))

    plt.hist(means, bins=100)
    plt.show()


    #return vals

#vals1 = np.array([1, 2, 3, 1, 2, 3, 1, 2, 2])
#vals2 = np.array([1, 2, 3, 3, 2, 1, 1, 1, 3])
#args =  np.array([0, 0, 0, 1, 1, 1, 2, 2, 3])
#valsCor(args, vals1, vals2)
#uit()






def sellerAnalysis3():
    print ("K")

    #'''
    cut1 = 0
    cut2 = 0
    sum1 = 0
    sum2 = 0
    sum3 = 0
    #loopNumber = 1
    c = 1

    N = 0
    S = 0.0
    #valsFull = [[], []]

    valsFull = []


    names = giveFNames()

    #11, 12, 14
    #4, 5, 6
    #quit()
    for a in range(0, 15): #13
        name = names[a]
        print ("")
        print (name)
        oType =  "B"
        #oType =  "S"
        if oType == 'B':
            bid = 'Bid'
        else:
            bid = 'Ask'





        #finalData = loadnpz('./recursive/0/profitData/X_' + oType + '_' + name + '.npz')
        #min1, max1 = np.min(finalData[:, 0]), np.max(finalData[:, 0])
        '''
        for a in [16]:#range(16, 17):

            finalData = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')
            profitTrans = loadnpz('./recursive/0/orderData/withCancels/Trans_' + oType + '_' + name + '.npz')

            bestPrice0 = loadnpz('./inputData/ITCH_LOB/bestPrice/Bid_' + name + '.npz')
            plt.plot(bestPrice0[80000:85000])
            plt.show()
            #quit()
            #plt.hist(finalData[:, 0][finalData[:, 1] == a], bins=100, range=(min1, max1), histtype='step')
            plt.hist(profitTrans[finalData[:, 1] == a], bins=1000, histtype='step', range=(80000, 87000))
            plt.yscale('log')
            plt.show()

        quit()
        #'''

        '''
        profitSum = [0, 0]
        Nsum = [0, 0]

        for name in names:
            for oType in ['B', 'S']:
                finalData = loadnpz('./recursive/0/profitData/X_' + oType + '_' + name + '.npz')
                profit = loadnpz('./recursive/0/profitData/Trans_' + oType + '_' + name + '.npz')
                profitSum[0] += np.sum(profit[finalData[:, 1] == 1])
                profitSum[1] += np.sum(profit[finalData[:, 1] == 5])
                Nsum[0] += profit[finalData[:, 1] == 1].shape[0]
                Nsum[1] += profit[finalData[:, 1] == 5].shape[0]
            print (np.array(profitSum) / np.array(Nsum))
        quit()


        finalData = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')
        print (finalData.shape)
        finalData = loadnpz('./recursive/1/orderData/withCancels/X_' + oType + '_' + name + '.npz')
        print (finalData.shape)
        quit()
        '''

        '''
        for name in names:
            for oType in ['B', 'S']:
                output = loadnpz('./recursive/0/samples/prob_' + oType + '_' + name + '.npz')
                print (output[output[:, 2] > 0.2].shape[0] // 1000)

                finalData = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')
                #plt.hist(output[:, 2], bins=100, range=(-2, 2))
                #plt.show()
                #print(np.max(output[:, 0]))
                #print (finalData.shape)
                finalData = finalData[output[:, 0].astype(int)]

                #plt.hist(finalData[output[:, 2] > 0.2, 1], bins=100)
                #plt.show()
                finalData = finalData[output[:, 2] > 0.2]
                #finalData = finalData[finalData[:, 1] == 0]
                X = finalData[:, 0]

                plt.hist(X, bins=100, range=(0, np.max(finalData[:, 0])))
                plt.show()
                #quit()

        quit()
        #'''

        '''
        for name in names[:2]:
            allShares = loadnpz('./inputData/ITCH_LOB/NearBestOrders/' + name + '.npz') / 200.0 #[:, :, :, 1] / 200.0
            midpoint = allShares.shape[1] // 2
            allShares = allShares[:, midpoint-20:midpoint+20]
            allShares = np.sum(allShares, axis=2)

            allSharesSum = np.sum(allShares, axis=1)
            allSharesSum = np.cumsum(allSharesSum)
            N = 1000
            allShares = allShares[N:]
            allSharesSum = (allSharesSum[N:] - allSharesSum[:-N]) / N

            for b in range(0, allShares.shape[1]):
                allShares[:, b] = allShares[:, b] / allSharesSum

            #allShares1 = allShares[:, 19]
            #allShares2 = allShares[:, 22]
            allShares1 = np.sum(allShares[:, :20], axis=1)
            allShares2 = np.sum(allShares[:, 20:], axis=1)

            plt.plot(allShares1)
            plt.plot(allShares2)
            plt.show()
        quit()
        '''


        '''
        X = loadnpz('./recursive/0/profitData/X_' + oType + '_' + name + '.npz')[:, 0]
        profit = loadnpz('./recursive/0/profitData/Trans_' + oType + '_' + name + '.npz')
        profit = profit[np.argsort(X)]

        L = 10000
        profit1 = (profit[L:] - profit[:-L]) / L
        profit[:-L] = profit[:-L] - profit1
        profit[-L:] = profit[-L:] - profit1[-1]

        profit = np.abs(profit)
        profitSum = np.cumsum(profit)
        N = 1000
        profitSum = profitSum[0::N]
        profitSum = profitSum[1:] - profitSum[:-1]
        plt.plot(profitSum)
        plt.show()
        quit()
        '''




        '''
        output = loadnpz('./recursive/0/outputPredictions/' + oType + '_' + name + '.npz')[:, 1]
        finalData = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')
        YData = loadnpz('./recursive/0/orderData/withCancels/Y_' + oType + '_' + name + '.npz')

        #print (scipy.stats.pearsonr(finalData[:, 2], output))
        #quit()

        allShares = loadnpz('./inputData/ITCH_LOB/NearBestOrders/' + name + '.npz') / 200.0 #[:, :, :, 1] / 200.0
        midpoint = allShares.shape[1] // 2
        allShares = allShares[:, midpoint-20:midpoint+20]
        allShares = np.sum(allShares, axis=2)

        #finalData = finalData[output > -0.2]
        #YData = YData[output > -0.2]
        #output = output[output > -0.2]

        #argsAbove = np.arange(output.shape[0])#np.argwhere(output > -0.2)[:, 0]
        argsAbove = np.argwhere(finalData[:, 1] == 0)[:, 0]
        argRandom = np.random.choice(argsAbove.shape[0], 1000)
        argRandom = argsAbove[argRandom]
        times1 = finalData[argRandom, 0]

        count = 0
        for a in range(0, 1000):
            time1 = times1[a]
            price1 = YData[argRandom[a], 1]

            #argsSameTime = np.argwhere(finalData[:, 0] == time1)[:, 0]
            argsSameTime = np.argwhere(np.logical_and(finalData[:, 0] == time1, YData[:, 1] == price1))[:, 0]

            if argsSameTime.shape[0] > 1:
                #print (finalData[argsSameTime, 0])
                #print (finalData[argsSameTime, 1])
                #print (finalData[argsSameTime, 2].shape)
                #quit()
                plt.plot(finalData[argsSameTime, 2], output[argsSameTime])
                count += 1
            if count == 30:
                plt.show()


            if argsSameTime.shape[0] > 1:

                quePos = YData[argsSameTime, 2]
                depth = finalData[argsSameTime, 2]
                relPrice = finalData[argsSameTime, 1]


                #plt.hist(output[argsSameTime][relPrice == 0], bins=100)
                #plt.scatter(depth[relPrice == 0], output[argsSameTime][relPrice == 0])
                #plt.scatter(relPrice, depth, c=output[argsSameTime])

                shares1 = allShares[time1]

                relPriceU = np.unique(relPrice)
                for b in range(0, relPriceU.shape[0]):
                    price1 = relPriceU[b]
                    depth[relPrice == price1] = np.max(depth[relPrice == price1]) - depth[relPrice == price1]



                #plt.plot(shares1)
                #plt.show()
                plt.scatter(relPrice, depth, c=output[argsSameTime])
                #plt.scatter(relPrice, quePos, c=output[argsSameTime])
                plt.show()


        quit()
        #'''










        '''
        choice1 = 2
        #output1 = loadnpz('./recursive/' + str(0) + '/samplesMix/prob_' + str(choice1) + '.npz')[:, 1]
        #plt.hist(output1, bins=100, range=(-2.0, 2.0))
        #plt.show()
        #quit()
        order = loadnpz('./recursive/0/neuralInputs/2/order_g0_' + str(choice1) + '.npz')
        #orderExtra = torch.load('./recursive/0/neuralInputs/2/orderExtra_g0_' + str(choice1) + '.pt')
        #orderExtra = orderExtra.data.numpy()
        #orderExtra = orderExtra[:, 1, 10:12]
        #depth2 = orderExtra[:, 1] - orderExtra[:, 0]
        #quit()
        spread = loadnpz('./recursive/0/neuralInputs/2/spread_g0_' + str(choice1) + '.npz') * 2
        #allShares = loadnpz('./recursive/0/neuralInputs/2/shares_' + str(choice1) + '.npz') / 20
        profit = loadnpz('./recursive/' + str(0) + '/neuralInputs/2/profit_add_g0_' + str(choice1) + '.npz')
        #output1 = output1[profit[:, 1] == 0]
        order = order[profit[:, 1] == 0]
        #allShares = allShares[profit[:, 1] == 0]
        spread = spread[profit[:, 1] == 0]
        #depth2 = depth2[profit[:, 1] == 0]
        profit = profit[profit[:, 1] == 0][:, 0]
        depth = np.sum(order[:, 1, :], axis=1)
        order = order[:, 0, :]
        order = order[:, np.array([16, 17, 20, 21])]
        #print (order[:20])
        #quit()

        #cutOff0 = -0.5
        #cutOff = -0.4
        cutOff0 = -0.5
        cutOff = -0.2

        for orderNum in range(0, 4):
            print (scipy.stats.pearsonr(depth[order[:, orderNum] == 1], profit[order[:, orderNum] == 1]))
            #print (scipy.stats.pearsonr(depth2[order[:, orderNum] == 1], profit[order[:, orderNum] == 1]))
        quit()



        #plt.scatter(depth[spread==1][0::1000], output1[spread==1][0::1000])
        #plt.scatter(depth[spread==2][0::1000], output1[spread==2][0::1000])
        for orderNum in range(0, 2):
            for spreadNum in range(0, 2):
                args = np.argwhere(np.logical_and(spread == (spreadNum+1), order[:, orderNum] == 1))[:, 0]
                plt.scatter(depth[args][0::1000], output1[args][0::1000])

        plt.show()
        #print (scipy.stats.pearsonr(depth, profit))
        #print (scipy.stats.pearsonr(depth, output1))
        quit()

        #print (np.mean(spread[output1 < cutOff0]))
        print (np.mean(spread))
        print (np.mean(spread[output1 > cutOff]))

        #print (np.mean(depth[output1 < cutOff0]))
        print (np.mean(depth))
        print (np.mean(depth[output1 > cutOff]))

        print (np.mean(output1[order[:, 0, 22] == 1]))
        print (np.mean(output1[order[:, 0, 23] == 1]))

        #print (np.argsort(np.mean(order[:, 0, :][output1 > cutOff], axis=0)))
        #quit()

        #plt.plot(np.mean(order[:, 0, :][output1 < cutOff0], axis=0))
        plt.plot(np.mean(order[:, 0, :], axis=0))
        plt.plot(np.mean(order[:, 0, :][output1 > cutOff], axis=0))
        #plt.plot(np.mean(order[:, 0, :][np.abs(output1 + 0.05) < 0.05], axis=0))
        plt.show()
        #plt.hist(output1[diff < -0.05], bins=200, histtype='step', range=(-1.0, 0.2))
        #plt.hist(output2[diff < -0.05], bins=200, histtype='step', range=(-1.0, 0.2))
        #plt.show()

        #plt.plot(np.mean(np.sum(allShares[output1 < cutOff0], axis=2), axis=0))
        plt.plot(np.mean(np.sum(allShares, axis=2), axis=0))
        #plt.plot(np.mean(np.sum(allShares[output < -1.2], axis=2), axis=0))
        plt.plot(np.mean(np.sum(allShares[output1 > cutOff], axis=2), axis=0))
        #plt.plot(np.mean(np.sum(allShares[np.abs(output1 + 0.05) < 0.05], axis=2), axis=0))
        plt.show()

        plt.plot(np.sum(allShares, axis=2)[:10].T, c='b')
        #plt.plot(np.mean(np.sum(allShares[output < -1.2], axis=2), axis=0))
        plt.plot(np.sum(allShares[output1 > cutOff], axis=2)[:10].T, c='r')
        #plt.plot(np.mean(np.sum(allShares[np.abs(output1 + 0.05) < 0.05], axis=2), axis=0))
        plt.show()
        quit()

        #plt.plot(np.mean(np.sum(allShares[output1 < cutOff0], axis=1), axis=0))
        plt.plot(np.mean(np.sum(allShares, axis=1), axis=0))
        #plt.plot(np.mean(np.sum(allShares[output < -1.2], axis=1), axis=0))
        plt.plot(np.mean(np.sum(allShares[output1 > cutOff], axis=1), axis=0))
        plt.show()

        #plt.plot(np.mean(order, axis=0))
        #plt.plot(np.mean(order[output < -1.2], axis=0))
        #plt.plot(np.mean(order[output > 0], axis=0))
        #plt.show()
        quit()
        #'''














        #'''
        choice1 = 0
        #output1 = loadnpz('./recursive/' + str(0) + '/samplesMix/prob_' + str(choice1) + '.npz')[:, 1]
        #plt.hist(output1, bins=100, range=(-0.65, -0.15))
        #plt.show()
        #quit()



        #choice1 = 1
        for choice1 in range(0, 3):
            N = 1#000
            profit = loadnpz('./recursive/' + str(1) + '/neuralInputs/2/profit_g0_' + str(choice1) + '.npz')[0::N]
            #profitShift =loadnpz('./recursive/' + str(0) + '/neuralInputs/2/profit_g0_' + str(choice1) + '.npz')
            #profitShift = profitShift[profitShift[:, 1] == 0][:, 0]
            #profitShift = np.mean(profitShift)
            #print (profitShift)
            #quit()
            order = loadnpz('./recursive/0/neuralInputs/2/order_g0_' + str(choice1) + '.npz')[0::N]
            #spread = loadnpz('./recursive/0/neuralInputs/2/spread_' + str(choice1) + '.npz')[0::N] #* 2
            #allShares = loadnpz('./recursive/0/neuralInputs/2/shares_' + str(choice1) + '.npz')[0::N] #/ 20
            #print (spread.shape)
            #output = output[profit[:, 1] == 0]
            order = order[profit[:, 1] == 0]
            #allShares = allShares[profit[:, 1] == 0]
            #spread = spread[profit[:, 1] == 0]
            profit = profit[profit[:, 1] == 0][:, 0]
            depth = np.sum(order[:, 1, :], axis=1)

            print (profit.shape)
            #profit = profit + profitShift

            order = order[:, 0, :]


            #'''
            #argsGood = np.argwhere(np.sum(order[:, 19-2:19+2], axis=1) == 1)[:, 0]
            #argsGood = np.argwhere(np.sum(order[:, np.array([19-2, 19+1])], axis=1) == 1)[:, 0]
            for c in range(0, 5):
                argsGood = np.argwhere(np.sum(order[:, np.array([19-2-c, 19+1+c])], axis=1) == 1)[:, 0]

                #plt.hist(depth[argsGood], bins=100, range=(0, 30))
                #plt.show()
                #quit()

                profs1 = []
                for a in range(0, 10):
                    argsSub = np.argwhere(np.abs(depth[argsGood] - a - 0.5) <= 0.5)[:, 0]
                    profitMean = np.mean(profit[argsGood[argsSub]])
                    profs1.append(profitMean)

                plt.plot(profs1)
            plt.ylim(-0.5, -0.15)
            plt.show()

            #plt.hist(depth[argsGood], bins=100, range=(0, 30))
            #plt.show()
            quit()

            #'''

            #print (np.mean(profit ** 2.0) ** 0.5)
            print (scipy.stats.pearsonr(depth, profit))
            #print (scipy.stats.pearsonr(depth[order[:, 16] == 1], profit[order[:, 16] == 1]))
            #print (scipy.stats.pearsonr(depth[order[:, 17] == 1], profit[order[:, 17] == 1]))


            #print (np.sum(profit[np.abs(profit) > 10]**2.0))
            #print (np.sum(profit[np.abs(profit) < 10]**2.0))

            #plt.hist(profit, bins=100, range=(-20, 20))
            #plt.show()
            #quit()

            order = order[depth == 0]
            profit = profit[depth == 0]

            order1 = order#[:, 0, :]
            profs1 = []
            yerrors = []
            for a in range(0, order1.shape[1]):
                args = np.argwhere(order1[:, a] == 1)[:, 0]
                if args.shape[0] != 0:
                    #print (np.mean(profit[args]))
                    print (args.shape[0])
                    profs1.append(np.mean(profit[args]))
                    yerror = (np.mean(profit[args] ** 2.0) / args.shape[0]) ** 0.5
                    yerrors.append(yerror)
            plt.plot(profs1)
            #plt.plot(yerrors)
            plt.show()
        quit()



        #print (np.mean(output[output < -2.5]))
        #print (np.mean(profit[output < -2.5]))
        #quit()

        model = torch.load('./recursive/0/Models/predictBoth/1.pt')
        #model = torch.load('./temporary/fakeModel.pt')
        #model1 = torch.load('./recursive/0/Models/predictBoth/2_3.pt')
        #model2 = torch.load('./recursive/0/Models/predictBothEst/2_2.pt')
        midpoint = order.shape[2] // 2
        args = np.argwhere(order[:, 0, midpoint-2] == 1)[:, 0][1]
        #print (np.mean(order[args, 1, midpoint-2]))
        #quit()
        args = (np.zeros(100) + args).astype(int)
        allShares1 = torch.tensor(allShares[args]).float()
        order1 = order[args]
        #print (to)
        for b in range(0, 100):
            order1[b, 1, midpoint-2] = (b / 10)
        order1 = torch.tensor(order1).float()
        spread1 = torch.tensor(spread[args]).float()

        output1 =  model(allShares1, order1, spread1)[:, 1]


        plt.plot(output1.data.numpy())
        plt.show()
        #output1 = model(allShares1, order1, spread1)[:, 1]
        #output2 = model(allShares1 + 10, order1, spread1)[:, 1]

        #print (torch.mean(torch.abs(output1 - output2)))



        quit()
        choice1 = 0
        N = 1#00
        output = loadnpz('./recursive/' + str(0) + '/samplesMix/prob_' + str(choice1) + '.npz')[:, 1]

        plt.hist(output, bins=500, range=(-0.65, -0.3))
        plt.show()

        quit()
        #'''

        #'''
        choice1 = 0
        output1 = loadnpz('./recursive/' + str(0) + '/samplesMix/prob_' + str(choice1) + '.npz')[:, 1]
        #plt.hist(output1, bins=100)
        #plt.show()
        #quit()
        order = loadnpz('./recursive/0/neuralInputs/2/order_' + str(choice1) + '.npz')
        spread = loadnpz('./recursive/0/neuralInputs/2/spread_' + str(choice1) + '.npz') * 2
        allShares = loadnpz('./recursive/0/neuralInputs/2/shares_' + str(choice1) + '.npz') / 20
        profit = loadnpz('./recursive/' + str(0) + '/neuralInputs/2/profit_' + str(choice1) + '.npz')
        output1 = output1[profit[:, 1] == 0]
        order = order[profit[:, 1] == 0]
        allShares = allShares[profit[:, 1] == 0]
        spread = spread[profit[:, 1] == 0]
        profit = profit[profit[:, 1] == 0][:, 0]
        depth = np.sum(order[:, 1, :], axis=1)

        #cutOff0 = -0.5
        #cutOff = -0.4
        cutOff0 = -0.48
        cutOff = -0.42

        print (np.mean(spread[output1 < cutOff0]))
        print (np.mean(spread))
        print (np.mean(spread[output1 > cutOff]))

        print (np.mean(depth[output1 < cutOff0]))
        print (np.mean(depth))
        print (np.mean(depth[output1 > cutOff]))


        #plt.plot(np.mean(order[:, 0, :][output1 < cutOff0], axis=0))
        plt.plot(np.mean(order[:, 0, :], axis=0))
        plt.plot(np.mean(order[:, 0, :][output1 > cutOff], axis=0))
        #plt.plot(np.mean(order[:, 0, :][np.abs(output1 + 0.05) < 0.05], axis=0))
        plt.show()
        #plt.hist(output1[diff < -0.05], bins=200, histtype='step', range=(-1.0, 0.2))
        #plt.hist(output2[diff < -0.05], bins=200, histtype='step', range=(-1.0, 0.2))
        #plt.show()

        #plt.plot(np.mean(np.sum(allShares[output1 < cutOff0], axis=2), axis=0))
        plt.plot(np.mean(np.sum(allShares, axis=2), axis=0))
        #plt.plot(np.mean(np.sum(allShares[output < -1.2], axis=2), axis=0))
        plt.plot(np.mean(np.sum(allShares[output1 > cutOff], axis=2), axis=0))
        #plt.plot(np.mean(np.sum(allShares[np.abs(output1 + 0.05) < 0.05], axis=2), axis=0))
        plt.show()

        #plt.plot(np.mean(np.sum(allShares[output1 < cutOff0], axis=1), axis=0))
        plt.plot(np.mean(np.sum(allShares, axis=1), axis=0))
        #plt.plot(np.mean(np.sum(allShares[output < -1.2], axis=1), axis=0))
        plt.plot(np.mean(np.sum(allShares[output1 > cutOff], axis=1), axis=0))
        plt.show()

        #plt.plot(np.mean(order, axis=0))
        #plt.plot(np.mean(order[output < -1.2], axis=0))
        #plt.plot(np.mean(order[output > 0], axis=0))
        #plt.show()







        quit()
        #choice1 = 0
        #N = 1
        #output = loadnpz('./recursive/' + str(0) + '/samplesMix/prob_' + str(choice1) + '.npz')[:, 1]#[0::N]
        #plt.hist(output, bins=500, histtype='step')
        #plt.hist(output, bins=500, histtype='step', range=(-0.55, -0.3))
        #plt.show()
        #'''





        #quit()

        '''
        choice1 = 0
        N = 1000
        profit = loadnpz('./recursive/' + str(0) + '/neuralInputs/2/profit_' + str(choice1) + '.npz')[0::N]
        order = loadnpz('./recursive/0/neuralInputs/2/order_' + str(choice1) + '.npz')[0::N]
        spread = loadnpz('./recursive/0/neuralInputs/2/spread_' + str(choice1) + '.npz')[0::N] #* 2
        allShares = loadnpz('./recursive/0/neuralInputs/2/shares_' + str(choice1) + '.npz')[0::N] #/ 20
        print (spread.shape)
        #output = output[profit[:, 1] == 0]
        order = order[profit[:, 1] == 0]
        allShares = allShares[profit[:, 1] == 0]
        spread = spread[profit[:, 1] == 0]
        profit = profit[profit[:, 1] == 0][:, 0]
        depth = np.sum(order[:, 1, :], axis=1)



        #print (np.mean(output[output < -2.5]))
        #print (np.mean(profit[output < -2.5]))
        #quit()

        #model = torch.load('./recursive/0/Models/predictBoth/2_3.pt')
        model = torch.load('./temporary/fakeModel.pt')
        '''

        '''
        #model1 = torch.load('./recursive/0/Models/predictBoth/2_3.pt')
        #model2 = torch.load('./recursive/0/Models/predictBothEst/2_2.pt')
        allShares1 = torch.tensor(allShares).float()
        order1 = torch.tensor(order).float()
        spread1 = torch.tensor(spread).float()
        output1 = model(allShares1, order1, spread1)[:, 1]
        output2 = model(allShares1 + 10, order1, spread1)[:, 1]

        print (torch.mean(torch.abs(output1 - output2)))

        #print (torch.mean(output1))
        #print (torch.mean(output2))

        quit()



        output1 = model1(allShares1, order1, spread1)[:, 1]
        output2 = model2(allShares1, order1, spread1)[:, 1]

        output1_np = output1.data.numpy()
        output2_np = output2.data.numpy()

        diff = np.abs(output2_np - output1_np)

        argsAbove = np.argwhere(diff > 0.2)[:, 0]

        print (np.mean(spread))
        print (np.mean(spread[argsAbove]))

        print (np.mean(depth))
        print (np.mean(depth[argsAbove]))


        plt.plot(np.sum(allShares, axis=1)[:10].T, c='b')
        plt.plot(np.sum(allShares[argsAbove], axis=1)[:10].T, c='r')
        plt.show()

        plt.plot(np.sum(allShares, axis=2)[:10].T, c='b')
        plt.plot(np.sum(allShares[argsAbove], axis=2)[:10].T, c='r')
        plt.show()

        #plt.plot(order[:, 0, :][:10].T, c='b')
        #plt.plot(order[:, 0, :][argsAbove][:10].T, c='r')
        #plt.show()

        quit()
        #'''


        plt.plot(np.mean(np.sum(allShares, axis=1), axis=0))
        plt.plot(np.mean(np.sum(allShares[argsAbove], axis=1), axis=0))
        plt.show()

        plt.plot(np.mean(np.sum(allShares, axis=2), axis=0))
        plt.plot(np.mean(np.sum(allShares[argsAbove], axis=2), axis=0))
        plt.show()

        plt.plot(np.mean(order[:, 0, :], axis=0))
        plt.plot(np.mean(order[:, 0, :][argsAbove], axis=0))
        plt.show()



        #plt.hist(diff, bins=100, range=(0, 1))
        #plt.show()
        #quit()

        print (scipy.stats.pearsonr(output1_np, output2_np))

        #plt.hist(output2, bins=500, range=(-1.5, 0.5))
        #plt.show()





        quit()

        #'''
        choice1 = 0
        N = 1#00
        output = loadnpz('./recursive/' + str(0) + '/samplesMix/prob_' + str(choice1) + '.npz')[:, 1][0::N]
        output2 = loadnpz('./recursive/' + str(1) + '/samplesMix/prob_' + str(choice1) + '.npz')[:, 1][0::N]
        profit = loadnpz('./recursive/' + str(0) + '/neuralInputs/2/profit_' + str(choice1) + '.npz')[0::N]
        profit2 = loadnpz('./recursive/' + str(1) + '/neuralInputs/2/profit_' + str(choice1) + '.npz')[0::N]

        output = output[profit2[:, 1] == 0]
        profit = profit[profit2[:, 1] == 0][:, 0]
        output2 = output2[profit2[:, 1] == 0]
        profit2 = profit2[profit2[:, 1] == 0][:, 0]

        args = np.argwhere(np.abs(output2 + 0.8) < 0.05)[:, 0]

        print (np.mean(profit2[args]))
        print (np.mean(profit[args]))
        quit()

        plt.hist(output , bins=200, range=(-1.5, 0.5), histtype='step')
        plt.hist(output2, bins=200, range=(-1.5, 0.5), histtype='step')
        plt.show()
        quit()


        print (np.mean(profit2[output2 < -0.5]))
        print (np.mean(output2[output2 < -0.5]))

        print (np.mean(profit2[np.logical_and(output2 < -0.5, output > 0.5)]))
        print (np.mean(output2[np.logical_and(output2 < -0.5, output > 0.5)]))

        quit()
        print (np.mean(profit2[outp]))

        print (scipy.stats.pearsonr(output, output2))
        quit()
        #plt.hist(output , bins=200, range=(-1.5, 0.5), histtype='step')
        #plt.hist(output2, bins=200, range=(-1.5, 0.5), histtype='step')
        plt.hist(output , bins=50, range=(-30.0, -1.0), histtype='step')
        plt.hist(output2, bins=50, range=(-30.0, -1.0), histtype='step')
        plt.show()
        quit()

        profit = loadnpz('./recursive/' + str(1) + '/neuralInputs/2/profit_' + str(choice1) + '.npz')[0::N]
        order = loadnpz('./recursive/0/neuralInputs/2/order_' + str(choice1) + '.npz')[0::N]
        spread = loadnpz('./recursive/0/neuralInputs/2/spread_' + str(choice1) + '.npz')[0::N] #* 2
        allShares = loadnpz('./recursive/0/neuralInputs/2/shares_' + str(choice1) + '.npz')[0::N] #/ 20
        print (spread.shape)
        output = output[profit[:, 1] == 0]
        order = order[profit[:, 1] == 0]
        allShares = allShares[profit[:, 1] == 0]
        spread = spread[profit[:, 1] == 0]
        profit = profit[profit[:, 1] == 0][:, 0]
        depth = np.sum(order[:, 1, :], axis=1)

        print (np.mean(output[output < -2.5]))
        print (np.mean(profit[output < -2.5]))
        quit()


        model = torch.load('./recursive/0/Models/predictBoth/3.pt')
        allShares1 = torch.tensor(allShares).float()
        order1 = torch.tensor(order).float()
        spread1 = torch.tensor(spread).float()
        output1 = model(allShares1, order1, spread1)[:, 1]

        output2 = output1.data.numpy()

        plt.hist(output2, bins=500, range=(-1.5, 0.5))
        plt.show()



        quit()
        #'''




        '''
        choice1 = 0
        output1 = loadnpz('./recursive/' + str(0) + '/samplesMix/prob_' + str(choice1) + '.npz')[:, 1]
        output2 = loadnpz('./recursive/' + str(1) + '/samplesMix/prob_' + str(choice1) + '.npz')[:, 1]

        print (np.mean(output1))
        print (np.mean(output2))
        #quit()
        plt.hist(output1, bins=500, range=(-1.5, 0.5), histtype='step')
        plt.hist(output2, bins=500, range=(-1.5, 0.5), histtype='step')
        plt.show()
        plt.hist(output1, bins=500, range=(-2.5, -1.2), histtype='step')
        plt.hist(output2, bins=500, range=(-2.5, -1.2), histtype='step')
        plt.show()
        quit()
        profit = loadnpz('./recursive/' + str(1) + '/neuralInputs/2/profit_' + str(choice1) + '.npz')

        output1 = output1[profit[:, 1] == 0]
        profit = profit[profit[:, 1] == 0][:, 0]

        preds = []
        reals = []
        X = []
        stepSize = 0.01
        for a in range(0, 110):
            val = a * -1 * stepSize
            val = val + 0.2
            args = np.argwhere(np.abs(output1 - val)  < (stepSize / 2))[:, 0]
            preds.append(np.mean(output1[args]))
            reals.append(np.mean(profit[args]))
            X.append(val)

        plt.plot(X, preds)
        plt.plot(X, reals)
        #plt.plot(X, np.array(reals) - np.array(preds))
        plt.show()

        quit()
        '''



        choice1 = 0
        output1 = loadnpz('./recursive/' + str(0) + '/samplesMix/prob_' + str(choice1) + '.npz')[:, 1]
        order = loadnpz('./recursive/0/neuralInputs/2/order_' + str(choice1) + '.npz')
        spread = loadnpz('./recursive/0/neuralInputs/2/spread_' + str(choice1) + '.npz') * 2
        allShares = loadnpz('./recursive/0/neuralInputs/2/shares_' + str(choice1) + '.npz') / 20
        profit = loadnpz('./recursive/' + str(0) + '/neuralInputs/2/profit_' + str(choice1) + '.npz')
        output1 = output1[profit[:, 1] == 0]
        order = order[profit[:, 1] == 0]
        allShares = allShares[profit[:, 1] == 0]
        spread = spread[profit[:, 1] == 0]
        profit = profit[profit[:, 1] == 0][:, 0]
        depth = np.sum(order[:, 1, :], axis=1)

        print (np.mean(spread))
        print (np.mean(spread[output1 > 0.0]))

        print (np.mean(depth))
        print (np.mean(depth[output1 > 0.0]))




        plt.plot(np.mean(order[:, 0, :], axis=0))
        plt.plot(np.mean(order[:, 0, :][output1 > 0.0], axis=0))
        plt.plot(np.mean(order[:, 0, :][np.abs(output1 + 0.05) < 0.05], axis=0))
        plt.show()
        #plt.hist(output1[diff < -0.05], bins=200, histtype='step', range=(-1.0, 0.2))
        #plt.hist(output2[diff < -0.05], bins=200, histtype='step', range=(-1.0, 0.2))
        #plt.show()

        plt.plot(np.mean(np.sum(allShares, axis=2), axis=0))
        #plt.plot(np.mean(np.sum(allShares[output < -1.2], axis=2), axis=0))
        plt.plot(np.mean(np.sum(allShares[output1 > 0.0], axis=2), axis=0))
        plt.plot(np.mean(np.sum(allShares[np.abs(output1 + 0.05) < 0.05], axis=2), axis=0))
        plt.show()

        plt.plot(np.mean(np.sum(allShares, axis=1), axis=0))
        #plt.plot(np.mean(np.sum(allShares[output < -1.2], axis=1), axis=0))
        plt.plot(np.mean(np.sum(allShares[output1 > 0.0], axis=1), axis=0))
        plt.show()

        #plt.plot(np.mean(order, axis=0))
        #plt.plot(np.mean(order[output < -1.2], axis=0))
        #plt.plot(np.mean(order[output > 0], axis=0))
        #plt.show()


        quit()

        #print (np.mean(profit[output1 > 0.02]))
        #plt.hist(output1, bins=500, histtype='step', range=(-1.4, 0.4))
        #plt.show()
        #quit()




        quit()

        output2 = loadnpz('./recursive/' + str(0) + '/samplesMix2/prob_' + str(choice1) + '.npz')[:, 1]
        output3 = loadnpz('./recursive/' + str(0) + '/samplesMix3/prob_' + str(choice1) + '.npz')[:, 1]

        profit = loadnpz('./recursive/' + str(1) + '/neuralInputs/2/profit_' + str(choice1) + '.npz')[:, 0]
        #print (np.mean(profit[output1 > 0]))
        #quit()

        print (scipy.stats.pearsonr(output1, output2))
        print (scipy.stats.pearsonr(output1, output3))
        print (scipy.stats.pearsonr(output2, output3))
        plt.hist(output1, bins=200, histtype='step', range=(-1.4, 0.4))
        plt.hist(output2, bins=200, histtype='step', range=(-1.4, 0.4))
        plt.hist(output3, bins=200, histtype='step', range=(-1.4, 0.4))
        plt.show()
        quit()


        '''
        data = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz')

        miliseconds = data[:, 0].astype(int)
        _, miliArgs = np.unique(miliseconds[-1::-1], return_index=True)
        miliArgs = data.shape[0] - 1 - miliArgs
        reverse = np.zeros(data.shape[0])
        reverse[miliArgs] = 1
        reverse = np.cumsum(reverse).astype(int)

        trans = loadnpz('./recursive/0/orderData/withCancels/Trans_' + oType + '_' + name + '.npz')
        trans = miliArgs[reverse[trans]]
        YData = loadnpz('./recursive/0/orderData/withCancels/Y_' + oType + '_' + name + '.npz')

        #allShares = loadnpz('./inputData/ITCH_LOB/NearBestOrders/' + name + '.npz') / 200.0 #[:, :, :, 1] / 200.0
        #midpoint = allShares.shape[1] // 2
        #allShares = allShares[:, midpoint-20:midpoint+20]
        #allShares = np.sum(allShares, axis=2)

        print (np.max(trans))

        price = YData[:, 1]

        if oType:
            bestPrice0 = loadnpz('./inputData/ITCH_LOB/bestPrice/Bid_' + name + '.npz')
        else:
            bestPrice0 = loadnpz('./inputData/ITCH_LOB/bestPrice/Ask_' + name + '.npz')

        print (bestPrice0.shape)
        #quit()
        print (np.min(bestPrice0))
        bestPrice0 = bestPrice0 - np.min(bestPrice0)
        M = 1000
        plt.scatter(trans[0::M], price[0::M])
        plt.scatter(trans[0::M], bestPrice0[trans][0::M])
        plt.show()
        quit()

        diff = price - bestPrice0[trans]
        plt.hist(price, bins=100, histtype='step')
        plt.hist(diff, bins=100, histtype='step')
        plt.show()

        #for a in range(0, 100):
        #    arg = np.random.choice(trans.shape[0], 1)[0]
        #    print (relPrice)





        quit()
        '''


        rec1 = 2
        choice1 = 3
        output1 = loadnpz('./recursive/' + str(0) + '/samplesMix/prob_' + str(choice1) + '.npz')[:, 1]
        #output2 = loadnpz('./recursive/' + str(1) + '/samplesMix/prob_' + str(choice1) + '.npz')[:, 1]
        #output3 = loadnpz('./recursive/' + str(2) + '/samplesMix/prob_' + str(choice1) + '.npz')[:, 1]
        output2 = loadnpz('./recursive/' + str(3) + '/samplesMix/prob_' + str(choice1) + '.npz')[:, 1]

        profit = loadnpz('./recursive/' + str(1) + '/neuralInputs/2/profit_' + str(choice1) + '.npz')
        rec1 = 0
        order = loadnpz('./recursive/' + str(rec1) + '/neuralInputs/2/order_' + str(choice1) + '.npz')
        spread = loadnpz('./recursive/' + str(rec1) + '/neuralInputs/2/spread_' + str(choice1) + '.npz') * 2
        allShares = loadnpz('./recursive/' + str(rec1) + '/neuralInputs/2/shares_' + str(choice1) + '.npz') / 20


        diff = output2 - output1


        plt.plot(np.mean(order[:, 1, :], axis=0))
        plt.plot(np.mean(order[:, 1, :][diff < -0.05], axis=0))
        plt.show()
        #plt.hist(output1[diff < -0.05], bins=200, histtype='step', range=(-1.0, 0.2))
        #plt.hist(output2[diff < -0.05], bins=200, histtype='step', range=(-1.0, 0.2))
        #plt.show()

        plt.plot(np.mean(np.sum(allShares, axis=2), axis=0))
        #plt.plot(np.mean(np.sum(allShares[output < -1.2], axis=2), axis=0))
        plt.plot(np.mean(np.sum(allShares[diff < -0.05], axis=2), axis=0))
        plt.show()

        plt.plot(np.mean(np.sum(allShares, axis=1), axis=0))
        #plt.plot(np.mean(np.sum(allShares[output < -1.2], axis=1), axis=0))
        plt.plot(np.mean(np.sum(allShares[diff < -0.05], axis=1), axis=0))
        plt.show()

        #plt.plot(np.mean(order, axis=0))
        #plt.plot(np.mean(order[output < -1.2], axis=0))
        #plt.plot(np.mean(order[output > 0], axis=0))
        #plt.show()


        quit()

        #plt.hist2d(output3, output4, bins=50)
        #plt.show()
        #quit()
        print (scipy.stats.pearsonr(output1, output2))
        print (scipy.stats.pearsonr(output2, output3))
        print (scipy.stats.pearsonr(output3, output4))
        print (scipy.stats.pearsonr(output1, output4))

        #plt.hist(output1[output > 0], bins=200, histtype='step', range=(-1.0, 0.2))
        plt.hist(output1, bins=200, histtype='step', range=(-1.0, 0.2))
        plt.hist(output2, bins=200, histtype='step', range=(-1.0, 0.2))
        plt.hist(output3, bins=200, histtype='step', range=(-1.0, 0.2))
        plt.hist(output4, bins=200, histtype='step', range=(-1.0, 0.2))
        plt.show()
        plt.hist(output1, bins=200, histtype='step', range=(-1.2, -.75))
        plt.hist(output2, bins=200, histtype='step', range=(-1.2, -.75))
        plt.hist(output3, bins=200, histtype='step', range=(-1.2, -.75))
        plt.hist(output4, bins=200, histtype='step', range=(-1.2, -.75))
        plt.show()

        quit()







        '''





        output = loadnpz('./recursive/0/outputPredictions/' + oType + '_' + name + '.npz')[:, 1]
        finalData = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')
        YData = loadnpz('./recursive/0/orderData/withCancels/Y_' + oType + '_' + name + '.npz')


        argsPrice = np.argwhere(finalData[:, 1] == 0)[:, 0]
        plt.hist(output[argsPrice], bins=100, histtype='step')
        plt.hist(output, bins=100, histtype='step')
        plt.show()
        quit()


        allShares = loadnpz('./inputData/ITCH_LOB/NearBestOrders/' + name + '.npz') / 200.0 #[:, :, :, 1] / 200.0
        midpoint = allShares.shape[1] // 2
        allShares = allShares[:, midpoint-20:midpoint+20]
        allShares = np.sum(allShares, axis=2)

        #finalData = finalData[output > -0.2]
        #YData = YData[output > -0.2]
        #output = output[output > -0.2]

        #print (scipy.stats.pearsonr(finalData[:, 2], output))
        #quit()





        argsAbove = np.argwhere(output > -0.2)[:, 0]
        argRandom = np.random.choice(argsAbove.shape[0], 100)
        argRandom = argsAbove[argRandom]
        times1 = finalData[argRandom, 0]

        for a in range(0, 100):
            time1 = times1[a]

            argsSameTime = np.argwhere(finalData[:, 0] == time1)[:, 0]
            if argsSameTime.shape[0] > 1:

                quePos = YData[argsSameTime, 2]
                depth = finalData[argsSameTime, 2]
                relPrice = finalData[argsSameTime, 1]


                #plt.hist(output[argsSameTime][relPrice == 0], bins=100)
                #plt.scatter(depth[relPrice == 0], output[argsSameTime][relPrice == 0])
                #plt.scatter(relPrice, depth, c=output[argsSameTime])

                shares1 = allShares[time1]

                relPriceU = np.unique(relPrice)
                for b in range(0, relPriceU.shape[0]):
                    price1 = relPriceU[b]
                    depth[relPrice == price1] = np.max(depth[relPrice == price1]) - depth[relPrice == price1]



                #plt.plot(shares1)
                #plt.show()
                plt.scatter(relPrice, depth, c=output[argsSameTime])
                #plt.scatter(relPrice, quePos, c=output[argsSameTime])
                plt.show()

        quit()















        #'''
        for name in names[8:12]:
            for oType in ['B', 'S']:

                #profit0 = loadnpz('./recursive/0/profitData/Trans_' + oType + '_' + name + '.npz')
                #profit1 = loadnpz('./recursive/1/profitData/Trans_' + oType + '_' + name + '.npz')
                profit2 = loadnpz('./recursive/2/profitData/Trans_' + oType + '_' + name + '.npz')
                depth = loadnpz('./recursive/2/profitData/X_' + oType + '_' + name + '.npz')[:, 2]

                print (name)

                #print (profit0.shape)
                #print (profit1.shape)
                #rint (profit2.shape)
                #print (np.mean(profit0))
                #print (np.mean(profit1))
                #print (np.mean(profit2))



        quit()


        output = loadnpz('./recursive/' + str(0) + '/samplesMix/prob_' + str(0) + '.npz')[:, 1]

        print (output[output < -1.8].shape[0] / output.shape[0])
        argsBad = np.argwhere(output < -1.8)[:, 0]

        output = loadnpz('./recursive/' + str(1) + '/samplesMix/prob_' + str(0) + '.npz')[:, 1]

        print (output[output < -0.8].shape[0] / output.shape[0])

        #plt.hist(output[argsBad][:, 0], bins=100)
        #plt.show()

        quit()
        #'''

        name = names[3]


        #for name in names:
        for oType in ['B', 'S']:
            #output = loadnpz('./recursive/3/outputPredictions/' + oType + '_' + name + '.npz')[:, 1]
            output1 = loadnpz('./recursive/0/samples/prob_' + oType + '_' + name + '.npz')#[:, 2]
            output2 = loadnpz('./recursive/1/samples/prob_' + oType + '_' + name + '.npz')#[:, 2]



            profit1 = loadnpz('./recursive/0/profitData/profitFull_' + oType + '_' + name + '.npz')
            #profit2 = loadnpz('./recursive/1/profitData/profitFull_' + oType + '_' + name + '.npz')
            profit2 = loadnpz('./recursive/1/profitData2/profitFull_' + oType + '_' + name + '.npz')
            position = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')[:, 0]

            #plt.hist(output1[:, 2], bins=100, histtype='step')
            #plt.hist(output2[:, 2], bins=100, histtype='step')
            #plt.show()
            #quit()
            #print (profit[:20, 1])
            #print (np.mean(profit[:, 1]))
            print (name, oType)
            #print (np.mean(profit1[:, 0][output2[:, 0].astype(int)][output2[:, 2] > 0]))
            #print (np.mean(profit2[:, 0][output2[:, 0].astype(int)][output2[:, 2] > 0]))
            profit1 = profit1[output2[:, 0].astype(int)][output2[:, 2] > 0]
            profit2 = profit2[output2[:, 0].astype(int)][output2[:, 2] > 0]
            max1 = np.max(position)
            position = position[output2[:, 0].astype(int)][output2[:, 2] > 0]
            diff = profit2 - profit1


            print (np.sum( diff[diff[:, 1] != 0, 0] ))
            print (np.sum( diff[np.logical_and(diff[:, 1] == 0, diff[:, 0] != 0), 0] ))
            print (np.mean( profit1[diff[:, 1] != 0, 0] ))

            #position[diff[:, 1] != 0]
            if position[diff[:, 1] != 0].shape[0] > 10:
                plt.hist(position[diff[:, 1] != 0] / max1, bins=100)
                plt.show()


            #print (np.mean(np.abs(profit1[:, 0][output2[:, 0].astype(int)][output2[:, 2] > 0])))
            #print (np.mean(np.abs(profit2[:, 0][output2[:, 0].astype(int)][output2[:, 2] > 0])))

            #print (np.mean(profit[:, 0][output2[:, 0].astype(int)][output2[:, 2] > 0]))
            #quit()

            #position = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')[:, 0]

            #position = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')[:, 0]
            #position1 = loadnpz('./recursive/1/orderData/withCancels/X_' + oType + '_' + name + '.npz')[:, 0]
            #argsGood0 = loadnpz('./recursive/1/orderData/withCancels/argsGood_' + oType + '_' + name + '.npz')

            #print (position.shape)
            #print (position1.shape)
            #quit()
            #argsGood1 = loadnpz('./recursive/1/profitData/argsGood_' + oType + '_' + name + '.npz')

            #print (output2[output2[:, 2] > 0].shape)
            #print (position.shape[0] / output2.shape[0])

            #print (argsGood0[np.isin(argsGood0, argsGood1) == False].shape)
            #print (position1[np.isin(argsGood0, argsGood1) == False].shape)

            #plt.hist(position1[np.isin(argsGood0, argsGood1) == False], bins=100, density=True)
            #plt.show()

            #quit()

            #position = position[output2[:, 0].astype(int)]

            #plt.hist(position[output2[:, 2] > 0], bins=100, density=True)
            #plt.show()
            #quit()




        quit()












        rec1 = 2
        choice1 = 0
        output1 = loadnpz('./recursive/' + str(0) + '/samplesMix/prob_' + str(choice1) + '.npz')[:, 1]
        output = loadnpz('./recursive/' + str(1) + '/samplesMix/prob_' + str(choice1) + '.npz')[:, 1]

        #plt.hist2d(output1, output, bins=50)
        #plt.show()
        #quit()
        #print (scipy.stats.pearsonr(output1, output))

        #plt.hist(output1[output > 0], bins=200, histtype='step', range=(-1.0, 0.2))
        plt.hist(output1, bins=200, histtype='step', range=(-1.0, 0.2))
        plt.hist(output, bins=200, histtype='step', range=(-1.0, 0.2))
        plt.show()

        profit = loadnpz('./recursive/' + str(1) + '/neuralInputs/2/profit_' + str(choice1) + '.npz')
        rec1 = 0
        order = loadnpz('./recursive/' + str(rec1) + '/neuralInputs/2/order_' + str(choice1) + '.npz')
        spread = loadnpz('./recursive/' + str(rec1) + '/neuralInputs/2/spread_' + str(choice1) + '.npz') * 2
        allShares = loadnpz('./recursive/' + str(rec1) + '/neuralInputs/2/shares_' + str(choice1) + '.npz') / 20
        depth = np.sum(order[:, 1, :], axis=1)
        #output = output[profit[:, 1] == 0]
        allShares = allShares[:, 1:-1, :]

        order = order[:, 0, :]
        order1 = np.matmul(order, np.arange(38)).astype(int)

        order1 = np.matmul(order, np.arange(38)).astype(int)



        spread = spread[profit[:, 1] == 0]
        order = order[profit[:, 1] == 0]
        allShares = allShares[profit[:, 1] == 0]
        depth = depth[profit[:, 1] == 0]
        output = output[profit[:, 1] == 0]
        profit = profit[profit[:, 1] == 0][:, 0]

        output = output[depth == 0]
        spread = spread[depth == 0]
        order = order[depth == 0]
        allShares = allShares[depth == 0]
        profit = profit[depth == 0]


        outputCutoff = 0.03

        #print(np.mean(profit))
        #print(np.mean(profit[output > 0.0]))
        #print(np.mean(profit[output > 0.03]))

        print(np.mean(spread[output > -0.4]))
        #print(np.mean(spread[output < -1.2]))
        print(np.mean(spread[output > 0.0]))

        #print(np.mean(depth[output > -0.4]))
        #print(np.mean(depth[output < -1.2]))

        allShares1 = np.sum(allShares, axis=2)

        plt.plot(np.mean(np.sum(allShares[output > -0.4], axis=2), axis=0))
        #plt.plot(np.mean(np.sum(allShares[output < -1.2], axis=2), axis=0))
        plt.plot(np.mean(np.sum(allShares[output > 0], axis=2), axis=0))
        plt.show()

        plt.plot(np.mean(np.sum(allShares[output > -0.4], axis=1), axis=0))
        #plt.plot(np.mean(np.sum(allShares[output < -1.2], axis=1), axis=0))
        plt.plot(np.mean(np.sum(allShares[output > 0], axis=1), axis=0))
        plt.show()

        plt.plot(np.mean(order[output > -0.4], axis=0))
        #plt.plot(np.mean(order[output < -1.2], axis=0))
        plt.plot(np.mean(order[output > 0], axis=0))
        plt.show()



        quit()




        '''

        Yargs0 = loadnpz('./recursive/' + str(0) + '/orderData/initial/Trans_' + oType + '_' + name + '.npz')
        Yargs1 = loadnpz('./recursive/' + str(1) + '/orderData/initial/Trans_' + oType + '_' + name + '.npz')
        print (Yargs0.shape)
        print (Yargs1.shape)

        Yargs0 = loadnpz('./recursive/' + str(0) + '/profitData/Trans_' + oType + '_' + name + '.npz')
        Yargs1 = loadnpz('./recursive/' + str(1) + '/profitData/Trans_' + oType + '_' + name + '.npz')
        print (Yargs0.shape)
        print (Yargs1.shape)

        print (np.mean(Yargs0))
        print (np.mean(Yargs1))

        quit()
        ##'''

        '''
        loopNumber = '1'
        diffs = []
        diffs1 = []
        for name in names:
            for oType in ['B', 'S']:
                transProfits0 = np.load('./recursive/' + str(0) + '/transProfits/final/transSalePrices_' + oType + '_' + name + '.npy')
                transProfits1 = np.load('./recursive/' + str(loopNumber) + '/transProfits/final/transSalePrices_' + oType + '_' + name + '.npy')

                profit0 = loadnpz('./recursive/' + str(0) + '/profitData/Trans_' + oType + '_' + name + '.npz')
                profit1 = loadnpz('./recursive/' + str(loopNumber) + '/profitData/Trans_' + oType + '_' + name + '.npz')

                diff = np.mean(transProfits1[:, 1]) - np.mean(transProfits0[:, 1])
                diff1 = np.mean(profit1) - np.mean(profit0)
                diffs1.append(diff1)
                print ("A")
                print (diff1)
                print (diff)
                if oType == 'B':
                    diffs.append(-1.0 * diff)
                else:
                    diffs.append(diff)
        #print (np.mean(diffs))
        plt.plot(diffs1)
        plt.plot(diffs)
        plt.show()
        quit()
        '''







        '''
        choice1 = 0
        profit = loadnpz('./recursive/' + str(0) + '/neuralInputs/2/profit_' + str(choice1) + '.npz')
        #print (profit[:10])
        #quit()
        print (np.mean(profit[:, 0][profit[:, 1] == 0]))
        profit = loadnpz('./recursive/' + str(1) + '/neuralInputs/2/profit_' + str(choice1) + '.npz')
        print (np.mean(profit[:, 0][profit[:, 1] == 0]))
        quit()
        '''

        #'''
        rec1 = 2
        choice1 = 0
        output1 = loadnpz('./recursive/' + str(0) + '/samplesMix/prob_' + str(choice1) + '.npz')[:, 1]
        output2 = loadnpz('./recursive/' + str(1) + '/samplesMix/prob_' + str(choice1) + '.npz')[:, 1]
        #output3 = loadnpz('./recursive/' + str(2) + '/samplesMix/prob_' + str(choice1) + '.npz')[:, 1]
        #output4 = loadnpz('./recursive/' + str(3) + '/samplesMix/prob_' + str(choice1) + '.npz')[:, 1]

        profit = loadnpz('./recursive/' + str(1) + '/neuralInputs/2/profit_' + str(choice1) + '.npz')#[:, 0]
        output2 = output2[profit[:, 1] == 0]

        #print (np.mean(output2[output2 > 0]))
        #print (np.mean(profit[output2 > 0]))
        #quit()


        print (output1.shape)
        print (output2.shape)
        #print (scipy.stats.pearsonr(output1, output2))
        #print (scipy.stats.spearmanr(output1, output2))
        #print (scipy.stats.pearsonr(output2, output3))
        #print (scipy.stats.pearsonr(output3, output4))
        #print (scipy.stats.pearsonr(output1, output4))
        #quit()
        plt.hist(output1, histtype='step', bins=200, range=(-1.1, 0.2))
        plt.hist(output2, histtype='step', bins=200, range=(-1.1, 0.2))
        #plt.hist(output3[:, 1], histtype='step', bins=50, range=(-0.9, 0.2))
        #plt.hist(output4[:, 1], histtype='step', bins=50, range=(-0.9, 0.2))
        plt.show()
        quit()
        #'''




        '''
        sums3 = np.array(np.load('./temporary/sums3.npy'))
        sums4 = np.array(np.load('./temporary/sums5.npy'))
        sums3 = sums3 / np.max(sums3)
        sums4 = sums4 / np.max(sums4)

        plt.plot(sums3[:len(sums4)])
        plt.plot(sums4)
        plt.show()
        quit()
        #'''


        sums1 = []

        for name in names:
            for oType in ['B', 'S']:
                #output = loadnpz('./recursive/3/outputPredictions/' + oType + '_' + name + '.npz')[:, 1]
                #output = loadnpz('./recursive/0/samples/prob_' + oType + '_' + name + '.npz')[:, 2]
                output2 = loadnpz('./recursive/1/samples/prob_' + oType + '_' + name + '.npz')[:, 2]
                output3 = loadnpz('./recursive/2/samples/prob_' + oType + '_' + name + '.npz')[:, 2]
                output4 = loadnpz('./recursive/3/samples/prob_' + oType + '_' + name + '.npz')[:, 2]
                #profit = loadnpz('./recursive/3/profitData/profitFull_' + oType + '_' + name + '.npz')
                #position = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')[:, 0]
                print ("A")
                #print (scipy.stats.pearsonr(output1, output2))
                print (scipy.stats.pearsonr(output2, output3))
                print (scipy.stats.pearsonr(output3, output4))
                print (scipy.stats.pearsonr(output2, output4))
                '''
                position = loadnpz('./recursive/0/orderData/withCancels/Y_' + oType + '_' + name + '.npz')[:, 1]

                output = output[np.argsort(position)]
                profit = profit[np.argsort(position)]


                #position = position / np.max(position)
                #print (output.shape, position.shape)
                #print (np.mean(p))

                output = output[profit[:, 1] == 0]
                profit = profit[profit[:, 1] == 0][:, 0]

                profitAbove = profit[output>0.1]

                print (np.mean(profitAbove))
                print (np.mean(profitAbove[1::2]))
                print (np.mean(profitAbove[0::2]))
                midpoint = profitAbove.shape[0] // 2
                print (np.mean(profitAbove[midpoint:]))
                print (np.mean(profitAbove[:midpoint]))
                '''


                #plt.hist(position[output>0.1], bins=100, histtype='step')
                #plt.show()
                #profit = loadnpz('./recursive/2_1/profitData/profitFull_' + oType + '_' + name + '.npz')
                #profit = loadnpz('./recursive/3/profitData/profitFull_' + oType + '_' + name + '.npz')
                #output = loadnpz('./recursive/3/samples/prob_' + oType + '_' + name + '.npz')
                #print (np.mean(profit[:, 0][profit[:, 1] == 0]))
                #plt.hist(output, bins=100, histtype='step', range=(-1.0, 0.5))
                #plt.show()
                #quit()

                '''
                profit = profit[output[:, 0].astype(int)]
                output = output[:, 2]

                output = output[profit[:, 1] == 0]
                profit = profit[profit[:, 1] == 0]



                #print (np.mean(profit[:, 0][output > -0.3]))
                #print (np.mean(profit[:, 0][output < -0.5]))
                #quit()
                sums1.append(np.sum(profit[:, 0][output > 0.1]))
                #reverser =
                #finalData = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')
                #YData = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')
                np.save('./temporary/sums5.npy', sums1)
                #'''
        quit()

        rec1 = 2
        choice1 = 0
        output = loadnpz('./recursive/' + str(rec1) + '/samplesMix/prob_' + str(choice1) + '.npz')[:, 1]
        #profit = loadnpz('./recursive/' + str(rec1) + '/neuralInputs/2/profit_' + str(choice1) + '.npz')
        rec1 = 0
        order = loadnpz('./recursive/' + str(rec1) + '/neuralInputs/2/order_' + str(choice1) + '.npz')
        spread = loadnpz('./recursive/' + str(rec1) + '/neuralInputs/2/spread_' + str(choice1) + '.npz') * 2
        allShares = loadnpz('./recursive/' + str(rec1) + '/neuralInputs/2/shares_' + str(choice1) + '.npz') / 20
        depth = np.sum(order[:, 1, :], axis=1)
        #output = output[profit[:, 1] == 0]
        allShares = allShares[:, 1:-1, :]
        #print (allShares.shape)
        #quit()
        #print (order.shape)
        #quit()
        order = order[:, 0, :]
        order1 = np.matmul(order, np.arange(38)).astype(int)

        posVal = 17 #17, 20
        spread = spread[np.isin(order1, np.array([posVal]))]
        order = order[np.isin(order1, np.array([posVal]))]
        allShares = allShares[np.isin(order1, np.array([posVal]))]
        depth = depth[np.isin(order1, np.array([posVal]))]
        output = output[np.isin(order1, np.array([posVal]))]

        #plt.hist(output, bins=100, histtype='step', range=(-1.2, 0.2))
        #plt.show()
        #quit()

        order1 = np.matmul(order, np.arange(38)).astype(int)

        #argsBad = np.argwhere(output < -0.7)[:, 0]
        argsBad = np.argwhere(output > 0.0)[:, 0]

        argChoice = np.random.choice(argsBad, 10)#[0]

        #allShares = allShares[:, 18, :]
        allShares = np.sum(allShares, axis=2)

        #plt.plot(np.mean(allShares, axis=0))
        #plt.plot(np.mean(allShares[output > 0.0], axis=0))
        #plt.show()

        for a in range(0, 10):
            print (np.sum(allShares[argChoice[a]]))
            plt.plot(allShares[argChoice[a]])
        plt.show()

        #plt.imshow(allShares[argChoice])
        #plt.show()

        quit()
        #plt.hist(output[profit[:, 1] == 0], bins=100, histtype='step', range=(-1.2, 0.2))
        #plt.hist(output[profit[:, 1] == 1], bins=100, histtype='step', range=(-1.2, 0.2))
        #plt.show()


        '''
        spread = spread[profit[:, 1] == 0]
        order = order[profit[:, 1] == 0]
        allShares = allShares[profit[:, 1] == 0]
        depth = depth[profit[:, 1] == 0]
        output = output[profit[:, 1] == 0]
        profit = profit[profit[:, 1] == 0][:, 0]

        output = output[depth == 0]
        spread = spread[depth == 0]
        order = order[depth == 0]
        allShares = allShares[depth == 0]
        profit = profit[depth == 0]
        '''

        outputCutoff = 0.03

        #print(np.mean(profit))
        #print(np.mean(profit[output > 0.0]))
        #print(np.mean(profit[output > 0.03]))

        print(np.mean(spread[output > -0.4]))
        print(np.mean(spread[output < -0.7]))

        print(np.mean(depth[output > -0.4]))
        print(np.mean(depth[output < -0.7]))

        allShares1 = np.sum(allShares, axis=2)


        #order = order[:, 0, :]
        '''
        for a in range(0, 10):
            allShares_1 = allShares1[output < -0.7][a]
            allShares_2 = allShares1[output > -0.4][a]
            plt.plot(allShares_1, color='b')
            plt.scatter([order1[output < -0.7][a]], [allShares_1[order1[output < -0.7][a]]], color='b')
            #plt.plot(np.sum(allShares[output < 0], axis=2)[a])
            plt.plot(allShares_2, color='r')
            #plt.plot(order[output > 0.1][a], color='r')
            plt.scatter([order1[output > -0.4][a]], [allShares_2[order1[output > -0.4][a]]], color='r')
            plt.show()
        quit()
        '''


        plt.plot(np.mean(np.sum(allShares[output > -0.4], axis=2), axis=0))
        plt.plot(np.mean(np.sum(allShares[output < -0.7], axis=2), axis=0))
        #plt.plot(np.mean(np.sum(allShares[output > 0], axis=2), axis=0))
        plt.show()

        plt.plot(np.mean(np.sum(allShares[output > -0.4], axis=1), axis=0))
        plt.plot(np.mean(np.sum(allShares[output < -0.7], axis=1), axis=0))
        #plt.plot(np.mean(np.sum(allShares[output > 0], axis=1), axis=0))
        plt.show()

        plt.plot(np.mean(order[output > -0.4], axis=0))
        plt.plot(np.mean(order[output < -0.7], axis=0))
        #plt.plot(np.mean(order[output > 0], axis=0))
        plt.show()


        quit()







        rec1 = 3
        choice1 = 0
        output = loadnpz('./recursive/' + str(4) + '/samplesMix/prob_' + str(choice1) + '.npz')
        profit = loadnpz('./recursive/' + str(rec1) + '/neuralInputs/2/profit_' + str(choice1) + '.npz')

        print (np.mean(profit[output[:, 1] > 0]))










        quit()







        '''
        for name in names[:4]:
            print ("A")
            for oType in ['B', 'S']:
                outputFull = loadnpz('./recursive/' + '2' + '/samples/prob_' + oType + '_' + name + '.npz')

                size1 = outputFull[outputFull[:, 2] > 0.0].shape[0]
                print (size1)
        '''


        #quit()



        for name in names:

            #outputFull = loadnpz('./recursive/' + '1' + '/samples/prob_' + oType + '_' + name + '.npz')
            output = loadnpz('./recursive/' + '1' + '/outputPredictions/' + oType + '_' + name + '.npz')[:, 1]
            finalData = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')
            YData = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')


            data = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz')
            argsExecute = np.argwhere(np.isin(data[:, 3], np.array(['E', 'F']) ))[:, 0]
            argsExecute = argsExecute[np.argwhere(data[argsExecute, 4] == 'S')[:, 0]]
            #print (argsExecute.shape)
            #quit()

            #miliseconds = data[:, 0].astype(int)
            #_, miliArgs = np.unique(miliseconds[-1::-1], return_index=True)
            #miliArgs = data.shape[0] - 1 - miliArgs

            #print (finalData.shape)
            #print (finalData[np.isin(finalData[:, 0], miliArgs) == False].shape)
            #quit()

            X_unique, X_inverse = np.unique(finalData[:, 0], return_inverse=True)
            finalData[:, 0] = X_inverse

            #allShares = loadnpz('./inputData/ITCH_LOB/NearBestOrders/' + name + '.npz') / 200.0 #[:, :, :, 1] / 200.0
            #midpoint = allShares.shape[1] // 2
            #allShares = allShares[:, midpoint-20:midpoint+20]
            #allShares = np.sum(allShares, axis=2)

            argsBad = np.argwhere(output < -0.7)[:, 0]
            '''
            posBad = finalData[argsBad, 0]
            posNext = posBad + 100

            bestBid = loadnpz('./inputData/ITCH_LOB/bestPrice/Bid_' + name + '.npz')
            bestAsk = loadnpz('./inputData/ITCH_LOB/bestPrice/Ask_' + name + '.npz')

            changeBid = bestBid[posNext] - bestBid[posBad]
            changeAsk = bestAsk[posNext] - bestAsk[posBad]

            print (oType)
            print (np.mean(changeBid))
            print (np.mean(changeAsk))
            '''


            for b in range(0, 200):
                print (b)
                argChoice = np.random.choice(argsBad, 1)[0]

                pos1 = finalData[argChoice, 0]
                price1 = YData[argChoice, 1]

                argsSame = np.argwhere(np.logical_and(np.abs(finalData[:, 0] - pos1) < 50, YData[:, 1] == price1))[:, 0]
                relTime = finalData[argsSame, 0] #- pos1
                val = output[argsSame]

                if np.max(val) > -0.6:
                    #posses2 = X_unique[pos1-50:pos1+50]
                    #pos2 = X_unique[pos1]
                    #diff =
                    pos2_1 = X_unique[pos1 - 50]
                    pos2_2 = X_unique[pos1 + 50]
                    pos2 = (pos2_1 + pos2_2) / 2
                    diff = (pos2_2 - pos2_1) / 2

                    print (diff)

                    nearEx = argsExecute[np.abs(argsExecute - pos2) < diff]
                    print (nearEx.shape)

                    plt.scatter(X_unique[relTime], val)
                    plt.scatter(nearEx, (nearEx * 0.0) - 0.4)
                    plt.show()

                    #plt.imshow(allShares[posses2].T)
                    #plt.show()

                #argsSame1 = np.argwhere(np.logical_and(finalData[:, 0] == pos1 - 1000, YData[:, 1] == price1))[:, 0]
                #argsSame2 = np.argwhere(np.logical_and(finalData[:, 0] == pos1, YData[:, 1] == price1))[:, 0]
                #argsSame3 = np.argwhere(np.logical_and(finalData[:, 0] == pos1 + 1000, YData[:, 1] == price1))[:, 0]
                #min1 = np.min(np.array([output[argsSame1], output[argsSame2], output[argsSame3]]))
                #max1 = np.max(np.array([output[argsSame1], output[argsSame2], output[argsSame3]]))

                if False:#(argsSame1.shape[0] != 0) and (argsSame3.shape[0] != 0):
                    if (np.mean(output[argsSame1]) > -0.6) and (np.mean(output[argsSame3]) > -0.6):
                        #allShares = allShares[X[:, 0]]
                        print (np.mean(output[argsSame1]))
                        print (np.mean(output[argsSame2]))
                        print (np.mean(output[argsSame3]))
                        allShares_1 = allShares[pos1 - 1000]
                        allShares_2 = allShares[pos1]
                        allShares_3 = allShares[pos1 + 1000]
                        plt.plot(allShares_1)
                        plt.plot(allShares_2)
                        plt.plot(allShares_3)
                        plt.show()


                #min1, max1 = -1.2, 0.2
                #plt.hist(output[argsSame1], range=(min1, max1), histtype='step', bins=50)
                #plt.hist(output[argsSame2], range=(min1, max1), histtype='step', bins=50)
                #plt.hist(output[argsSame3], range=(min1, max1), histtype='step', bins=50)
                #plt.show()
            quit()

            maxX = np.max(finalData[:, 0])
            #finalData = finalData[outputFull[:, 0].astype(int)]
            X1 = finalData[:, 0][output < -0.7] / maxX
            X2 = finalData[:, 0][output > -0.5] / maxX

            #plt.hist(X, bins=100)
            plt.hist(X1, bins=100, histtype='step', range=(0, 0.04))#, density=True)
            plt.hist(X2, bins=100, histtype='step', range=(0, 0.04))#, density=True)
            plt.show()


        quit()

        for name in names:
            output = loadnpz('./recursive/' + '1' + '/outputPredictions/' + 'B' + '_' + name + '.npz')
            plt.hist(output[:, 1], bins=100, histtype='step')
            output = loadnpz('./recursive/' + '1' + '/outputPredictions/' + 'S' + '_' + name + '.npz')
            plt.hist(output[:, 1], bins=100, histtype='step')
            plt.show()


        quit()


        quit()



        #rec1 = 1
        choice1 = 0
        output1 = loadnpz('./recursive/' + str(0) + '/samplesMix/prob_' + str(choice1) + '.npz')
        output2 = loadnpz('./recursive/' + str(1) + '/samplesMix/prob_' + str(choice1) + '.npz')
        output3 = loadnpz('./recursive/' + str(2) + '/samplesMix/prob_' + str(choice1) + '.npz')

        plt.hist(output1[:, 1], bins=100, histtype='step', range=(-1.2, 0.2))
        plt.hist(output2[:, 1], bins=100, histtype='step', range=(-1.2, 0.2))
        plt.hist(output3[:, 1], bins=100, histtype='step', range=(-1.2, 0.2))
        plt.show()


        quit()
        profit1 = loadnpz('./recursive/' + str(1) + '/neuralInputs/2/profit_' + str(choice1) + '.npz')
        profit2 = loadnpz('./recursive/' + str(2) + '/neuralInputs/2/profit_' + str(choice1) + '.npz')
        argsNew = np.argwhere(profit2[:, 1] == 0)[:, 0]
        #print (np.mean(profit2[argsNew, 0]))
        #argsNew = np.argwhere(profit1[:, 1] == 1)[:, 0]
        #print (np.mean(profit2[argsNew, 0]))
        print (np.mean(profit2[argsNew, 0]))
        argsNew = np.argwhere(np.logical_and(profit1[:, 1] == 1,  profit2[:, 1] == 0  ))[:, 0]
        print (np.mean(profit2[argsNew, 0]))
        quit()





        profits1 = []
        profits2 = []
        profits3 = []

        for c in range(0, 11):
            for oType in ['B', 'S']:
                print ("A")
                name = names[c]
                #profit = loadnpz('./recursive/0/profitData/Trans_' + oType + '_' + name + '.npz')
                #print (profit.shape)
                #mean = np.mean(profit)
                #profits1.append(mean)
                #print (np.mean(mean))
                profit = loadnpz('./recursive/1/profitData/Trans_' + oType + '_' + name + '.npz')
                #print (profit.shape)
                #mean = np.mean(profit)
                #profits2.append(mean)
                #print (np.mean(mean))
                profit = loadnpz('./recursive/2/profitData/Trans_' + oType + '_' + name + '.npz')
                #print (profit.shape)
                #mean = np.mean(profit)# - mean
                #profits3.append(mean)
                #print (np.mean(mean))

                #output = loadnpz('./recursive/' + '0' + '/outputPredictions/' +  oType + '_' + name + '.npz')[:, 1]
                #plt.hist(output, bins=100, histtype='step')
            #if c % 4 == 0:
            #    plt.plot(profits1)
            #    plt.plot(profits2)
            #    plt.show()
        #plt.show()
        plt.plot(profits1)
        plt.plot(profits2)
        plt.plot(profits3)
        plt.plot(np.array(profits3) - np.array(profits1))
        plt.plot((np.array(profits2) - np.array(profits1)) * 0.0)
        plt.show()
        quit()



        rec1 = 1
        choice1 = 0
        output1 = loadnpz('./recursive/' + str(rec1) + '/samplesMix2/prob_' + str(choice1) + '.npz')
        #argsGood = loadnpz('./recursive/' + str(rec1) + '/neuralInputs/2/argsGood_' + str(choice1) + '.npz')
        #print (argsGood.shape)
        #output1 = output1[np.argsort(argsGood)]

        #plt.hist(output[:, 1], bins=100, histtype='step')
        rec1 = 1
        profit = loadnpz('./recursive/' + str(rec1) + '/neuralInputs/2/profit_' + str(choice1) + '.npz')
        output = loadnpz('./recursive/' + str(rec1) + '/samplesMix/prob_' + str(choice1) + '.npz')
        #argsGood = loadnpz('./recursive/' + str(rec1) + '/neuralInputs/2/argsGood_' + str(choice1) + '.npz')
        #output = output[np.argsort(argsGood)]
        #profit = profit[np.argsort(argsGood)]

        #plt.hist(output1[:, 1][output1[:, 1] < -0.7], bins=100, histtype='step')
        plt.hist(output[:, 1][output1[:, 1] < -0.7], bins=100, histtype='step')

        #output = output[profit[:, 1] == 1]
        #output1 = output1[profit[:, 1] == 1]
        #plt.hist(output[:, 1], bins=100, histtype='step')
        #plt.hist(output1[:, 1], bins=100, histtype='step')
        #plt.hist2d(output[:, 0], output[:, 1], bins=100)

        plt.show()
        quit()


        rec1 = 1
        choice1 = 0

        sums1 = []
        sums2 = []

        for choice1 in range(0, 5):
            print (choice1)
            order = loadnpz('./recursive/' + str(rec1) + '/neuralInputs/2/order_' + str(choice1) + '.npz')
            spread = loadnpz('./recursive/' + str(rec1) + '/neuralInputs/2/spread_' + str(choice1) + '.npz') * 2
            allShares = loadnpz('./recursive/' + str(rec1) + '/neuralInputs/2/shares_' + str(choice1) + '.npz')
            output = loadnpz('./recursive/' + str(rec1) + '/samplesMix/prob_' + str(choice1) + '.npz')[:, 1]
            profit = loadnpz('./recursive/' + str(rec1) + '/neuralInputs/2/profit_' + str(choice1) + '.npz')#[:, 0]
            output = output[profit[:, 1] == 0]
            spread = spread[profit[:, 1] == 0]
            order = order[profit[:, 1] == 0]
            allShares = allShares[profit[:, 1] == 0]
            profit = profit[profit[:, 1] == 0][:, 0]

            depth = np.sum(order[:, 1, :], axis=1)

            output = output[depth == 0]
            spread = spread[depth == 0]
            order = order[depth == 0]
            allShares = allShares[depth == 0]
            profit = profit[depth == 0]

            outputCutoff = 0.03

            sums1.append(np.sum(profit[output > 0.0]))
            sums2.append(np.sum(profit[output > 0.03]))

            #print (np.mean(profit[output > outputCutoff]))
        plt.plot(sums1)
        plt.show()
        plt.plot(sums2)
        plt.show()
        quit()


        print (np.mean(spread))
        print (np.mean(spread[output > outputCutoff]))
        order = order[:, 0, :]

        plt.plot(np.mean(np.sum(allShares, axis=2), axis=0))
        plt.plot(np.mean(np.sum(allShares[output > outputCutoff], axis=2), axis=0))
        plt.show()

        #plt.plot(np.mean(np.sum(allShares, axis=1), axis=0))
        #plt.plot(np.mean(np.sum(allShares[output > 0], axis=1), axis=0))
        #plt.show()

        plt.plot(np.mean(order, axis=0))
        plt.plot(np.mean(order[output > outputCutoff], axis=0))
        plt.show()



        #for a in range(3, 8):
        #    plt.hist(output[spread == a], bins=100, histtype='step', range=(-1, 0.2))#range=(-1, 0.2))
        #plt.show()
        quit()


        #rec1 = 0
        #choice1 = 0
        #output = loadnpz('./recursive/' + str(rec1) + '/samplesMix/prob_' + str(choice1) + '.npz')[:, 1]
        #profit = loadnpz('./recursive/' + str(rec1) + '/neuralInputs/2/profit_' + str(choice1) + '.npz')#[:, 0]
        #output = output[profit[:, 1] == 0]
        #plt.hist(output, bins=100, histtype='step', range=(0, 0.2))

        rec1 = 1
        for choice1 in range(0, 5):
            #choice1 = 0
            output = loadnpz('./recursive/' + str(rec1) + '/samplesMix/prob_' + str(choice1) + '.npz')[:, 1]
            profit = loadnpz('./recursive/' + str(rec1) + '/neuralInputs/2/profit_' + str(choice1) + '.npz')#[:, 0]
            output = output[profit[:, 1] == 0]
            profit = profit[profit[:, 1] == 0][:, 0]

            print (np.mean(profit[output > 0]))
        quit()


        plt.hist(output, bins=100, histtype='step', range=(0, 0.2))#range=(-1, 0.2))
        plt.show()



        rec1 = 1
        choice1 = 0
        output = loadnpz('./recursive/' + str(rec1) + '/samplesMix/prob_' + str(choice1) + '.npz')[:, 1]

        #order = loadnpz('./recursive/' + str(0) + '/neuralInputs/2/order_' + str(a) + '.npz')
        #spread = loadnpz('./recursive/' + str(0) + '/neuralInputs/2/spread_' + str(a) + '.npz')
        #allShares = loadnpz('./recursive/' + str(0) + '/neuralInputs/2/shares_' + str(a) + '.npz')
        profit = loadnpz('./recursive/' + str(rec1) + '/neuralInputs/2/profit_' + str(choice1) + '.npz')#[:, 0]
        output = output[profit[:, 1] == 0]
        profit = profit[profit[:, 1] == 0][:, 0]


        print (np.mean(output))
        print (np.mean(profit))

        vals1 = []
        vals2 = []
        for a in range(0, 9):
            args = np.argwhere(np.abs(output + (0.1 * a) ) < 0.05)[:, 0]
            vals1.append(np.mean(output[args]))
            vals2.append(np.mean(profit[args]))
        #print (np.mean(profit[np.abs(output + 0.5) < 0.05]))
        #print (np.mean(output[np.abs(output + 0.5) < 0.05]))
        #print (np.mean(profit[np.abs(output + 0.2) < 0.05]))

        plt.plot(vals1)
        plt.plot(vals2)
        plt.show()
        quit()

        #output = loadnpz('./recursive/' + '0' + '/outputPredictions/' +  oType + '_' + name + '.npz')[:, 1]
        #output = loadnpz('./recursive/' + '0' + '/samplesMix/prob_' + str(0) + '.npz')
        #plt.hist(output[:, 1], bins=100, histtype='step', range=(0, 0.2))
        #output = loadnpz('./recursive/' + '1' + '/samplesMix/prob_' + str(0) + '.npz')
        plt.hist(output, bins=100, histtype='step', range=(-1, 0.2))#range=(-1, 0.2))
        plt.show()
        quit()



        '''
        for c in range(0, 4):
            for oType in ['B', 'S']:
                print ("A")
                name = names[c]
                output = loadnpz('./recursive/' + '0' + '/outputPredictions/' +  oType + '_' + name + '.npz')[:, 1]
                output = output - np.mean(output)
                y, x = np.histogram(output, bins=100)
                x = (x[1:] + x[:-1]) / 2
                y = np.log(y)
                y = np.max(y) - y
                y = y ** 0.5
                y[x < 0] = -1 * y[x < 0]
                plt.plot(y)
        plt.show()
        quit()
        '''
        profits1 = []
        profits2 = []

        for c in range(0, 15):
            for oType in ['B', 'S']:
                print ("A")
                name = names[c]
                profit = loadnpz('./recursive/0/profitData/Trans_' + oType + '_' + name + '.npz')
                #print (profit.shape)
                mean = np.mean(profit)
                #profits1.append(mean)
                #print (np.mean(mean))
                profit = loadnpz('./recursive/1/profitData/Trans_' + oType + '_' + name + '.npz')
                #print (profit.shape)
                mean = np.mean(profit) - mean
                profits2.append(mean)
                #print (np.mean(mean))

                #output = loadnpz('./recursive/' + '0' + '/outputPredictions/' +  oType + '_' + name + '.npz')[:, 1]
                #plt.hist(output, bins=100, histtype='step')
            if c % 4 == 0:
            #    plt.plot(profits1)
                plt.plot(profits2)
                plt.show()
        #plt.show()
        #plt.plot(profits1)
        plt.plot(profits2)
        plt.show()
        quit()

        output = loadnpz('./recursive/' + '0' + '/outputPredictions/' +  oType + '_' + name + '.npz')[:, 1]
        plt.hist(output, bins=100, histtype='step')
        plt.show()
        quit()

        #outputFull = loadnpz('./recursive/' + '0_1' + '/samples/prob_' + oType + '_' + name + '.npz')
        #plt.hist(outputFull[:, 2], bins=100, histtype='step')
        #outputFull = loadnpz('./recursive/' + str(1) + '/samples/prob_' + oType + '_' + name + '.npz')
        #plt.hist(outputFull[:, 2], bins=100, histtype='step')
        #plt.show()
        #quit()


        outputFull = loadnpz('./recursive/' + '0' + '/samplesMix/prob_' + str(1) + '.npz')
        #order1 = loadnpz('./recursive/' + str(0) + '/neuralInputs/2/order_' + str(a) + '.npz')
        spread1 = loadnpz('./recursive/' + str(0) + '/neuralInputs/2/spread_' + str(1) + '.npz') * 2
        #allShares = loadnpz('./recursive/' + str(0) + '/neuralInputs/2/shares_' + str(a) + '.npz')
        #profit = loadnpz('./recursive/' + str(0) + '/neuralInputs/2/profit_' + str(0) + '.npz')
        #depth1 = np.sum(order1[:, 1, :], axis=1)

        #print (np.mean(profit[outputFull[:, 1] > 0.05]))
        for c in range(3, 13):
            plt.hist(outputFull[:, 1][spread1 == c], bins=100, histtype='step')

        #plt.hist(spread1, bins=100, histtype='step', range=(0, 10))
        plt.show()
        quit()
        #plt.hist(outputFull[:, 1], bins=100, histtype='step')

        #plt.hist(outputFull[:, 1], bins=100, histtype='step')#, range=(0, 1))
        #outputFull = loadnpz('./recursive/' + '0_1' + '/samplesMix/prob_' + str(c) + '.npz')
        #plt.hist(outputFull[:, 1], bins=100, histtype='step')#, range=(0, 1))
        #outputFull = loadnpz('./recursive/' + str(1) + '/samplesMix/prob_' + str(c) + '.npz')
        ##print (outputFull.shape)
        #plt.hist(outputFull[:, 1], bins=100, histtype='step', range=(0, 1))
        #plt.show()
        quit()

        profits1 = []
        profits2 = []
        shapes1 = []
        shapes2 = []

        for c in range(0, 4):
            for oType in ['B', 'S']:
                name = names[c]
                print ("C")
                #output = loadnpz('./recursive/' + str(1) + '/outputPredictions/' + oType + '_' + name + '.npz')[:, 1]
                #argsGood_1 = loadnpz('./recursive/0/orderData/withCancels/argsGood_' + oType + '_' + name + '.npz')
                #reverser = np.zeros(np.max(argsGood_1)+1).astype(int)
                #reverser[argsGood_1] = np.arange(argsGood_1.shape[0])

                profit = loadnpz('./recursive/0/profitData/Trans_' + oType + '_' + name + '.npz')
                mean1 = np.mean(profit)
                #argsGood = loadnpz('./recursive/0/profitData/argsGood_' + oType + '_' + name + '.npz').astype(int)
                #print (np.mean(output[reverser[argsGood]]))
                #plt.hist(output[reverser[argsGood]], bins=100, histtype='step')
                print (profit.shape)
                #print (np.mean(profit))
                print (mean1)
                profits1.append(mean1)
                #shapes1.append((profit.shape[0] / 2000000) - 1)
                profit = loadnpz('./recursive/0_1/profitData/Trans_' + oType + '_' + name + '.npz')
                #argsGood = loadnpz('./recursive/1/profitData/argsGood_' + oType + '_' + name + '.npz').astype(int)
                #print (np.mean(output[reverser[argsGood]]))
                mean2 = np.mean(profit)
                print (profit.shape)
                print (mean2)
                profits2.append(mean2)
                #shapes2.append((profit.shape[0] / 2000000) - 1)

                #plt.hist(output[reverser[argsGood]], bins=100, histtype='step')
                #plt.show()

            plt.plot(profits1)
            plt.plot(profits2)
            #plt.plot(shapes1)
            #plt.plot(shapes2)
            plt.show()

        quit()





        #'''
        output1 = loadnpz('./recursive/' + str(1) + '/samplesMix/prob_' + '0' + '.npz')[:, 1]
        #output5 = loadnpz('./recursive/' + str(1) + '/samplesMix5/prob_' + '0' + '.npz')[:, 1]
        #plt.hist(output1, bins=100, histtype='step', range=(0, 1))
        #plt.hist(output5, bins=100, histtype='step', range=(0, 1))
        #plt.show()
        #quit()
        order1 = loadnpz('./recursive/' + str(0) + '/neuralInputs/2/order_' + str(a) + '.npz')
        spread1 = loadnpz('./recursive/' + str(0) + '/neuralInputs/2/spread_' + str(a) + '.npz')
        allShares = loadnpz('./recursive/' + str(0) + '/neuralInputs/2/shares_' + str(a) + '.npz')
        profit = loadnpz('./recursive/' + str(0) + '/neuralInputs/2/profit_' + str(a) + '.npz')
        depth1 = np.sum(order1[:, 1, :], axis=1)

        #depth1 = depth1[spread1 > 5]
        #output1 = output1[spread1 > 5]
        #print (np.mean(output1[depth1 == 0]))
        #print (np.mean(output1))
        #quit()

        #plt.hist(output1, bins=100, histtype='step', density=True, range=(0, 1))
        output1 = output1[depth1 == 0]
        spread1 = spread1[depth1 == 0]
        order1 = order1[depth1 == 0]
        allShares = allShares[depth1 == 0]
        profit = profit[depth1 == 0]

        output1 = output1[spread1 > 5]
        order1 = order1[spread1 > 5]
        allShares = allShares[spread1 > 5]
        profit = profit[spread1 > 5]

        allShares3 = np.mean(allShares, axis=0)

        plt.imshow(allShares3)
        plt.show()
        quit()
        #spread2 = spread1[output1 > 0.0]
        #spread3 = spread1[output1 < 0.0]

        order1_1 = np.mean(order1[:, 0, :], axis=0)
        order1_2 = np.mean(order1[:, 0, :][spread1 > 5], axis=0)

        allShares1 = np.sum(allShares, axis=1)
        allShares2 = np.sum(allShares, axis=2)
        #plt.hist(spread2, bins=100, histtype='step', range=(0, 10))
        #plt.hist(spread3, bins=100, histtype='step', range=(0, 10))
        #plt.hist(spread1, bins=100, histtype='step', range=(0, 10))
        #plt.hist(output1, bins=100, histtype='step', range=(0, 1))

        print (allShares1.shape)

        allShares2 = allShares2[spread1 > 5] * 200
        midpoint = allShares2.shape[1] // 2
        allShares2 = allShares2[:, midpoint - 5:midpoint + 5]

        for a in range(1, 10):
            plt.plot(np.log(allShares2[a]+1))
            #plt.plot(allShares2[a])
        plt.show()
        quit()

        plt.plot(np.mean(allShares1, axis=0))
        plt.plot(np.mean(allShares1[spread1 > 5], axis=0))
        plt.show()

        plt.plot(np.mean(allShares2, axis=0))
        plt.plot(np.mean(allShares2[spread1 > 5], axis=0))
        plt.show()


        plt.plot(order1_1)
        plt.plot(order1_2)
        plt.show()
        quit()
        #'''


        profitVals = []
        for oType in ['B', 'S']:
            for name in names:##names[:7]:
                #output = loadnpz('./recursive/' + str(1) + '/samples/prob_' + oType + '_' + name + '.npz')#[:, 1]

                finalData = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')
                #finalData = loadnpz('./recursive/0/profitData/X_' + oType + '_' + name + '.npz')
                spread = loadnpz('./inputData/ITCH_LOB/spread/' + name + '.npz') / 100
                profit = loadnpz('./recursive/0/profitData/Trans_' + oType + '_' + name + '.npz')

                finalData = finalData[:profit.shape[0]]
                #finalData = finalData[output[:, 0] < profit.shape[0]]
                #output = output[output[:, 0] < profit.shape[0]]

                spread1 = spread[finalData[:, 0]]
                print (np.mean(profit[spread1 > 5]))
                #spread 5 is start of almost all positive
                profitVals.append(np.sum(profit[spread1 > 5]))

                #plt.hist(output[:, 2], bins=100, histtype='step', density=True)




                #
                #finalData = finalData[output[:, 0].astype(int)]
                #plt.show()

                #output = output[finalData[:, 2] == 0]

                '''
                finalData = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')
                profit = loadnpz('./recursive/0/profitData/Trans_' + oType + '_' + name + '.npz')

                spread = loadnpz('./inputData/ITCH_LOB/spread/' + name + '.npz') / 100

                output = output[output[:, 0] < profit.shape[0]]
                finalData = finalData[output[:, 0].astype(int)]
                profit = profit[output[:, 0].astype(int)]
                #depth = np.sum(order1[:, 1, :], axis=1)

                spread1 = spread[finalData[:, 0]]

                depth = finalData[:, 1]

                plt.hist(output[:, 2], bins=100, histtype='step', density=True)

                finalData = finalData[depth == 0]
                output = output[depth == 0]
                profit = profit[depth == 0]
                spread1 = spread1[depth == 0]

                plt.hist(output[:, 2], bins=100, histtype='step', density=True)
                plt.show()
                print ("A")
                print (np.mean(output[:, 2][spread1 >= 6]))
                print (np.mean(profit[spread1 >= 6]))
                '''

        plt.plot(profitVals)
        plt.show()
        quit()




        #quit()
        #'''
        output1 = loadnpz('./recursive/' + str(1) + '/samplesMix/prob_' + '0' + '.npz')[:, 1]
        spread1 = loadnpz('./recursive/' + str(0) + '/neuralInputs/2/spread_' + str(a) + '.npz') * 2
        profit1 = loadnpz('./recursive/' + str(0) + '/neuralInputs/2/profit_' + str(a) + '.npz')
        order1 = loadnpz('./recursive/' + str(0) + '/neuralInputs/2/order_' + str(a) + '.npz')
        #allShares = loadnpz('./recursive/' + str(0) + '/neuralInputs4/2/Shares_' + str(a) + '.npz')
        depth1 = np.sum(order1[:, 1, :], axis=1)

        spread1 = spread1[depth1 == 0]
        output1 = output1[depth1 == 0]
        profit1 = profit1[depth1 == 0]
        order1 = order1[depth1 == 0]
        #allShares = allShares[depth1 == 0]



        spreadMin = 3
        output1 = output1[spread1 > spreadMin]
        profit1 = profit1[spread1 > spreadMin]
        #order1 = order1[spread1 > spreadMin]
        #allShares = allShares[spread1 > spreadMin]
        spread1 = spread1[spread1 > spreadMin].astype(int)


        #print (np.mean(output1))
        #print (np.mean(profit1))
        #quit()



        #spreadMin = 6
        #output1 = output1[spread1 == spreadMin]
        #profit1 = profit1[spread1 == spreadMin]
        #order1 = order1[spread1 == spreadMin]
        #allShares = allShares[spread1 == spreadMin]
        #spread1 = spread1[spread1 == spreadMin].astype(int)

        #plt.hist(output1, bins=100)
        #plt.show()



        for a in range(3, 15):
            args = np.argwhere(spread1 == a)[:, 0]
            print ("A", a)
            print (np.mean(profit1[args]))
            print (np.mean(output1[args]))
            plt.hist(output1[args], bins=100, range=(-0.5, 0.2), histtype='step')
        plt.show()
        quit()


        allShares1 = np.sum(allShares, axis=1)
        allShares2 = np.sum(allShares, axis=2)

        #print (np.mean(profit1[output1 <]))


        #plt.hist(output1, bins=100)
        #plt.show()
        #quit()


        #quit()

        print ("A")
        print (np.mean(output1[output1 < -0.2]))
        print (np.mean(profit1[output1 < -0.2]))

        print ("A")
        print (np.mean(output1[output1 < -0.1]))
        print (np.mean(profit1[output1 < -0.1]))

        print ("A")
        print (np.mean(output1[output1 < -0.0]))
        print (np.mean(profit1[output1 < -0.0]))

        print ("A")
        print (np.mean(output1[output1 > 0.0]))
        print (np.mean(profit1[output1 > 0.0]))

        print ("A")
        print (np.mean(output1[output1 > 0.1]))
        print (np.mean(profit1[output1 > 0.1]))

        #quit()

        plt.plot(np.mean(allShares1, axis=0))
        plt.plot(np.mean(allShares1[output1 > 0.05], axis=0))
        plt.show()

        plt.plot(np.mean(allShares2, axis=0))
        plt.plot(np.mean(allShares2[output1 > 0.05], axis=0))
        plt.show()

        plt.plot(np.mean(order1[:, 0, :], axis=0))
        plt.plot(np.mean(order1[:, 0, :][output1 > 0.05], axis=0))
        plt.show()

        quit()
        '''



        profitVals = []
        for name in names[:12]:
            output = loadnpz('./recursive/' + str(1) + '/samples/prob_' + oType + '_' + name + '.npz')#[:, 1]
            finalData = loadnpz('./recursive/0/orderData/withCancels/X_' + oType + '_' + name + '.npz')
            profit = loadnpz('./recursive/0/profitData/Trans_' + oType + '_' + name + '.npz')



            spread = loadnpz('./inputData/ITCH_LOB/spread/' + name + '.npz') / 100
            #spread = np.cumsum(spread)
            #N = 1000
            #spread = (spread[N:] -  spread[:-N]) / N



            output = output[output[:, 0] < profit.shape[0]]
            finalData = finalData[output[:, 0].astype(int)]
            profit = profit[output[:, 0].astype(int)]
            #depth = np.sum(order1[:, 1, :], axis=1)

            spread1 = spread[finalData[:, 0]]

            plt.hist(output[:, 2][spread1 > 3], bins=100)
            plt.show()

            quit()

            depth = finalData[:, 1]

            finalData = finalData[depth == 0]
            output = output[depth == 0]
            profit = profit[depth == 0]

            print (np.mean(profit[output[:, 2] > 0.1]))
            #print (np.sum(profit[output[:, 2] > 0.1]))
            profitVals.append(np.sum(profit[output[:, 2] > 0.1]))

            spread[spread > 10] = 10
            spread[spread < 0] = 0

            #plt.plot(spread)
            #plt.hist(finalData[:, 0][output[:, 2] > 0.0], bins=1000)
            #plt.show()
            #print (output.shape)

        plt.plot(profitVals)
        plt.show()
        quit()


        output1 = loadnpz('./recursive/' + str(1) + '/samplesMix/prob_' + '0' + '.npz')[:, 1]
        output4 = loadnpz('./recursive/' + str(1) + '/samplesMix4/prob_' + '0' + '.npz')[:, 1]

        order1 = loadnpz('./recursive/' + str(0) + '/neuralInputs/2/order_' + str(a) + '.npz')
        depth1 = np.sum(order1[:, 1, :], axis=1)
        order4 = loadnpz('./recursive/' + str(0) + '/neuralInputs4/2/order_' + str(a) + '.npz')
        depth4 = np.sum(order4[:, 1, :], axis=1)

        spread1 = loadnpz('./recursive/' + str(0) + '/neuralInputs/2/spread_' + str(a) + '.npz')

        output1 = output1[depth1 == 0]
        output4 = output4[depth4 == 0]

        order1 = order1[depth1 == 0]
        spread1 = spread1[depth1 == 0]


        orderMean = np.mean(order1[:, 0, :][output1 > 0], axis=0)
        plt.plot(orderMean)
        plt.show()
        quit()
        #plt.hist(output4, histtype='step', bins=100)#, range=(0, 1))
        #plt.hist(output1, histtype='step', bins=100)#, range=(0, 1))

        #plt.hist(spread1[output1 > 0], bins=100, range=(0, 10))
        #plt.show()






        quit()

        size1 = 0
        size2 = 0
        for name in names:
            for oType in ['B', 'S']:
                print ("A")
                YData = loadnpz('./recursive/0/orderData/withCancels/Y_' + oType + '_' + name + '.npz')
                print (YData.shape)
                size1 += YData.shape[0]
                YData = loadnpz('./recursive/0/orderData4/withCancels/Y_' + oType + '_' + name + '.npz')
                print (YData.shape)
                size2 += YData.shape[0]

                print (size1 / size2)
                #YData = loadnpz('./recursive/0/orderData3/withCancels/Y_' + oType + '_' + name + '.npz')
                #print (YData.shape)
                #YData = loadnpz('./recursive/0/orderData2/withCancels/Y_' + oType + '_' + name + '.npz')
                #print (YData.shape)

        quit()

        output = loadnpz('./recursive/' + str(1) + '/samplesMix4/prob_' + '0' + '.npz')[:, 1]
        #outputFull2 = loadnpz('./recursive/' + str(1) + '/samplesMix2/prob_' + '0' + '.npz')[:, 1]
        #allShares = loadnpz('./recursive/' + str(0) + '/neuralInputs4/2/Shares_' + str(a) + '.npz')
        order = loadnpz('./recursive/' + str(0) + '/neuralInputs4/2/order_' + str(a) + '.npz')
        #profit = loadnpz('./recursive/' + str(0) + '/neuralInputs4/2/profit_' + str(a) + '.npz')
        spread = loadnpz('./recursive/' + str(0) + '/neuralInputs4/2/spread_' + str(a) + '.npz')
        #order2 = loadnpz('./recursive/' + str(0) + '/neuralInputs2/2/order_' + str(a) + '.npz')
        depth = np.sum(order[:, 1, :], axis=1)

        print (depth[depth == 0].shape[0] / depth.shape[0])
        print (depth[np.logical_and(depth == 0, output > 0.05)].shape[0] / depth[output > 0.05].shape[0])
        quit()
        #depth2 = np.sum(order2[:, 1, :], axis=1)
        output = output[depth == 0]
        order = order[depth == 0]
        spread = spread[depth == 0]
        allShares = allShares[depth == 0]
        profit = profit[depth == 0]





        order = order[output > 0.0]
        spread = spread[output > 0.0]
        allShares = allShares[output > 0.0]
        profit = profit[output > 0.0]
        output = output[output > 0.0]

        #plt.hist(spread, bins=100, range=(0, 10))
        #plt.show()
        #quit()



        plt.plot(np.mean(order[:, 0, :][spread > 2], axis=0))
        plt.plot(np.mean(order[:, 0, :][spread <= 2], axis=0))
        plt.show()

        #print (np.mean(profit[output > 0.05]))
        #print (np.mean(output[output > 0.05]))
        quit()


        orderMean = np.mean(order1[outputFull1 > 0.0], axis=0)


        plt.hist(spread[outputFull1 > 0.0], bins=100, range=(0, 10))
        plt.show()
        quit()

        plt.plot(orderMean[0])
        plt.show()
        plt.plot(orderMean[1])
        plt.show()

        #outputFull2 = outputFull2[depth2 == 0]

        #print (np.sum(outputFull1[outputFull1>0.0])/np.sum(outputFull2[outputFull2>0.0]))

        #plt.hist(outputFull1, histtype='step', bins=100, range=(0, 1))
        #plt.hist(outputFull2, histtype='step', bins=100, range=(0, 1))
        #plt.show()
        quit()



        '''
        for name in names:
            print ("A")
            profit = loadnpz('./recursive/' + str(0) + '/profitData/Trans_' + oType + '_' + name + '.npz')
            #print (np.mean(profit))
            profit2 = loadnpz('./recursive/' + str(0) + '/profitData2/Trans_' + oType + '_' + name + '.npz')
            print (np.mean(profit) - np.mean(profit2))
            #profit = loadnpz('./recursive/' + str(0) + '/profitData3/Trans_' + oType + '_' + name + '.npz')
            #print (np.mean(profit))

        quit()
        #'''


        for name in names[:5]:
            #outputFull = loadnpz('./recursive/' + str(1) + '/samples/prob_' + oType + '_' + name + '.npz')
            outputFull = loadnpz('./recursive/' + str(1) + '/samplesMix/prob_' + '0' + '.npz')[:, 1]
            outputFull2 = loadnpz('./recursive/' + str(1) + '/samplesMix2/prob_' + '0' + '.npz')[:, 1]

            print (np.sum(outputFull[outputFull>0.0])/np.sum(outputFull2[outputFull2>0.0]))
            #print (np.sum(outputFull2[outputFull2>0.0]))

            quit()
            #finalData = loadnpz('./recursive/' + str(0) + '/profitData2/X_' + oType + '_' + name + '.npz')
            #argsChoice = outputFull[:, 0]
            #outputFull = outputFull[argsChoice < finalData.shape[0]]
            #argsChoice = outputFull[:, 0].astype(int)

            #X = finalData[argsChoice, 0]
            #plt.hist(X, density=True, histtype='step', bins=100)
            #plt.hist(X[outputFull[:, 2] > 0.0], density=True, histtype='step', bins=100)
            #plt.show()


            plt.hist(outputFull[:, 1], histtype='step', bins=100, range=(0, 1))
            plt.hist(outputFull2[:, 1], histtype='step', bins=100, range=(0, 1))
            plt.show()
        quit()








        #outputFull = loadnpz('./recursive/1/samplesMix/prob_' + str(0) + '.npz')[:, 1]

        order = loadnpz('./recursive/' + str(0) + '/neuralInputs2/2/order_' + str(a) + '.npz')

        depth = np.sum(order[:, 1, :], axis=1)


        #profit = loadnpz('./recursive/' + str(0) + '/neuralInputs/2/profit_' + str(a) + '.npz')[:, 0]#[0::100]
        allShares = loadnpz('./recursive/' + str(0) + '/neuralInputs2/2/Shares_' + str(a) + '.npz')#[0::100]
        #spread = loadnpz('./recursive/' + str(0) + '/neuralInputs2/2/spread_' + str(a) + '.npz')
        profit = loadnpz('./recursive/' + str(0) + '/neuralInputs2/2/profit_' + str(0) + '.npz')[:, 0]
        outputFull = loadnpz('./recursive/' + str(1) + '/samplesMix2/prob_' + str(0) + '.npz')[:, 1]

        #plt.hist(outputFull, bins=100)
        #plt.show()
        #quit()

        plt.plot(np.mean(order[:, 0, :], axis=0))
        #plt.plot(np.sum(np.mean(allShares, axis=0), axis=1))


        #plt.show()

        profit = profit[depth == 0]
        outputFull = outputFull[depth == 0]
        order = order[depth == 0]
        allShares = allShares[depth == 0]



        print (np.mean(profit[outputFull > 0.1]))
        print (np.mean(outputFull[outputFull > 0.1]))
        plt.plot(np.mean(order[:, 0, :][outputFull > 0.1], axis=0))
        #plt.plot(np.sum(np.mean(allShares[outputFull > 0.1], axis=0), axis=1))
        plt.show()




        #print (depth[depth == 0].shape[0] / depth.shape[0])
        #depth = depth[outputFull > 0.1]
        #print (depth[depth == 0].shape[0] / depth.shape[0])


        #plt.hist(depth, bins=100, density=True, histtype='step')
        #plt.hist(depth[outputFull > 0.0], bins=100, histtype='step', density=True)
        #plt.show()
        quit()

        #print (scipy.stats.pearsonr(outputFull, profit))
        #quit()

        order1 = np.mean(order, axis=0)
        order2 = np.mean(order[outputFull > 0.0], axis=0)
        allShares1 = np.mean(allShares, axis=0)
        allShares2 = np.mean(allShares[outputFull > 0.0], axis=0)

        #print (np.mean(spread))
        #print (np.mean(spread[outputFull > 0.0]))


        quit()
        plt.imshow(allShares1)
        plt.show()
        plt.imshow(allShares2)
        plt.show()
        plt.plot(np.sum(allShares1, axis=0))
        plt.plot(np.sum(allShares2, axis=0))
        plt.show()
        plt.plot(np.sum(allShares1, axis=1))
        plt.plot(np.sum(allShares2, axis=1))
        plt.show()
        plt.plot(order1[0])
        plt.plot(order2[0])
        plt.show()
        plt.plot(order1[1])
        plt.plot(order2[1])
        plt.show()


        quit()


        #'''
        profitSum = 0
        for name in names[:]:
            profit = loadnpz('./recursive/' + str(0) + '/profitData/Trans_' + oType + '_' + name + '.npz')
            finalData = loadnpz('./recursive/' + str(0) + '/profitData/X_' + oType + '_' + name + '.npz')
            #profitSum += np.mean(profit[finalData[:, 1] == -1])
            profitSum += np.mean(profit)
            #print (np.mean(profit))
        print (profitSum / 15)
        quit()
        #finalData = loadnpz('./recursive/' + str(0) + '/profitData3/X_' + oType + '_' + name + '.npz')
        #times = finalData[:, 0]
        #timesUnique, times = np.unique(times, return_inverse=True)
        #YData = loadnpz('./recursive/' + str(0) + '/orderData/withCancels/Y_' + oType + '_' + name + '.npz')
        #outputFull = loadnpz('./recursive/' + str(1) + '/samples/prob_' + oType + '_' + name + '.npz')
        outputFull = loadnpz('./recursive/1/samplesMix/prob_' + str(0) + '.npz')[:, 1]

        #order = loadnpz('./recursive/' + str(0) + '/neuralInputs/2/order' + '_0' + '.npz')
        #profit = loadnpz('./recursive/' + str(0) + '/neuralInputs/2/profit' + '_0' + '.npz')[:, 0]

        order = order[:, 0, :]
        #val = np.matmul(order, np.arange(order.shape[1]).reshape((order.shape[1], 1)))[:, 0]

        #plt.plot(np.mean(order[outputFull > 0.0], axis=0))
        #plt.show()


        means1 = []
        means2 = []
        for a in range(0, order.shape[1]):
            mean1 = np.sum(order[:, a] * profit) / np.sum(order[:, a])
            means1.append(mean1)
            #mean2 = np.sum(order[:, a] * outputFull) / np.sum(order[:, a])
            #means2.append(mean2)
        plt.plot(means1)
        #plt.plot(means2)
        plt.show()
        quit()


        print (np.mean)
        #print (order[0])
        #quit()
        depth = np.sum(order[:, 1, :], axis=1)
        order = order[:, 0, :]

        print (np.mean(depth))
        print (np.mean(depth[outputFull[:, 1] > 0.0]))


        plt.plot(np.mean(order, axis=0))
        plt.plot(np.mean(order[outputFull[:, 1] > 0.0], axis=0))
        plt.show()

        print (np.mean(order, axis=0))
        print (np.mean(order[outputFull[:, 1] > 0.0], axis=0))
        #plt.hist(outputFull[:, 1], bins=100)
        #plt.show()
        quit()
        #'''

        finalData = loadnpz('./recursive/' + str(0) + '/profitData/X_' + oType + '_' + name + '.npz')
        output = loadnpz('./recursive/' + str(1) + '/outputPredictions/' + oType + '_' + name + '.npz')

        finalData = finalData[:output.shape[0]]
        #allShares = loadnpz('./inputData/ITCH_LOB/NearBestOrders/' + name + '.npz') / 200.0



        #plt.hist(output[finalData[:, 1] == -1, 1], bins=100, histtype='step', density=True)
        #plt.hist(output[:, 1], bins=100, histtype='step', density=True)
        #plt.show()
        #quit()


        #finalData2 = loadnpz('./recursive/' + str(0) + '/orderData/withCancels/X_' + 'B' + '_' + name + '.npz')
        #output2 = loadnpz('./recursive/' + str(1) + '/outputPredictions/' + 'B' + '_' + name + '.npz')[:, 1]
        #print ("A")
        X = finalData[:, 0]


        output = output[:, 1]
        X_argsort = np.argsort(X)
        output = output[X_argsort]
        X = X[X_argsort]
        finalData = finalData[X_argsort]


        #_, X = np.unique(X, return_inverse=True)
        #args = np.argwhere(np.isin(X2, np.array([404738, 404739, 404750, 404751, 404752, 404753]) ))
        #print (np.unique(X[args]))
        #quit()
        #[1084212 1084213 1084234 1084235 1084236 1084237]

        #args = 200000 + 50000 + np.arange(50000) + (X.shape[0] // 2)
        args = 200000 + 50000 + np.arange(20000) + (X.shape[0] // 2)
        #args = 200000 + 50000 + np.arange(200000) + (X.shape[0] // 2)

        X = X[args]
        output = output[args]
        relativePrice = finalData[args, 1]




        #min1, max1 = np.min(X), np.max(X)
        #X2 = finalData2[:, 0]
        #output2 = output2[X2 > min1]
        #X2 = X2[X2 > min1]
        #output2 = output2[X2 < max1]
        #X2 = X2[X2 < max1]

        #print (X2.shape)
        #print (output2.shape)


        #output = output[np.abs(X - 404750) < 50]
        #X = X[np.abs(X - 404750) < 50]

        #print (np.unique(X[output > 0.2]))

        #output = output[X > 404700]
        #output = output[X > 404700]
        #print (finalData.shape)
        #print (output.shape)
        #quit()

        #print (X.shape)
        #print (output.shape)

        #spread = loadnpz('./inputData/ITCH_LOB/spread/' + name + '.npz') / 200

        #plt.plot(np.unique(X), spread[np.unique(X)] - 1)

        plt.scatter(X, output)
        #plt.scatter(X, output)
        #plt.scatter(X[relativePrice == 17], output[relativePrice == 17])
        plt.show()


        quit()


















        #spread = loadnpz('./inputData/ITCH_LOB/spread/' + name + '.npz')
        #spread = spread[X[:, 0]] / 200.0

        #plt.hist(spread[finalData[args1, 0]], bins=100, histtype='step', density=True)
        #plt.hist(spread[finalData[args2, 0]], bins=100, histtype='step', density=True)
        #plt.show()

        buyAr = 1
        #'''
        relativePrice = finalData[args1][:, 1] + 1
        relativePrice = relativePrice * (1 - (buyAr*2))
        relativePrice = relativePrice - buyAr
        relativePrice = relativePrice + 18 + 1

        relativePrice = relativePrice + 22
        #print (np.max(relativePrice))
        #quit()

        relativePrice = relativePrice - 1

        relativePrice1 = relativePrice.astype(int)

        relativePrice2 = relativePrice[outputFull[:, 2] > 0.5].astype(int)

        #plt.hist(relativePrice, density=True, histtype='step', bins=100)
        #plt.hist(relativePrice[outputFull[:, 2] > 0.5], density=True, histtype='step', bins=100)
        #plt.show()
        #quit()
        #'''

        #allShares = loadnpz('./recursive/' + str(4) + '/neuralInputs/2/Shares_' + str(a) + '.npz')
        allShares = loadnpz('./inputData/ITCH_LOB/NearBestOrders/' + name + '.npz') / 200.0
        allShares = np.sum(allShares, axis=2)

        Xpos1 = finalData[args1, 0]
        Xpos2 = finalData[args2, 0]

        for a in range(0, 10):
            plt.plot(allShares[Xpos1[a]])
            ar = np.zeros(allShares.shape[1])
            ar[relativePrice1[a]] = 20
            plt.plot(ar)
            plt.show()

        #for a in range(0, 10):
        #    plt.plot(allShares[Xpos2[a]])
        #plt.show()
        quit()

        #plt.hist(Xpos1, density=True, histtype='step', bins=100)
        #plt.hist(Xpos2, density=True, histtype='step', bins=100)
        #plt.show()

        mean1 = np.mean(allShares[Xpos1], axis=0)
        mean2 = np.mean(allShares[Xpos2], axis=0)

        #plt.plot(np.mean(mean1, axis=0))
        #plt.plot(np.mean(mean2, axis=0))
        #plt.show()
        #plt.plot(np.mean(mean1, axis=1))
        #plt.plot(np.mean(mean2, axis=1))
        #plt.show()
        plt.imshow(mean2 - mean1)
        plt.show()
        #plt.imshow(mean2)
        #plt.show()

        #arFull = np.array([allShares[Xpos1, relativePrice-1:relativePrice+1], allShares[Xpos2, relativePrice[outputFull[:, 2] > 0]-1:relativePrice[outputFull[:, 2] > 0]+1]])
        #for b in range(0, 2):
        #    if b == 0:
        #        arVals = [allShares[Xpos1, relativePrice-1], allShares[Xpos1, relativePrice], allShares[Xpos1, relativePrice+1]]
        #    else:
        #        arVals = [allShares[Xpos2, relativePrice[outputFull[:, 2] > 0.25]-1], allShares[Xpos2, relativePrice[outputFull[:, 2] > 0.25]], allShares[Xpos2, relativePrice[outputFull[:, 2] > 0.25]+1]]
        #    means = [np.mean(np.abs(arVals[0] - arVals[1])), np.mean(np.abs(arVals[1] - arVals[2])), np.mean(np.abs( (arVals[1] * 2) - arVals[0] - arVals[2]))]
        #    plt.plot(means)
        #plt.show()
        quit()


        #mean1 = np.mean(allShares[Xpos1, relativePrice-1:relativePrice+1], axis=0)
        #mean2 = np.mean(allShares[Xpos2, relativePrice[outputFull[:, 2] > 0]-1:relativePrice[outputFull[:, 2] > 0]+1], axis=0)



        plt.plot(mean1)
        plt.plot(mean2)
        plt.show()
        plt.imshow(mean1)
        plt.show()
        plt.imshow(mean2)
        plt.show()
        quit()


        #print (finalData[args][0::1000][:10])
        #print (YData[args][0::1000][:10])

        for a in range(0, 3):
            print (a)
            min1, max1 = np.min(finalData[:, a]), np.max(finalData[:, a])
            plt.hist(finalData[:, a], density=True, histtype='step', bins=100, range=(min1, max1))
            plt.hist(finalData[args, a], density=True, histtype='step', bins=100, range=(min1, max1))
            plt.show()
            print ("B")
            min1, max1 = np.min(YData[:, a]), np.max(YData[:, a])
            plt.hist(YData[:, a], density=True, histtype='step', bins=100, range=(min1, max1))
            plt.hist(YData[args, a], density=True, histtype='step', bins=100, range=(min1, max1))
            plt.show()
        #print (scipy.stats.pearsonr(profit, depth))
        quit()


        '''
        model = orderProfitModel()
        #allShares = loadnpz('./temporary/tempNeuralInput/allShares.npz')#[:100000]
        #order = loadnpz('./temporary/tempNeuralInput/order.npz')
        #spread = loadnpz('./temporary/tempNeuralInput/spread.npz')
        #profit = loadnpz('./temporary/tempNeuralInput/profit.npz')

        endingName2 = '_0'
        loopNumber = 0

        allShares = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/2/Shares' + endingName2 + '.npz')[:500000]
        order = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/2/order' + endingName2 + '.npz')[:500000]
        spread = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/2/spread' + endingName2 + '.npz')[:500000]
        profit = loadnpz('./recursive/' + str(loopNumber) + '/neuralInputs/2/profit' + endingName2 + '.npz')[:500000]


        allShares = torch.tensor(allShares).float()
        order = torch.tensor(order).float()
        spread = torch.tensor(spread).float()
        profit = torch.tensor(profit).float()
        output = model(allShares, order, spread)
        quit()
        '''


def findOutputCancels(loopNumber, bidCutOff, askCutOff, filesRun=(0, 15)):

    names = giveFNames()
    for name in names[filesRun[0]:filesRun[1]]:
        print (name)
        for oType in ['B', 'S']:
            #print (oType)
            #output = loadnpz('./recursive/' + str(loopNumber) + '/outputPredictionsFast/' + oType + '_' + name + '.npz')

            finalData = loadnpz('./recursive/' + str(loopNumber) + '/orderData/withCancels/X_' + oType + '_' + name + '.npz')

            cancelsOutput = np.zeros(finalData.shape[0])
            cancelsOutput[np.logical_and(finalData[:, 1] == 0, finalData[:, 2] < 5)] = 1
            #print (np.mean(cancelsOutput))



            '''
            #plt.hist(output[:, 1], bins=100)
            #plt.show()
            cancels = np.argwhere(np.logical_and(output[:, 1] < bidCutOff, output[:, 0] != 2))[:, 0]

            cancelsOutput = np.zeros(output.shape[0]).astype(int)
            cancelsOutput[cancels] = 1

            print (np.mean(cancelsOutput))
            #quit()

            if False:
                output = loadnpz('./recursive/' + str(1) + '/outputPredictionsFast/' + oType + '_' + name + '.npz')
                cancels = np.argwhere(np.logical_and(output[:, 1] < bidCutOff, output[:, 0] != 2))[:, 0]
                cancelsOutput[cancels] = 1


                cancels2 = loadnpz('./recursive/' + str(0) + '/outputPredictions/' + oType + '_' + name + '.npz')
                cancels2 = np.argwhere(cancels2 < -1.8)[:, 0]
                cancelsOutput[cancels2] = 1
                #plt.hist(output[:, 1][cancels], bins=100)
                #plt.show()
                #quit()
            '''


            np.savez_compressed('./recursive/' + str(loopNumber+1) + '/outputCancels/' + oType + '_' + name + '.npz', cancelsOutput)

def saveProfitFull(loopNumber, filesRun=(0, 15)):

    names = giveFNames()
    for name in names:
        print (name)
        for oType in ['B', 'S']:
            profit1 = loadnpz('./recursive/' + str(loopNumber) + '/profitData/Trans_' + oType + '_' + name + '.npz')
            argsGood1 = loadnpz('./recursive/' + str(loopNumber) + '/profitData/argsGood_' + oType + '_' + name + '.npz').astype(int)
            argsGood0 = loadnpz('./recursive/' + str(0) + '/orderData/withCancels/argsGood_' + oType + '_' + name + '.npz').astype(int)
            profit2 = np.zeros((argsGood0.shape[0], 2))
            profit2[np.isin(argsGood0, argsGood1), 0] = profit1
            profit2[np.isin(argsGood0, argsGood1) == False, 1] = 1
            np.savez_compressed('./recursive/' + str(loopNumber) + '/profitData/profitFull_' + oType + '_' + name + '.npz', profit2)

def fullProfitProcessing(loopNumber):
    print ("Stage 0.1 --------------------------------------------------------------------------------------")
    findOutputCancels(loopNumber - 1, -2.5, -2.5, filesRun=filesRun) #-0.9
    print ("Stage 1 --------------------------------------------------------------------------------------")
    saveMajorCancelations(loopNumber, filesRun=filesRun)
    print ("Stage 2 --------------------------------------------------------------------------------------")
    saveMajorEmptyCancelations(loopNumber, filesRun=filesRun)
    print ("Stage 3 --------------------------------------------------------------------------------------")
    findLOBpossibleTrans(loopNumber, filesRun=filesRun)
    #print ("Stage 3.1 --------------------------------------------------------------------------------------")
    #findFullMinorExecute()
    print ("Stage 4 --------------------------------------------------------------------------------------")
    saveCancelationInput(loopNumber, filesRun=filesRun)
    print ("Stage 5 --------------------------------------------------------------------------------------")
    applyMinorCancelation(loopNumber, filesRun=filesRun)
    ###print ("Stage 6 --------------------------------------------------------------------------------------")
    ###findValidSells(loopNumber, filesRun=filesRun)
    print ("Stage 7 --------------------------------------------------------------------------------------")
    saveAllTransactionProfits(loopNumber, filesRun=filesRun)
    print ("Stage 8 --------------------------------------------------------------------------------------")
    findSubsetProfit(loopNumber, filesRun=filesRun)
    print ("Stage 9 --------------------------------------------------------------------------------------")
    doXsubset(loopNumber, filesRun=filesRun)
    print ("Stage 10 --------------------------------------------------------------------------------------")
    doXsubset3(loopNumber, filesRun=filesRun)

def saveExtraOrder():
    for c1 in range(0, 1):
        for c2 in range(0, 2):
            endingName2 = '_' + str(c1)


            #allShares = torch.load('./recursive/0/neuralInputs/2/Shares_g0' + endingName2 + '.pt')
            #order = torch.load('./recursive/0/neuralInputs/2/order_g0' + endingName2 + '.pt')

            allShares = loadnpz('./recursive/0/neuralInputs/2/Shares_g' + str(c2) + endingName2 + '.npz')
            order = loadnpz('./recursive/0/neuralInputs/2/order_g' + str(c2) + endingName2 + '.npz')

            #print (allShares.shape)

            order1 = order[:, 0, :]
            #print (order1.shape)
            order1 = np.argwhere(order1 == 1)[:, 1]
            midpoint1 = order.shape[2] // 2
            midpoint2 = allShares.shape[1] // 2
            order1 = order1 - midpoint1
            #print (np.unique(order1))
            order1 = (np.abs(order1) - 1) * np.sign(order1)
            order1 = order1 + midpoint2
            shares = allShares[np.arange(order1.shape[0]), order1, :]

            totalDepth = np.sum(shares, axis=1)
            #print (shares.shape)
            shares1 = np.copy(shares)
            shares1 = shares1[:, -1::-1]
            shares1 = np.cumsum(shares1, axis=1)

            depth = np.sum(order[:, 1, :], axis=1)
            #diff1 = shares1[:, -1] - depth
            #print (diff1[diff1 < 0].shape)
            #print (diff1[diff1 < 0][:10])



            for a in range(0, 10):
                shares1[:, a] = shares1[:, a] - depth
            shares1[shares1 < 0] = 0
            shares1[shares1 > 0] = 1
            shares1 = shares1[:, -1::-1]
            shares = shares * shares1

            print (shares[:10])

            order[:, 1, :10] = shares
            order[:, 1, 10] = depth
            order[:, 1, 11] = totalDepth

            order = torch.tensor(order).float()

            #np.savez_compressed('./recursive/0/neuralInputs/2/orderExtra_g1' + endingName2 + '.npz', order)
            torch.save(order, './recursive/0/neuralInputs/2/orderExtra_g' + str(c2) + endingName2 + '.pt')

    #np.savez_compressed('./recursive/0/neuralInputs/2/totalDepth_g1' + endingName2 + '.npz', totalDepth)
    #np.savez_compressed('./recursive/0/neuralInputs/2/sharesDepth_g1' + endingName2 + '.npz', shares)


    #print (order1[:10])

    quit()

def findChunkSubset():

    #ar = np.array([[4, 1], [3, 2], [1, 5]])
    #print (np.argsort(ar, axis=0))
    #quit()

    allInput = np.load('./temporary/input1.npy')
    allInput[:, -2] = allInput[:, -2] / 100

    normalMini = np.random.normal(size=allInput.size).reshape(allInput.shape)
    normalMini = normalMini * 1e-8

    normalSample = np.random.normal(size=allInput.shape[0])
    normalSample = np.sort(normalSample)

    allInput = allInput + normalMini

    allInput = np.argsort(allInput, axis=0)
    allInput = normalSample[allInput]




    #allInputMean = np.mean(allInput, axis=0)
    #allInputMean = allInputMean.repeat(allInput.shape[0]).reshape((allInputMean.shape[0], allInput.shape[0])).T
    #allInput = allInput - allInputMean
    #print (np.mean(np.abs(allInput), axis=0))



    quit()


    name = giveFNames()[0]
    oType = 'B'
    loopNumber = 0

    profitTrans = np.array([])
    profit = np.array([])
    finalData = np.zeros((0, 3))
    isBuyList = np.array([])

    for oType in ['B', 'S']:
        profit_0 = loadnpz('./recursive/' + str(loopNumber) + '/profitData/Trans_' + oType + '_' + name + '.npz') / 100
        profitTrans_0 = loadnpz('./recursive/' + str(loopNumber) + '/orderData/withCancels/Trans_' + oType + '_' + name + '.npz')
        argsGood0 = loadnpz('./recursive/' + str(loopNumber) + '/orderData/withCancels/argsGood_' + oType + '_' + name + '.npz')
        argsGood1 = loadnpz('./recursive/' + str(loopNumber) + '/profitData/argsGood_' + oType + '_' + name + '.npz')

        reverser = np.zeros(int(np.max(argsGood0)+1))
        reverser[argsGood0] = profitTrans_0
        profitTrans_0 = reverser[argsGood1]

        finalData_0 = loadnpz('./recursive/' + str(loopNumber) + '/profitData/X_' + oType + '_' + name + '.npz')

        profitTrans = np.concatenate((profitTrans, profitTrans_0))
        profit = np.concatenate((profit, profit_0))
        finalData = np.concatenate((finalData, finalData_0))

        buyList = np.zeros(finalData_0.shape[0])
        if oType == 'S':
            buyList = buyList + 1
        #print (buyList.shape)
        isBuyList = np.concatenate((isBuyList, buyList))



    X_argsort = np.argsort(finalData[:, 0])
    profit = profit[X_argsort]
    finalData = finalData[X_argsort]
    profitTrans = profitTrans[X_argsort]
    isBuyList = isBuyList[X_argsort]

    #N = 100000
    N = 2000000
    #N = 25637855
    #print (profitTrans.shape)
    #TODO Add milisecond rounding
    timeEnds = np.arange(profitTrans.shape[0])[2*N:]
    timeEnds = (timeEnds // N) * N
    timeEnds = finalData[timeEnds, 0]

    profitTrans = profitTrans[:-2*N]
    profit = profit[:-2*N]
    finalData = finalData[:-2*N]
    isBuyList = isBuyList[:-2*N]

    #plt.plot(profitTrans)
    #plt.plot(timeEnds)
    #plt.show()
    #quit()

    #validOnes = np.argwhere((profitTrans - timeEnds) < 0)[:, 0]

    spread = loadnpz('./inputData/ITCH_LOB/spread/' + name + '.npz') / 200.0

    allShares = loadnpz('./inputData/ITCH_LOB/NearBestOrders/' + name + '.npz') / 200.0 #[:, :, :, 1] / 200.0
    midpoint = allShares.shape[1] // 2
    allShares = allShares[:, midpoint-20:midpoint+20]
    allShares1 = np.sum(allShares, axis=2)

    print (np.min(finalData[:, 1]))
    print (np.max(finalData[:, 1]))


    for a in range(0, profitTrans.shape[0] // N):
        print ("A")
        argSubset = np.arange(N) + (a * N)
        validOnes = np.argwhere((profitTrans[argSubset] - timeEnds[argSubset]) < 0)[:, 0] + N
        X = finalData[:, 0][validOnes].astype(int)
        relPrice = finalData[:, 1][validOnes]

        size1 = validOnes.shape[0]

        shares1 = allShares1[X]
        isBuy1 = isBuyList[validOnes]
        isBuy1 = 1 - (isBuy1 * 2)
        relPrice = ((relPrice * 2) + 1) * isBuy1
        relPrice = (relPrice + 35) // 2
        relPrice = relPrice.astype(int)
        order = np.zeros((size1, 36))
        order[np.arange(size1).astype(int), relPrice] = 1

        print ("B")

        spread1 = spread[X].reshape((size1, 1))

        depth1 = finalData[:, 2][validOnes].reshape((size1, 1)) / 100

        print (order.shape, shares1.shape, depth1.shape, spread1.shape)
        allInput = np.concatenate((order, shares1, depth1, spread1), axis=1)

        np.save('./temporary/input1.npy', allInput)

        print ("C")
        print (allInput.shape)








        quit()
    print (profitTrans.shape)
    print (validOnes.shape)

    #timeEnds = profitTrans[0::N][1:]
    #endingArray = np.arange(timeEnds.shape[0] + 1).repeat(N)[:profitTrans.shape[0]]

    quit()





    waitTime = profitTrans - finalData[:, 0]



    print (np.max(profitTrans))
    plt.hist(waitTime, bins=100)
    plt.show()
    quit()




    #'''
    X = loadnpz('./recursive/' + str(loopNumber) + '/profitData/X_' + oType + '_' + name + '.npz')[:, 0]
    X_argsort = np.argsort(X)

    #argsKeep = np.argwhere
    profit1 = np.cumsum(profit[X_argsort])
    L = 10000
    profit1 = (profit1[L:] - profit1[:-L]) / L
    profit[X_argsort[:-L]] = profit[X_argsort[:-L]] - profit1
    profit[X_argsort[-L:]] = profit[X_argsort[-L:]] - profit1[-1]
    #profit[X_argsort[:-L]] = profit1
    #profit[X_argsort[-L:]] = profit1[-1]
    #'''

    #quit()
    argsGood = loadnpz('./recursive/' + str(loopNumber) + '/profitData/argsGood_' + oType + '_' + name + '.npz')


def saveSellPrice():
    oType = 'B'
    loopNumber = 0
    names= giveFNames()[60:]#[:5]
    #name = giveFNames()[60+1]

    spreadFull = np.array([])
    allSharesFull = np.zeros((0, 40, 10))
    diffFull = np.array([])
    XFull = np.array([])

    #'''
    size2 = 0
    for a in range(0, len(names)):
        name = names[a]
        print (a)
        size1 = np.load('./recursive/' + str(loopNumber) + '/transProfits/final/transSalePrices_' + oType + '_' + name + '.npy').shape[0]
        size2 += size1

    argFull = np.zeros(size2)
    count1 = 0

    for a in range(0, len(names)):
        name = names[a]
        print (a)
        size1 = np.load('./recursive/' + str(loopNumber) + '/transProfits/final/transSalePrices_' + oType + '_' + name + '.npy').shape[0]
        #[100:-100]
        argFull[count1:count1+500] = 1
        argFull[count1+size1-500:count1+size1] = 1
        print (np.mean(argFull))
        #quit()
        count1 += size1

    print (argFull.shape)
    np.savez_compressed('./recursive/0/directTrans/args_' + oType + '.npz', argFull)
    quit()
    #'''

    for a in range(0, len(names)):
        name = names[a]
        print (a)
        transProfits = np.load('./recursive/' + str(loopNumber) + '/transProfits/final/transSalePrices_' + oType + '_' + name + '.npy')#[100:-100]

        X = transProfits[:, 0].astype(int) #+ 10#- 1#+ 1
        transProfits = transProfits[:, 1]

        data = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz')
        _, milisecondArgs = np.unique(data[:, 0][-1::-1], return_index=True)
        milisecondArgs = data.shape[0] - 1 - milisecondArgs
        milisecondArgs = np.sort(milisecondArgs)

        roundMili = np.zeros(data.shape[0])
        roundMili[milisecondArgs] = 1
        roundMili = np.cumsum(roundMili)
        roundMili[milisecondArgs] = roundMili[milisecondArgs] - 1
        roundMili[milisecondArgs[-1]+1:] = roundMili[milisecondArgs[-1]+1:] - 1
        roundMili = milisecondArgs[roundMili.astype(int)]

        X = roundMili[X]

        spread = loadnpz('./inputData/ITCH_LOB/spread/' + name + '.npz') / 200.0
        spread = spread[X]

        if oType == 'B':
            bestPrice0 = loadnpz('./inputData/ITCH_LOB/bestPrice/Bid0_' + name + '.npz')
        else:
            bestPrice0 = loadnpz('./inputData/ITCH_LOB/bestPrice/Ask0_' + name + '.npz')

        diff = transProfits - bestPrice0[X]

        print (np.mean(diff))
        #quit()
        #diff = diff[100:-100]
        #plt.plot(diff)
        #plt.show()
        #quit()

        allShares = loadnpz('./inputData/ITCH_LOB/NearBestOrders/' + name + '.npz') / 200.0 #[:, :, :, 1] / 200.0
        midpoint = allShares.shape[1] // 2
        allShares = allShares[:, midpoint-20:midpoint+20]
        #allShares = np.sum(allShares, axis=2)
        allShares = allShares[X]

        spreadFull = np.concatenate((spreadFull, spread))
        print (spreadFull.shape)
        allSharesFull = np.concatenate((allSharesFull, allShares))
        diffFull = np.concatenate((diffFull, diff))

        XFull = np.concatenate((XFull, X))

        np.savez_compressed('./recursive/0/directTrans/X_' + oType + '.npz', XFull)
        np.savez_compressed('./recursive/0/directTrans/spread_' + oType + '.npz', spreadFull)
        np.savez_compressed('./recursive/0/directTrans/shares_' + oType + '.npz', allSharesFull)
        np.savez_compressed('./recursive/0/directTrans/diff_' + oType + '.npz', diffFull)




        #print (scipy.stats.pearsonr(spread, diff))
        #for b in range(19-3, 19+3):
        #    print (scipy.stats.pearsonr(allShares[:, a], diff))

def findSellPrice():

    '''
    names= giveFNames()[60:]
    name = names[0]
    data = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz')
    _, milisecondArgs = np.unique(data[:, 0][-1::-1], return_index=True)
    print (milisecondArgs.shape)
    quit()
    '''

    oType = 'S'

    for oType in ['B', 'S']:
        diff = loadnpz('./recursive/0/directTrans/diff_' + oType + '_c.npz') / 100
        spread = loadnpz('./recursive/0/directTrans/spread_' + oType + '_c.npz')[:diff.shape[0]]
        allShares = loadnpz('./recursive/0/directTrans/shares_' + oType + '_c.npz')[:diff.shape[0]] / 10
        argsGood = loadnpz('./recursive/0/directTrans/args_' + oType + '.npz')[:diff.shape[0]]
        position = loadnpz('./recursive/0/directTrans/X_' + oType + '_c.npz')[:diff.shape[0]]

        #print (np.mean(diff))
        #quit()

        if oType == 'B':
            diff = (spread * 2) - diff
        else:
            diff = (spread * 2) + diff

        argsGood[:300] = 1
        argsGood = argsGood[:diff.shape[0]]

        #plt.plot(argsGood)
        #plt.show()
        #quit()

        diff = diff[argsGood == 0]
        spread = spread[argsGood == 0]
        allShares = allShares[argsGood == 0]

        position = position[argsGood == 0]

        allShares = np.sum(allShares, axis=2)
        midpoint = allShares.shape[1] // 2
        allShares = allShares[:, midpoint-10:midpoint+10]

        X = np.zeros((allShares.shape[0], allShares.shape[1]+1))
        X[:, 1:] = allShares
        X[:, 0] = spread

        from sklearn.linear_model import LinearRegression
        from sklearn.neural_network import MLPRegressor
        print ("A")
        reg = LinearRegression().fit(X, diff)
        #reg = MLPRegressor(hidden_layer_sizes=(10,)).fit(X, diff)








        spread = loadnpz('./recursive/0/directTrans/spread_' + 'B' + '_c.npz')
        allShares = loadnpz('./recursive/0/directTrans/shares_' + 'B' + '_c.npz')[:spread.shape[0]] / 10
        argsGood = loadnpz('./recursive/0/directTrans/args_' + 'B' + '.npz')[:spread.shape[0]]
        position = loadnpz('./recursive/0/directTrans/X_' + 'B' + '_c.npz')[:spread.shape[0]]


        argsGood[:300] = 1
        #argsGood = argsGood[:diff.shape[0]]

        #plt.plot(argsGood)
        #plt.show()
        #quit()

        print (spread.shape)
        print (argsGood.shape)

        #diff = diff[argsGood == 0]
        spread = spread[argsGood == 0]
        allShares = allShares[argsGood == 0]

        position = position[argsGood == 0]

        allShares = np.sum(allShares, axis=2)
        midpoint = allShares.shape[1] // 2
        allShares = allShares[:, midpoint-10:midpoint+10]

        X = np.zeros((allShares.shape[0], allShares.shape[1]+1))
        X[:, 1:] = allShares
        X[:, 0] = spread







        predDiff = reg.predict(X[:500])#[:100]

        names= giveFNames()[60:]
        name = names[0]
        bestBid = loadnpz('./inputData/ITCH_LOB/bestPrice/Bid0_' + name + '.npz') / 100
        bestAsk = loadnpz('./inputData/ITCH_LOB/bestPrice/Ask0_' + name + '.npz') / 100
        position = position[:500].astype(int)
        bestBid = bestBid[position]
        bestAsk = bestAsk[position]

        if oType == 'B':
            plt.plot(position, bestBid, c='r')
            plt.plot(position, bestAsk, c='b')
            plt.plot(position, bestAsk-predDiff)
        else:
            plt.plot(position, bestBid, c='r')
            plt.plot(position, bestAsk, c='b')
            plt.plot(position, bestBid+predDiff)
    plt.show()
    quit()

    print (scipy.stats.pearsonr(spread, diff))
    for a in range(0, 20):
        print (scipy.stats.pearsonr(allShares[:, a], diff))
    quit()

    print (np.mean(np.abs(diff)))
    print (np.mean(np.abs(diff[np.abs(diff) < 10000])))

    print (np.mean(diff))

    #plt.plot(diff[argsGood])
    #plt.show()
    #quit()

    plt.hist(diff, bins=100, range=(-1000, 1000))
    plt.show()


class boundModel(nn.Module):
    def __init__(self):
        super(boundModel, self).__init__()
        self.nonlin = torch.tanh
        #self.nonlin = torch.relu

        self.lin1 = nn.Linear(21, 10)
        self.lin2 = nn.Linear(10, 10)
        self.lin3 = nn.Linear(10, 1) #39 or 2


    def forward(self, LOB, spread):
        LOB = torch.sum(LOB, axis=2)
        midpoint = LOB.shape[1]
        LOB = LOB[:, midpoint-20:midpoint+20]

        LOB[:, :5] = 0
        LOB[:, -5:] = 0

        X = torch.zeros((LOB.shape[0], LOB.shape[1] + 1))
        X[:, 1:] = LOB
        X[:, 0] = spread


        X = self.lin1(X)
        X = self.nonlin(X)
        X = self.lin2(X)
        X = self.nonlin(X)
        X = self.lin3(X)
        X = X.reshape((X.shape[0],))
        return X

def trainBounds():
    print ("Training Bound")

    oType = 'B'

    diff = loadnpz('./recursive/0/directTrans/diff_' + oType + '_c.npz') / 100
    spread = loadnpz('./recursive/0/directTrans/spread_' + oType + '_c.npz')[0:diff.shape[0]]
    allShares = loadnpz('./recursive/0/directTrans/shares_' + oType + '_c.npz')[0:diff.shape[0]] / 10
    argsGood = loadnpz('./recursive/0/directTrans/args_' + oType + '.npz')[0:diff.shape[0]]
    #diff = diff[300:]

    #plt.plot(diff[argsGood==1])
    #plt.plot(diff[argsGood==0])
    #plt.show()
    #quit()

    #position = loadnpz('./recursive/0/directTrans/X_' + oType + '_c.npz')[:diff.shape[0]]

    '''
    print (argsGood[0], argsGood[-1])
    change1 = (argsGood[1:] - argsGood[:-1]).astype(int)
    argCut1 = np.argwhere(change1 == -1)[:, 0]
    argCut2 = np.argwhere(change1 == 1)[:, 0]
    print (argCut1.shape)
    print (argCut2.shape)
    quit()
    '''

    argsTrain = np.array([])
    argsTest = np.array([])
    size1 = diff.shape[0] // 27

    for a in range(0, 27):
        argsOne = np.arange(size1) + (size1 * a)
        if a % 3 == 2:
            argsTest = np.concatenate((argsTest, argsOne))
        else:
            argsTrain = np.concatenate((argsTrain, argsOne))

    argsTest = argsTest.astype(int)
    argsTrain = argsTrain.astype(int)

    diff_T = diff[argsTest]
    spread_T = spread[argsTest]
    allShares_T = allShares[argsTest]

    diff = diff[argsTrain]
    spread = spread[argsTrain]
    allShares = allShares[argsTrain]
    #diff = diff[argsGood == 0]
    #spread = spread[argsGood == 0]
    #allShares = allShares[argsGood == 0]

    #trainCut = (diff.shape[0] * 2) // 3

    #diff_T = diff[trainCut:]
    #spread_T = spread[trainCut:]
    #allShares_T = allShares[trainCut:]

    args_T = np.random.permutation(allShares_T.shape[0])
    allShares_T = allShares_T[args_T][:1000000]
    spread_T = spread_T[args_T][:1000000]
    diff_T = diff_T[args_T][:1000000]



    args = np.random.permutation(allShares.shape[0])
    allShares = allShares[args]
    spread = spread[args]
    diff = diff[args]



    model = boundModel()

    learningRate = 1e-3
    optimizer = torch.optim.SGD(model.parameters(), lr = learningRate )# * 0.01


    M = 200
    #N = 10000
    N = 10000
    #M = 10
    #N = 10000
    losses = []
    lossCor = []

    args = np.random.permutation(allShares.shape[0])
    allShares = allShares[args]
    spread = spread[args]
    diff = diff[args]

    allShares = torch.tensor(allShares).float()
    spread = torch.tensor(spread).float()
    diff = torch.tensor(diff).float()

    allShares_T = torch.tensor(allShares_T).float()
    spread_T = torch.tensor(spread_T).float()
    diff_T = torch.tensor(diff_T).float()

    outputMean = 0.0
    count1 = 0

    iterN = 200

    for iter in range(0, iterN):
        print (iter)
        for a in range(0, diff.shape[0] // N):
            count1 += 1
            #print (a)
            #print ('a', a)
            allShares1 = allShares[a*N:(a+1)*N]
            spread1 = spread[a*N:(a+1)*N]
            diff1 = diff[a*N:(a+1)*N]

            output = model(allShares1, spread1)

            loss = torch.mean((output - diff1) ** 2.0)
            losses.append(loss.data.numpy())
            lossCor.append(scipy.stats.pearsonr(output.data.numpy(), diff1.data.numpy())[0] )

            #print ("A")
            #print (scipy.stats.pearsonr(output.data.numpy(), spread1.data.numpy()))
            #print (scipy.stats.pearsonr(spread1.data.numpy(), profit1.data.numpy()))

            if count1 % M == 0:
                lossGroup = np.mean(np.array(losses)[-M:])
                lossCorGroup = np.mean(np.array(lossCor)[-M:])
                #print (  a   )
                #print ((a // M) + ( c * (diff.shape[0] //  (N * M)) ))
                #print (scipy.stats.pearsonr(output.data.numpy(), profit1.data.numpy()))
                print (lossGroup)
                print (lossCorGroup)
                output_T = model(allShares_T, spread_T)
                print (torch.mean((output_T - diff_T) ** 2.0).data.numpy())
                print (scipy.stats.pearsonr(output_T.data.numpy(), diff_T.data.numpy()))


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


trainBounds()
quit()

saveSellPrice()
quit()

findSellPrice()
quit()
#findChunkSubset()
#quit()
#sellerAnalysis3()
#quit()


names = giveFNames()
#print (names[-15:])
#quit()
bidCutOff = -0.7
askCutOff = -0.7
#filesRun = (9, 15)
#filesRun = (0, 8)
#filesRun = (11, 15)
filesRun = (0, 52)
loopNumber = 1

#print ("formLOBrecentShares()--------------------------------------------------------------------------------------")
#formLOBrecentShares()
#print ("saveXCompressed()--------------------------------------------------------------------------------------")
#saveXCompressed()


#print ("Stage 0 --------------------------------------------------------------------------------------")
#savePredictions2(loopNumber)
#quit()
'''
print ("Stage 0.1 --------------------------------------------------------------------------------------")
findOutputCancels(loopNumber - 1, -5.0, -5.0, filesRun=filesRun) #-0.9
print ("Stage 1 --------------------------------------------------------------------------------------")
saveMajorCancelations(loopNumber, filesRun=filesRun)
print ("Stage 2 --------------------------------------------------------------------------------------")
saveMajorEmptyCancelations(loopNumber, filesRun=filesRun)
print ("Stage 3 --------------------------------------------------------------------------------------")
findLOBpossibleTrans(loopNumber, filesRun=filesRun)
#print ("Stage 3.1 --------------------------------------------------------------------------------------")
#findFullMinorExecute()
print ("Stage 4 --------------------------------------------------------------------------------------")
saveCancelationInput(loopNumber, filesRun=filesRun)
print ("Stage 5 --------------------------------------------------------------------------------------")
applyMinorCancelation(loopNumber, filesRun=filesRun)
###print ("Stage 6 --------------------------------------------------------------------------------------")
###findValidSells(loopNumber, filesRun=filesRun)
print ("Stage 7 --------------------------------------------------------------------------------------")
saveAllTransactionProfits(loopNumber, filesRun=filesRun)
print ("Stage 8 --------------------------------------------------------------------------------------")
findSubsetProfit(loopNumber, filesRun=filesRun)
print ("Stage 9 --------------------------------------------------------------------------------------")
doXsubset(loopNumber, filesRun=filesRun, doX=False)
#'''
print ("Stage 10 --------------------------------------------------------------------------------------")
doXsubset3(loopNumber, filesRun=(0, 52), doX=False)
#print ("Stage 11 --------------------------------------------------------------------------------------")
#autoTrainSeller_v2(loopNumber, bidCutOff, askCutOff, useCancel=False, predictProb=True)
#autoTrainSeller_v2(loopNumber, bidCutOff, askCutOff, useCancel=False, predictProb=False)
#quit()



print ("Stage 12.1 --------------------------------------------------------------------------------------")
autoTrainSeller_v3(loopNumber, bidCutOff, askCutOff)
quit()

#autoTrainSeller_compare(0)
#quit()


print ("Stage 12.2 --------------------------------------------------------------------------------------")

#savePredictions3(loopNumber, filesRun)
#quit()
savePredictions4(loopNumber, filesRun)
#print ("Stage 12.3 --------------------------------------------------------------------------------------")
quit()
savePredictions2(loopNumber)
quit()


valCut = trainCancelPredictor(loopNumber)
print ("R")
print (valCut)
#valCut = 0.02062
print ("Stage 12.4 --------------------------------------------------------------------------------------")
#print ("Stage 12 --------------------------------------------------------------------------------------")
#print ("Stage 12.5 --------------------------------------------------------------------------------------")
#quit()
savePredictions2_fast(loopNumber, valCut)

fullProfitProcessing(loopNumber+1)


quit()
#savePredictions3(loopNumber, filesRun)
#quit()
#print ("Stage 13 --------------------------------------------------------------------------------------")
savePredictions2(loopNumber)
print ("Stage 14 --------------------------------------------------------------------------------------")
quit()


#TODO Fix inconstent relativePrice


#findFullMinorExecute

#trainCancelPredictor


#Notes:

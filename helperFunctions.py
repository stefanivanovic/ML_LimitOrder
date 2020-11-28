
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
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

def easySubsetArgwhere(A, B):
    argsInB = np.argwhere(np.isin(A, B))[:, 0]
    A = A[argsInB]
    A_argsort = np.argsort(A)
    A = A[A_argsort]
    _, indicesStart = np.unique(A, return_index=True)
    indicesEnd = np.copy(indicesStart)
    indicesEnd[:-1] = indicesEnd[1:]
    indicesEnd[-1] = A.shape[0]

    A_unique = np.unique(A)

    places = []

    c = 0
    for a in range(0, B.shape[0]):
        if B[a] in A_unique:
            place = argsInB[A_argsort[indicesStart[c]:indicesEnd[c]]]
            places.append(place)
            c += 1
        else:
            places.append(np.array([]))

    #for a in range(0, indicesStart.shape[0]):
    #    place = argsInB[A_argsort[indicesStart[a]:indicesEnd[a]]]
    #    places.append(place)

    return places

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


def multiArgSort(ar1, ar2):
    _, ar1_ = np.unique(ar1, return_inverse=True)
    ar2u, ar2_ = np.unique(ar2, return_inverse=True)
    ar3 = (ar1_ * ar2u.shape[0]) + ar2_
    argsort1 = np.argsort(ar3)
    return argsort1

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


def leakyrelu(x):
    x = (torch.relu(x) * 0.9) + (x * 0.1)
    x = x * 2.0
    return x



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


def doLinearRegression(X, Y, returnCor=False):
    from sklearn.linear_model import LinearRegression
    if len(X.shape) == 1:
        X = X.reshape((-1, 1))
    reg = LinearRegression().fit(X, Y)
    if returnCor:
        cor = scipy.stats.pearsonr(reg.predict(X), Y)
        return reg, cor
    else:
        return reg

def groupingAverage1(int1, M, Y, returnN=False):
    #print (np.unique(int1)[:M])
    lists1 = easySubsetArgwhere(int1, np.arange(M))
    means1 = []
    Ns = []
    for a in range(0, M): #horizontal
        argsGood1 = lists1[a]
        if argsGood1.shape[0] == 0:
            mean1 = 0
            Ns.append(0)
        else:
            #print (argsGood1)
            mean1 = np.mean(Y[argsGood1])
            Ns.append(argsGood1.shape[0])
        means1.append(mean1)

    if returnN:
        return means1, Ns
    else:
        return means1


def groupingApplication1(int1, M, X, func1):
    lists1 = easySubsetArgwhere(int1, np.arange(M))
    means1 = []
    for a in range(0, M): #horizontal
        argsGood1 = lists1[a]
        if argsGood1.shape[0] == 0:
            mean1 = 0
        else:
            mean1 = func1(X[argsGood1])
        means1.append(mean1)
    return means1



'''
def groupingApplication1(int1, M, X, func1):
    lists1 = easySubsetArgwhere(int1, np.arange(M))
    means1 = []
    for a in range(0, M): #horizontal
        argsGood1 = lists1[a]
        mean1 = func1(X[argsGood1])
        means1.append(mean1)
    return means1
'''
def groupingNumber1(int1, M, X):
    def func1(Y):
        return Y.shape[0]
    return groupingApplication1(int1, M, X, func1)

def groupingSum1(int1, M, X):
    def func1(Y):
        return np.sum(Y)
    return groupingApplication1(int1, M, X, func1)


def groupingApplication2(int1, int2, M1, M2, X, func1):
    lists1 = easySubsetArgwhere(int1, np.arange(M1))
    img = np.zeros((M1, M2))
    for a in range(0, M1): #horizontal
        argsGood1 = lists1[a]
        lists2 = easySubsetArgwhere(int2[argsGood1], np.arange(M2))
        for b in range(0, M2):
            argsGood2 = lists2[b]
            argsGood2 = argsGood1[argsGood2]
            mean1 = func1(X[argsGood2])
            img[a, b] = mean1
    return img

def groupingAverage2(int1, int2, M1, M2, Y):
    def func1(C):
        return np.mean(C)

    img = groupingApplication2(int1, int2, M1, M2, Y, func1)

    return img

def groupingNumber2(int1, int2, M1, M2, Y):
    def func1(C):
        return C.shape[0]

    img = groupingApplication2(int1, int2, M1, M2, Y, func1)

    return img



def PlotHistReasonCut(X):
    X = np.sort(X)
    min1, max1 = X[X.shape[0]//100], X[-X.shape[0]//100]
    plt.hist(X, bins=100, range=(min1, max1), histype='step')

#Data Loading In

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

from helperFunctions import *


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






def individualLOB(isBuy):
    import copy
    def saveComponent(nameNum):
        names = giveFNames()
        print (nameNum)
        name = names[nameNum]
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
        #print (data.shape[0] // 10000)
        for a in range(0, data.shape[0]):
            #if a ==  149:
            #    print ("Hi")

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

            #if a ==  149:
            #    print (price_arg)
            #    print (LOBcurrent[price_arg])
            #    quit()
            LOBhistory[price_arg][0].append(a)
            #LOBhistory[price_arg][1].append(copy.copy(LOBcurrent[price_arg]))
            #LOBhistory[price_arg][1].append(np.copy(LOBcurrent[price_arg]))
            LOBhistory[price_arg][1].append(copy.deepcopy(LOBcurrent[price_arg]))
            #if a % 100000 == 0:
            #    print (a//100000)
            #    print (lenSum / 100000)
            #    lenSum = 0

        #print (LOBhistory[1361][0][:10])
        #print (LOBhistory[1361][1][:10])
        #quit()
        if True:
            if isBuy:
                #np.save('./resultData/temporary/LOB_Buy.npy', LOBhistory)
                np.savez_compressed('./inputData/ITCH_LOB/BuyLOB/' + name + '.npz', LOBhistory)
            else:
                #np.save('./resultData/temporary/LOB_Sell.npy', LOBhistory)
                np.savez_compressed('./inputData/ITCH_LOB/SellLOB/' + name + '.npz', LOBhistory)

    #70:
    multiProcessor(saveComponent, 50, 111, 4, doPrint = True)

#individualLOB(False)
#quit()

def shareLOB(isBuy):
    def saveComponent(nameNum):
        names = giveFNames()
        print (nameNum)
        name = names[nameNum]

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

    multiProcessor(saveComponent, 50, 111, 4, doPrint = True)

def constructBestPrice(isBuy):
    def saveComponent(nameNum):
        names = giveFNames()
        print (nameNum)
        name = names[nameNum]
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
    multiProcessor(saveComponent, 50, 111, 4, doPrint = True)

def saveNearByShare(isBuy):
    def saveComponent(nameNum):
        names = giveFNames()
        print (nameNum)
        name = names[nameNum]
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
            #if a % 100 == 0:
            #    print (a)
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

    multiProcessor(saveComponent, 50, 111, 4, doPrint = True)

def saveNearByShareFull():
    def saveComponent(nameNum):
        names = giveFNames()
        print (nameNum)
        name = names[nameNum]
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

    saveComponent(0)
    #multiProcessor(saveComponent, 50, 111, 4, doPrint = True)

#saveNearByShareFull()
#quit()

def saveLOBhistWithStarts():
    def saveComponent(nameNum):
        names = giveFNames()
        print (nameNum)
        name = names[nameNum]
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

    multiProcessor(saveComponent, 50, 111, 4, doPrint = True)


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

def saveNearShareSubset():

    def saveComponent(nameNum):
        names = giveFNames()
        #names = names[59:]
        name = names[nameNum]
        allShares = loadnpz('./inputData/ITCH_LOB/nearSharesAll/' + name + '.npz')
        print (allShares.shape)
        allShares = loadnpz('./inputData/ITCH_LOB/NearBestOrders/' + name + '.npz')
        print (allShares.shape)
        quit()

        midpoint = allShares.shape[1] // 2
        allShares = allShares[:, midpoint-20:midpoint+20]
        print (allShares.shape)

        shares1 = np.sum(allShares, axis=2)
        midpoint2 = allShares.shape[1] // 2
        shares2 = allShares[:, midpoint2-1:midpoint2+1]

        print (shares1.shape)
        print (shares2.shape)

        np.savez_compressed('./inputData/ITCH_LOB/nearSharesSubset/S1_' + name + '.npz', shares1)
        np.savez_compressed('./inputData/ITCH_LOB/nearSharesSubset/S2_' + name + '.npz', shares2)

        #print (shares.shape)
        #quit()



    saveComponent(59)

def saveAllLOBRelated():
    #individualLOB(True)
    #individualLOB(False)
    shareLOB(True)
    shareLOB(False)
    constructBestPrice(True)
    constructBestPrice(False)
    saveNearByShare(True)
    saveNearByShare(False)
    saveNearByShareFull()
    #saveLOBhistWithStarts()
    True

#saveAllLOBRelated()
#quit()


def saveXCompressed():
    #name = '20200117'
    #isBuy = True
    def saveComponent(nameNum):
        names = giveFNames()
        print (nameNum)
        name = names[nameNum]
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

    multiProcessor(saveComponent, 50, 111, 4, doPrint = True)


#saveXCompressed()
#individualLOB(True)
#individualLOB(False)
#quit()


def formLOBrecentShares():

    #isBuy = True
    #for name in names:
    def saveComponent(nameNum):
        names = giveFNames()
        name = names[nameNum]
        print (nameNum)
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

            argsBuy = np.argwhere(data[:, 4] == "B")[:, 0]
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
                #if a % 100 == 0:
                #    print (a)
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

    #saveComponent(0)
    #quit()
    #20:
    multiProcessor(saveComponent, 20, 111, 4, doPrint = True)



#formLOBrecentShares()
#quit()

def formTimedLOBrecentShares():

    #isBuy = True
    #for name in names:
    def saveComponent(nameNum):
        names = giveFNames()
        name = names[nameNum]
        print (nameNum)
        allSharesFull = []
        for isBuy in [True, False]:
            #print (isBuy)
            data = loadnpz('./inputData/ITCH/includeOrderInfo/' + name + '_SPY.npz')
            if isBuy:
                LOBhistory = loadnpz('./inputData/ITCH_LOB/BuyLOB/' + name + '.npz', allow_pickle=True)
                argsBuy = np.argwhere(data[:, 4] == "B")[:, 0]
            else:
                LOBhistory = loadnpz('./inputData/ITCH_LOB/SellLOB/' + name + '.npz', allow_pickle=True)
                argsBuy = np.argwhere(data[:, 4] == "S")[:, 0]
            M = 20
            sizes = []
            LOBNewFull = []
            FullNums = []
            for Num in range(0, len(LOBhistory)):
                LOBNew = [LOBhistory[Num][0], np.zeros((len(LOBhistory[Num][0]), M, 3)).astype(int)]
                tempNums = []
                for a in range(0, len(LOBhistory[Num][1])):
                    num = len(LOBhistory[Num][1][a])
                    tempNums.append(num)
                    vals = np.zeros((M, 3)).astype(int)
                    maxBack = min(len(LOBhistory[Num][1][a][-M:]), M)

                    if num != 0:
                        #if num > M:
                        #    print (np.array(LOBhistory[Num][1][a]))
                        vals2 = np.array(LOBhistory[Num][1][a])
                        vals2[:, 2] =  argsBuy[vals2[:, 2].astype(int)]
                        #print (vals2)
                        #print (LOBhistory[Num][1])
                        #quit()

                        if num >= M:
                            vals3 = vals2[:-2]
                            vals3 = np.concatenate((vals3[vals3[:, 1] <= 10], vals3[vals3[:, 1] > 10]))
                            vals2[:-2] = vals3

                        vals[-maxBack:, :] =  vals2[-M:, :] #sept 18 #vals2[-M:, :] #oct 21 #vals2[-M:, :2]
                        vals[0, 1] = np.sum( vals2[:-(M-1), 1] )
                        #if num > M:
                        #    print (vals)
                        #    quit()
                    LOBNew[1][a] = vals
                LOBNewFull.append(LOBNew)
                FullNums.append(tempNums)
            del LOBhistory

            if isBuy:
                data = data[data[:, 4] == "B"]
                #LOB_share = loadnpz('./inputData/ITCH_LOB/BuyShareLOB/' + name + '.npz', allow_pickle=True)
                bestPrice = loadnpz('./inputData/ITCH_LOB/BestBid/' + name + '.npz')
                #shift = 20
                shift = 1
            else:
                data = data[data[:, 4] == "S"]
                #LOB_share = loadnpz('./inputData/ITCH_LOB/SellShareLOB/' + name + '.npz', allow_pickle=True)
                bestPrice = loadnpz('./inputData/ITCH_LOB/BestAsk/' + name + '.npz')
                #shift = -20
                shift = -1

            oncePerM = np.arange(data.shape[0])
            bestPrice = bestPrice[oncePerM]
            allShares = np.zeros((oncePerM.shape[0], (np.abs(shift) * 2) + 1, M, 2))
            for a in range(0, len(LOBNewFull)): #1879
                if a % 100 == 0:
                    print (a)
                #a from bestPrice - 40 to bestPrice
                args = stefan_squeeze(np.argwhere( np.abs(a - bestPrice + shift) <= np.abs(shift) )).astype(int)
                a2 = (a - bestPrice[args] + shift + np.abs(shift)).astype(int)
                ar = np.array(LOBNewFull[a][0])
                argsInAr = stefan_nextAfter(args+1, ar) - 1 #TODO correct -1 to mean not existing.

                if argsInAr.shape[0] > 0:
                    #print ("A")
                    #print (np.array(LOBNewFull[a][1]).shape)
                    #print (argsInAr)
                    sharesNew = np.array(LOBNewFull[a][1])[argsInAr]
                    numsNow = np.array(FullNums[a])[argsInAr]

                    #print (sharesNew[:, :, 1:].shape)
                    #print (allShares[args, a2].shape)
                    allShares[args, a2] = sharesNew[:, :, 1:]
            #print (allShares.shape)
            allSharesFull.append(np.copy(allShares))

            #del data
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
        allShares = np.zeros((dataSize, shiftSize * 2, M, 2)).astype(int)
        #print ("D")
        allShares[:, :shiftSize, :] = allSharesFull[0][beforeBids, :, :]
        del allSharesFull[0]
        allShares[:, shiftSize:, :] = allSharesFull[0][beforeAsks, :, :]
        #print ("E")
        #print (allShares.shape)

        #midpoint = allShares.shape[1] // 2
        #allShares = allShares[:, midpoint-2:midpoint+2]
        #print (allShares.shape)

        np.savez_compressed('./inputData/ITCH_LOB/NearBestOrdersTimed/' + name + '.npz', allShares)
        del allShares
        del allSharesFull
        #print ("Saved")

    #multiProcessor(saveComponent, filesRun[0], filesRun[1], 4, doPrint = True)
    multiProcessor(saveComponent, 36, 111, 4, doPrint = True)
    #saveComponent(0)

#print ('formTimedLOBrecentShares()')
#formTimedLOBrecentShares()
#quit()

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import numpy as np
from modules import helper
from imblearn.over_sampling import RandomOverSampler
from collections import Counter



#select a dataset
def datasetSelector(dataset, seed_Value, topLevel='and', symbols = 4, nrEmpty = 2, andStack = 1, orStack = 1, xorStack = 1, nrAnds = 2, nrOrs = 2, nrxor = 2, trueIndexes=[1,3], test_size=0.2, orOffSet=0, xorOffSet=0, redaundantIndexes = [[0,0]]):
    symbolCount = 0
    if dataset == 'Andor':
        X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, dataName, num_of_classes, symbolCount=  doAndorMultiInNOut(seed_Value, topLevel=topLevel, symbols = symbols, nrEmpty = nrEmpty, andStack = andStack, orStack = orStack, xorStack = xorStack, nrAnds = nrAnds, nrOrs = nrOrs, nrxor = nrxor, trueIndexes=trueIndexes, test_size=test_size, orOffSet=orOffSet, xorOffSet=xorOffSet, redaundantIndexes = redaundantIndexes)
    else:
        X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, dataName, num_of_classes = []

    y_train = np.array(y_train)
    y_train = y_train.astype(float)
    y_test = np.array(y_test)
    y_test = y_test.astype(float)
    X_test = np.array(X_test)
    X_test = X_test.astype(float)
    X_train = np.array(X_train)
    X_train = X_train.astype(float)   
    y_testy = np.array(y_testy)
    y_trainy = np.array(y_trainy)

    return X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, dataName, num_of_classes, symbolCount

#create the andor dataset!
def doAndorMultiInNOut(seed_value, topLevel='and', symbols = 4, nrEmpty = 2, andStack = 1, orStack = 1, xorStack = 1, nrAnds = 2, nrOrs = 2, nrxor = 2, trueIndexes=[1,3], test_size=0.2, orOffSet=0, xorOffSet=0, redaundantIndexes = [[0,0]]): #TODO redundancy über 2 inputs? Wird da einer ignoriert?
    
    dataName = 'Andor'
    symbolA = helper.getMapValues(symbols)
    symbolA = np.array(symbolA)
    trueSymbols = symbolA[trueIndexes]
    dataLen = andStack * nrAnds + orStack * nrOrs + xorStack * nrxor + nrEmpty - orOffSet * orStack - xorOffSet * xorStack
    seqSize = dataLen
    size = pow(symbols, dataLen)

    dataSet = []
    labelSet = []
    n = -1
    for i in range(size):
        if True:
            dataSet.append(np.zeros(dataLen))
            n+=1
            p = i
            for j in range(dataLen):
                inV = p % symbols
                dataSet[n][j] = symbolA[inV]
                p = int(p/symbols)

            for r in redaundantIndexes:
                dataSet[n][r[1]] = dataSet[n][r[0]]

            andTs = []
            for j in range(andStack):
                andT = True
                for k in range((nrAnds*j), nrAnds*(j+1)):
                    if round(float(dataSet[n][k]), 4) not in trueSymbols:
                        andT = False
                        break
                if nrAnds == 0:
                    andT = True
                andTs.append(andT)
            maxAnds = (nrAnds * andStack)

            orTs = []
            for j in range(orStack):
                orT = False
                for k in range(maxAnds + (nrOrs *j - orOffSet * j),maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))):
                    if round(float(dataSet[n][k]), 4) in trueSymbols:
                        orT = True
                        break
                if nrOrs == 0:
                    orT = True
                orTs.append(orT)
            maxOrs = nrOrs * orStack - orOffSet * orStack

            xorTs = []
            for j in range(xorStack):
                xorT = False
                for k in range(maxAnds+maxOrs+(nrxor *j - xorOffSet * j),maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))):
                    if round(float(dataSet[n][k]), 4) in trueSymbols:
                        if xorT == True:
                            xorT = False
                            break
                        else:
                            xorT = True
                if nrxor == 0:
                    xorT = True
                xorTs.append(xorT)

            andTs = np.array(andTs)
            orTs = np.array(orTs)
            xorTs = np.array(xorTs)

            if topLevel == 'and':
                if (np.sum(andTs) + np.sum(orTs) + np.sum(xorTs) == andStack+orStack+xorStack):
                    labelSet.append(1)
                else:
                    labelSet.append(0)
            elif topLevel == 'or':
                if (np.sum(andTs) + np.sum(orTs) + np.sum(xorTs) >= 1):
                    labelSet.append(1)
                else:
                    labelSet.append(0)
            elif topLevel == 'xor':
                if (np.sum(andTs) + np.sum(orTs) + np.sum(xorTs) == 1):
                    labelSet.append(1)
                else:
                    labelSet.append(0)
            else:
                    raise ValueError("Need a valid top level, but got: "+ topLevel)

    num_of_classes = len(set(labelSet))
    dataSet = np.array(dataSet)

    X_train, y_train = shuffle(dataSet, labelSet, random_state = seed_value)

    if test_size > 0 and test_size < 1:
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size, shuffle=True, random_state=seed_value)
    else:
        X_test = X_train
        y_test = y_train

    ros = RandomOverSampler(random_state=0)
    X_train, y_train = ros.fit_resample(X_train, y_train)         

    print(sorted(Counter(y_train).items()))
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)


    y_trainy = np.array(y_train) +1
    y_train = []
    X_train = X_train
    y_testy = np.array(y_test) +1
    y_test = []
    X_test = X_test
    

    for y in y_trainy:
        y_train_puffer = np.zeros(num_of_classes)
        y_train_puffer[y-1] = 1
        y_train.append(y_train_puffer)

    for y in y_testy:
        y_puffer = np.zeros(num_of_classes)
        y_puffer[y-1] = 1
        y_test.append(y_puffer)

    y_train = np.array(y_train)
    y_train = y_train.astype(float)
    y_test_full = np.array(y_test)
    y_test_full = y_test_full.astype(float)
    y_test = y_test_full  
    y_test = y_test.astype(float)

    print(X_test.shape)
    print(X_train.shape)
    print(y_test.shape)
    print(y_train.shape)

    X_test = X_test.astype(float)
    X_train = X_train.astype(float)

    return X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, dataName, num_of_classes,symbols
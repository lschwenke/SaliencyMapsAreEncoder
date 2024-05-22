from sacred import Experiment
import seml
import warnings
import torch

from sklearn.metrics import accuracy_score

import os
import random
import numpy as np

from sklearn.model_selection import StratifiedKFold

from modules import dataset_selecter as ds
from modules import pytorchTrain as pt
from modules import saliencyHelper as sh
from modules import helper

from sklearn.ensemble import RandomForestClassifier

import ViT_LRP
import cnn_LRP

from datetime import datetime

ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

def arrayToString(indexes):
    out = ""
    for i in indexes:
        out = out + ',' + str(i)
    return out


class ExperimentWrapper:
    """
    A simple wrapper around a sacred experiment, making use of sacred's captured functions with prefixes.
    This allows a modular design of the configuration, where certain sub-dictionaries (e.g., "data") are parsed by
    specific method. This avoids having one large "main" function which takes all parameters as input.
    """

    def __init__(self, init_all=True):
        if init_all:
            self.init_all()

    #init before the experiment!
    @ex.capture(prefix="init")
    def baseInit(self, nrFolds: int, patience: int, seed_value: int):
        self.seed_value = seed_value
        os.environ['PYTHONHASHSEED']=str(seed_value)# 2. Set `python` built-in pseudo-random generator at a fixed value
        random.seed(seed_value)# 3. Set `numpy` pseudo-random generator at a fixed value
        np.random.RandomState(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(0)


        #save some variables for later
        self.kf = StratifiedKFold(nrFolds, shuffle=True, random_state=seed_value)
        self.fold = 0
        self.nrFolds = nrFolds
        self.seed_value = seed_value       
        self.patience = patience

        #init gpu
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")


    # Load the dataset
    @ex.capture(prefix="data")
    def init_dataset(self, dataset: str, toplevel: str, symbols: int, nrEmpty: int, andStack: int, orStack: int, xorStack: int, nrAnds: int, nrOrs: int, nrxor: int, trueIndexes, test_size: float, orOffSet: int, xorOffSet: int, redaundantIndexes):
        """
        Perform dataset loading, preprocessing etc.
        Since we set prefix="data", this method only gets passed the respective sub-dictionary, enabling a modular
        experiment design.
        """

        #Get dataset and prepare dataset
        self.X_train, self.X_test, self.y_train, self.y_test, self.y_trainy, self.y_testy, self.seqSize, self.dataName, self.num_of_classes, self.symbolCount = ds.datasetSelector(dataset, self.seed_value, topLevel=toplevel, symbols = symbols, nrEmpty = nrEmpty, andStack = andStack, orStack = orStack, xorStack = xorStack, nrAnds = nrAnds, nrOrs = nrOrs, nrxor = nrxor, trueIndexes=trueIndexes, test_size=test_size, orOffSet=orOffSet, xorOffSet=xorOffSet, redaundantIndexes=redaundantIndexes)

        self.test_size = test_size
        self.inDim = self.X_train.shape[1]
        self.dataset = dataset
        self.toplevel = toplevel
        self.symbols = symbols
        self.nrEmpty = nrEmpty
        self.andStack = andStack
        self.orStack = orStack
        self.xorStack = xorStack
        self.nrAnds = nrAnds
        self.nrOrs = nrOrs
        self.nrxor = nrxor
        self.trueIndexes = trueIndexes
        self.orOffSet = orOffSet
        self.xorOffSet = xorOffSet
        self.redaundantIndexes = redaundantIndexes


        self.dsName = str(self.dataName) +  ',l:' + str(toplevel) +',s:' +  str(symbols)+',e:' +  str(nrEmpty)+',a:' +  str(andStack)+',o:' +  str(orStack)+',x:' +  str(xorStack)+',na:' +  str(nrAnds)+',no:' +  str(nrOrs)+',nx:' +  str(nrxor)+',i:' +  arrayToString(trueIndexes)+',t:' +  str(test_size)+',oo:' +  str(orOffSet)+',xo:' +  str(xorOffSet) +',r:' + arrayToString(redaundantIndexes)
        print(self.dsName)


    #all inits
    def init_all(self):
        """
        Sequentially run the sub-initializers of the experiment.
        """
        self.baseInit()
        self.init_dataset()

    def printTime(self):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)


    @ex.capture(prefix="model")
    def trainExperiment(self, useSaves: bool, hypers, batch_size: int, stride: int, kernal_size: int, nc: int, thresholdSet, methods): #, foldModel: int):

        print('Dataname:')
        print(self.dsName)
        self.printTime()
        warnings.filterwarnings('ignore')   

        #big result dict!
        fullResults = dict()
        
        #get hyperparameters
        modelType = hypers[0]
        epochs = hypers[1]
        dmodel = hypers[2]
        dfff = hypers[3]
        doSkip = hypers[4]
        doBn = hypers[5]
        header = hypers[6]
        numOfLayers = hypers[7]
        dropout = hypers[8]
        att_dropout = hypers[9]
        if modelType == 'Transformer':
            doClsTocken = hypers[10]
        else:
            doClsTocken = False

        #save folders
        fullResultDir = 'presults' 
        filteredResults= 'filteredResults'

        # Don't calc again if already there
        wname = pt.getWeightName(self.dsName, self.dataName, batch_size, epochs, numOfLayers, header, dmodel, dfff, dropout, att_dropout, doSkip, doBn, doClsTocken, learning = False, results = True, resultsPath=filteredResults)
        if os.path.isfile(wname + '.pkl'):
            fullResults["Error"] = "dataset " + self.dsName + " already done: " + str(self.seqSize) + "; name: " + wname
            print('Already Done ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print("dataset " + self.dataName + " already done: " + str(self.seqSize) + "; name: " + wname)
        
            return "dataset " + self.dataName + "already done: " + str(self.seqSize)  + "; name: " + wname 

        fullResults['testData'] = self.X_test
        fullResults['testTarget'] = self.y_test

        #save all params!
        fullResults['params'] = dict()
        fullResults['params']['patience'] = self.patience
        fullResults['params']['fileName'] = self.dsName
        fullResults['params']['epochs'] = epochs
        fullResults['params']['batchSize'] = batch_size
        fullResults['params']['useSaves'] = useSaves
        fullResults['params']['numOfLayers'] = numOfLayers
        fullResults['params']['header'] = header
        fullResults['params']['dmodel'] = dmodel
        fullResults['params']['dfff'] = dfff
        fullResults['params']['dropout'] = dropout
        fullResults['params']['att_dropout'] = att_dropout
        fullResults['params']['doSkip'] = doSkip
        fullResults['params']['doBn'] = doBn
        fullResults['params']['modelType'] = modelType
        fullResults['params']['doClsTocken'] = doClsTocken
        fullResults['params']['thresholdSet'] = thresholdSet
        fullResults['params']['dataset'] = self.dataset
        fullResults['params']['toplevel'] = self.toplevel
        fullResults['params']['symbols'] = self.symbols
        fullResults['params']['nrEmpty'] = self.nrEmpty
        fullResults['params']['andStack'] = self.andStack
        fullResults['params']['orStack'] = self.orStack
        fullResults['params']['xorStack'] = self.xorStack
        fullResults['params']['nrAnds'] = self.nrAnds
        fullResults['params']['nrOrs'] = self.nrOrs
        fullResults['params']['nrxor'] = self.nrxor
        fullResults['params']['trueIndexes'] = self.trueIndexes
        fullResults['params']['test_size'] = self.test_size
        fullResults['params']['orOffSet'] = self.orOffSet
        fullResults['params']['xorOffSet'] = self.xorOffSet
        fullResults['params']['redaundantIndexes'] = self.redaundantIndexes
        print(fullResults['params'])

        fullResults['results'] = dict()
        resultDict = fullResults['results']
        resultDict['trainPred'] = []
        resultDict['trainAcc'] = []
        resultDict['trainLoss'] = []

        resultDict['valPred'] = []
        resultDict['valAcc'] = []
        resultDict['valLoss'] = []

        resultDict['testPred'] = []
        resultDict['testAcc'] = []
        resultDict['testLoss'] = []

        resultDict['trainData'] = []
        resultDict['trainTarget'] = []
        resultDict['valData'] = []
        resultDict['valTarget'] = []

        resultDict['treeScores'] = []
        resultDict['treeImportances'] = []


        print('Base data shapes:')
        print(self.X_train.shape)
        print(self.X_test.shape)

        #n fold cross validation
        self.fold = 0
        for train, test in self.kf.split(self.X_train, self.y_trainy):

            self.fold+=1
            print(f"Fold #{self.fold}")
            
            
            if self.test_size > 0 and  self.test_size < 1:
                x_train1 = self.X_train[train]
                x_val = self.X_train[test]
                y_train1 = self.y_train[train]
                y_trainy1 = self.y_trainy[train]
                y_val = self.y_train[test]
                
            else:
                x_train1 = self.X_train.copy()
                x_val = self.X_train.copy()
                y_train1 = self.y_train.copy()
                y_trainy1 = self.y_trainy.copy()
                y_val = self.y_train.copy()

            x_test = self.X_test.copy()

            #create model
            if modelType == 'CNN':
                model = cnn_LRP.ResNetLikeClassifier(outputs=self.num_of_classes, inDim=self.inDim, num_hidden_layers=numOfLayers, nc=nc, nf=dmodel, dropout=dropout, maskValue = -2, stride=stride, kernel_size=kernal_size, doSkip=doSkip, doBn=doBn)
                
                if True:
                    x_train1 = np.expand_dims(x_train1,1)
                    x_val = np.expand_dims(x_val,1)
                    x_test = np.expand_dims(x_test,1)
            elif modelType == 'Transformer':
                model = ViT_LRP.TSModel(num_hidden_layers=numOfLayers, inDim=self.inDim, dmodel=dmodel, dfff=dfff, num_heads=header, num_classes=self.num_of_classes, dropout=dropout, att_dropout=att_dropout, doClsTocken=doClsTocken)
            else:  
                raise ValueError('Not a valid model type: ' + modelType)

            print('Train data shapes:')
            print(x_train1.shape)
            print(x_test.shape)

            #train model
            model.double()
            model.to(self.device)
            model, trainPred, trainAcc, trainLoss, valPred, valAcc, valLoss, testPred, testAcc, testLoss = pt.trainBig(self.device, model, x_train1, y_train1, x_val, y_val, x_test, self.patience, useSaves, self.y_test, batch_size, epochs, fileAdd=self.dsName)
            model.eval()


            train_predictions = np.argmax(y_train1, axis=1)+1
            test_predictions = np.argmax(self.y_test, axis=1)+1

            #train random forest
            newT = np.squeeze(x_train1)
            newG = np.squeeze(x_test)
            clf = RandomForestClassifier()
            clf.fit(newT, train_predictions)
            scores = clf.score(newG, test_predictions)

            #save results
            resultDict['treeScores'].append(scores)
            resultDict['treeImportances'].append(clf.feature_importances_)
            
            resultDict['trainPred'].append(trainPred)
            resultDict['trainAcc'].append(trainAcc)
            resultDict['trainLoss'].append(trainLoss)

            resultDict['valPred'].append(valPred)
            resultDict['valAcc'].append(valAcc)
            resultDict['valLoss'].append(valLoss)

            resultDict['testPred'].append(testPred)
            resultDict['testAcc'].append(testAcc)
            resultDict['testLoss'].append(testLoss)

            resultDict['trainData'].append(x_train1)
            resultDict['trainTarget'].append(y_train1)
            resultDict['valData'].append(x_val)
            resultDict['valTarget'].append(y_val)

            if 'saliency' not in fullResults.keys():
                fullResults['saliency'] = dict()
            saliencies = fullResults['saliency']

            for method in methods[modelType].keys():
                if modelType == 'Transformer' and method == 'Attention':
                    continue
                for submethod in methods[modelType][method]:
                    if method+'-'+submethod not in saliencies.keys():
                        saliencies[method+'-'+submethod] = dict()
                        outMap = saliencies[method+'-'+submethod]
                        outMap['Fidelity'] = dict()
                        outMap['Infidelity'] = dict()
                        outMap['outTrain'] = []
                        outMap['outVal'] = []
                        outMap['outTest'] = []
                        outMap['modelTrain'] = []
                        outMap['modelVal'] = []
                        outMap['modelTest'] = []
                        outMap['means'] = dict()
                        outMap['means']['outTrain'] = []
                        outMap['means']['outVal'] = []
                        outMap['means']['outTest'] = []
                        outMap['means']['modelTrain'] = []
                        outMap['means']['modelVal'] = []
                        outMap['means']['modelTest'] = []
                        outMap['classes'] = dict()
                        for c in range(self.num_of_classes):
                            outMap['classes'][str(c)] = dict()
                            outMap['classes'][str(c)]['outTrain']= []
                            outMap['classes'][str(c)]['outVal'] = []
                            outMap['classes'][str(c)]['outTest'] = []
                    outMap = saliencies[method+'-'+submethod]
                    if submethod.startswith('smooth'):
                        smooth = True
                    else:
                        smooth = False

                    #get saliency maps!
                    outTrain, outVal, outTest, data3D, data2D = sh.getSaliencyMap(outMap, "out", self.device, self.num_of_classes, modelType, method, submethod, model, x_train1, x_val, x_test, trainPred, valPred, testPred, smooth, doClassBased=True)

                    for doFidelity in [False]:#[True, False]:
                        if doFidelity:
                            rfDict = outMap['Fidelity']
                        else:
                            rfDict = outMap['Infidelity']

                        #ROAR approach per threshold
                        for threshold in thresholdSet:
                            print('Starting threshold:')
                            print(threshold)

                            #creating masked data
                            if threshold == 'baseline':
                                newTrain, trainReduction = sh.doSimpleLasaROAR(outTrain, x_train1, self.nrEmpty,  doBaselineT=True, doFidelity=doFidelity, do3DData=data3D, do3rdStep=data2D)
                                newVal, valReduction = sh.doSimpleLasaROAR(outVal, x_val, self.nrEmpty, doBaselineT=True, doFidelity=doFidelity, do3DData=data3D, do3rdStep=data2D)
                                newTest, testReduction = sh.doSimpleLasaROAR(outTest, x_test, self.nrEmpty,  doBaselineT=True, doFidelity=doFidelity, do3DData=data3D, do3rdStep=data2D)
                            else:
                                newTrain, trainReduction = sh.doSimpleLasaROAR(outTrain, x_train1, threshold, doFidelity=doFidelity, do3DData=data3D, do3rdStep=data2D)
                                newVal, valReduction = sh.doSimpleLasaROAR(outVal, x_val, threshold, doFidelity=doFidelity, do3DData=data3D, do3rdStep=data2D)
                                newTest, testReduction = sh.doSimpleLasaROAR(outTest, x_test, threshold, doFidelity=doFidelity, do3DData=data3D, do3rdStep=data2D)

                            # retrain model creation
                            if modelType == 'CNN':
                                model2 = cnn_LRP.ResNetLikeClassifier(outputs=self.num_of_classes, inDim=self.inDim, num_hidden_layers=numOfLayers, nc=1, nf=dmodel, dropout=dropout, maskValue = -2, stride=1, kernel_size=3, doSkip=doSkip, doBn=doBn)
                                
                                if True:
                                    newTrain = np.expand_dims(newTrain,1)
                                    newVal = np.expand_dims(newVal,1)
                                    newTest = np.expand_dims(newTest,1)
                            elif modelType == 'Transformer':
                                model2 = ViT_LRP.TSModel(num_hidden_layers=numOfLayers, inDim=self.inDim, dmodel=dmodel, dfff=dfff, num_heads=header, num_classes=self.num_of_classes, dropout=dropout, att_dropout=att_dropout, doClsTocken=doClsTocken)
                            else: 
                                raise ValueError('Not a valid model type: ' + modelType)
                            
                            # retrain model start
                            model2.double()
                            model2.to(self.device)
                            model2, trainPred2, trainAcc2, trainLoss2, valPred2, valAcc2, valLoss2, testPred2, testAcc2, testLoss2 = pt.trainBig(self.device, model2, newTrain, y_train1, newVal, y_val, newTest, self.patience, False, self.y_test, batch_size, epochs, fileAdd=self.dsName+"-lasa")
                            model2.eval()

                            #save results
                            if str(threshold) not in rfDict.keys():
                                rfDict[str(threshold)] = dict()
                                rftDict = rfDict[str(threshold)]
                                rftDict['trainPred'] = []
                                rftDict['trainAcc'] = []
                                rftDict['trainLoss'] = []

                                rftDict['valPred'] = []
                                rftDict['valAcc'] = []
                                rftDict['valLoss'] = []

                                rftDict['testPred'] = []
                                rftDict['testAcc'] = []
                                rftDict['testLoss'] = []

                                rftDict['treeScores'] = []
                                rftDict['treeImportances'] = []

                                
                                rftDict['trainReduction'] = []
                                rftDict['valReduction'] = []
                                rftDict['testReduction'] = []

                            rftDict = rfDict[str(threshold)]
                            subrftDict = rftDict['approx data']

                            rftDict['trainPred'].append(trainPred2)
                            rftDict['trainAcc'].append(trainAcc2)
                            rftDict['trainLoss'].append(trainLoss2)

                            rftDict['valPred'].append(valPred2)
                            rftDict['valAcc'].append(valAcc2)
                            rftDict['valLoss'].append(valLoss2)

                            rftDict['testPred'].append(testPred2)
                            rftDict['testAcc'].append(testAcc2)
                            rftDict['testLoss'].append(testLoss2)

                            rftDict['trainReduction'].append(trainReduction)
                            rftDict['valReduction'].append(valReduction)
                            rftDict['testReduction'].append(testReduction)


                            train_predictions = np.argmax(y_train1, axis=1)+1
                            test_predictions = np.argmax(self.y_test, axis=1)+1

                            newT = np.squeeze(newTrain)
                            newG = np.squeeze(newTest)

                            clf = RandomForestClassifier()
                            clf.fit(newT, train_predictions)
                            scores = clf.score(newG, test_predictions)
                            rftDict['treeScores'].append(scores)
                            rftDict['treeImportances'].append(clf.feature_importances_)
                

        # Evaluate further metrics and results
        saveName = self.evaluateAndSaveResults(fullResults, useSaves, hypers, batch_size, stride, kernal_size, nc, thresholdSet, methods, filteredResults)
        saveName = pt.getWeightName(self.dsName, self.dataName, batch_size, epochs, numOfLayers, header, dmodel, dfff, dropout, att_dropout, doSkip, doBn, doClsTocken, learning = False, results = True, resultsPath=fullResultDir)
        print(saveName)

        #NOTE uncoment if full very big results should be saved.
        #helper.save_obj(fullResults, str(saveName))
        

        self.printTime()

        return saveName


    # evaluates further results and saves them
    def evaluateAndSaveResults(self, res, useSaves: bool, hypers, batch_size: int, stride: int, kernal_size: int, nc: int, thresholdSet, methods, filteredResults):
        # set hyper params
        self.printTime()
        modelType = hypers[0]
        epochs = hypers[1]
        dmodel = hypers[2]
        dfff = hypers[3]
        doSkip = hypers[4]
        doBn = hypers[5]
        header = hypers[6]
        numOfLayers = hypers[7]
        dropout = hypers[8]
        att_dropout = hypers[9]
        if modelType == 'Transformer':
            doClsTocken = hypers[10]
        else:
            doClsTocken = False

        
        trueIndexes = res['params']['trueIndexes']
        nrSymbols = res['params']['symbols']
        symbolA = helper.getMapValues(nrSymbols)
        symbolA = np.array(symbolA)
        res['params']

        andStack = res['params']['andStack']
        orStack = res['params']['orStack']
        xorStack = res['params']['xorStack']
        nrAnds = res['params']['nrAnds']
        nrOrs = res['params']['nrOrs']
        nrxor = res['params']['nrxor']
        orOffSet = res['params']['orOffSet']
        xorOffSet = res['params']['xorOffSet']
        topLevel = res['params']['toplevel']

        tl = res['testTarget'] 
        gt = res['testData']
        num_of_classes = len(set(list(tl.flatten())))
        
        # create result structure
        finalResults = dict()
        finalResults['performance'] = []
        finalResults['tree performance'] = []
        finalResults['treeImportances'] = []
        finalResults['baseline'] = []

        for s in res['saliency'].keys():
            finalResults[s] = dict()

            finalResults[s]['saliency'] = []
            thresholds = res['params']['thresholdSet']
            
            for t in thresholds:
                t = 't'+str(t)
                finalResults[s][t] = dict()
                finalResults[s][t]['treeScores'] = []
                finalResults[s][t]['lasa acc'] = []
                finalResults[s][t]['lasa red'] = []
                finalResults[s][t]['DoubleAssigmentFull'] = dict()
                finalResults[s][t]['DoubleAssigmentTruthTableMin'] = dict()
                finalResults[s][t]['DoubleAssigmentFullPercent'] = dict()
                finalResults[s][t]['DoubleAssigmentTruthTableMinPercent'] = dict()
                finalResults[s][t]['LogicalAcc'] = []
                finalResults[s][t]['LogicalAccStatistics'] = []                

            for c in range(2):
                finalResults[s][c] = dict()
                finalResults[s][c]['saliency'] = []
                finalResults[s][c]['wrongImportanceMeanPercent'] = [] 
                finalResults[s][c]['wrongImportancePercent'] = []
                finalResults[s][c]['generalInformationBelowBaseline'] = [] 
                finalResults[s][c]['neededInformationBelowBaseline'] = []

            finalResults[s]['saliencyPerStack'] = dict() 


            finalResults[s]['wrongImportanceMeanPercent'] = []
            finalResults[s]['wrongImportancePercent'] = []


        # fill all results
        for s in res['results']['treeScores']:
            finalResults['tree performance'].append(s)

        for s in res['results']['treeImportances']:
            finalResults['treeImportances'].append(s)

        for s in res['results']['testAcc']:
            finalResults['performance'].append(s)

        tl = res['testTarget']
        baselineAcc = accuracy_score(np.zeros(len(tl)), tl.argmax(axis=1))
        finalResults['baseline'].append(baselineAcc)

        for k in res['saliency'].keys():
            saliencyStacksValues = dict()


            for v in np.mean(np.array(res['saliency'][k]['outTest']), axis=(1)):
                finalResults[k]['saliency'].append(v)

            for c in res['saliency'][k]['classes'].keys():
                for v in np.mean(np.array(res['saliency'][k]['outTest'])[:,np.argmax(tl, axis=1)== int(c)], axis=(1)):
                    finalResults[k][int(c)]['saliency'].append(v)

            for f in range(len(res['saliency'][k]['outTest'])):
                sd = res['saliency'][k]['outTest'][f]
                do3DData = False
                do2DData = False
                if len(sd.shape) > 3:
                    do3DData = True
                elif len(sd.shape) > 2:
                    do2DData = True
                _, saliencyStacks, _, _ = sh.getPredictionMaps('', '', gt, sd, tl, num_of_classes, topLevel, nrSymbols, andStack, orStack, xorStack, nrAnds, nrOrs, nrxor, orOffSet, xorOffSet, trueIndexes)
                for k2 in saliencyStacks.keys():
                    if k2 not in finalResults[k]['saliencyPerStack'].keys():
                        finalResults[k]['saliencyPerStack'][k2] = dict()
                        finalResults[k]['saliencyPerStack'][k2][0] = []
                        finalResults[k]['saliencyPerStack'][k2][1] = []
                    finalResults[k]['saliencyPerStack'][k2][0].append(np.mean(saliencyStacks[k2][0], axis=0))
                    finalResults[k]['saliencyPerStack'][k2][1].append(np.mean(saliencyStacks[k2][1], axis=0))

            
                saliencyStacksValues = sh.splitSaliencyPerStack(saliencyStacksValues, sd, gt, gt, tl, num_of_classes, nrSymbols, andStack, orStack, xorStack, nrAnds, nrOrs, nrxor, orOffSet, xorOffSet, trueIndexes, do3DData=do3DData, do3rdStep=do2DData)  

            
            saliency1Ds = []
            for f in np.array(res['saliency'][k]['outTest']):
                saliency1Ds.append(sh.reduceMap(f, do3DData=do3DData, do3rdStep=do2DData))
            saliency1Ds = np.array(saliency1Ds)

            wrongImportanceMean = np.zeros(len(res['saliency'][k]['means']['outTest'][0]))
            for r in np.mean(saliency1Ds, axis=1):
                baseline = np.max(r[-1* res['params']['nrEmpty']:])

                for l in range(len(res['saliency'][k]['means']['outTest'][0])-res['params']['nrEmpty']):
                    if r[l] < baseline:
                        wrongImportanceMean[l] += 1
                    for p in range(res['params']['nrEmpty']):
                        if r[l] < r[-1*(p+1)]:
                            wrongImportanceMean[-1*(p+1)] += 1
            finalResults[k]['wrongImportanceMeanPercent'] = wrongImportanceMean / len(saliency1Ds) 
            

            
            for c in res['saliency'][k]['classes'].keys():
                wrongImportanceMeanC = np.zeros(len(res['saliency'][k]['means']['outTest'][0]))
                for r in np.mean(saliency1Ds[:,np.argmax(tl, axis=1)== int(c)], axis=(1)):
                    baseline = np.max(r[-1* res['params']['nrEmpty']:])
                    for l in range(len(res['saliency'][k]['means']['outTest'][0])-res['params']['nrEmpty']):
                        if r[l] < baseline:
                            wrongImportanceMeanC[l] += 1
                        for p in range(res['params']['nrEmpty']):
                            if r[l] < r[-1*(p+1)]:
                                wrongImportanceMeanC[-1*(p+1)] += 1
                finalResults[k][int(c)]['wrongImportanceMeanPercent'] = wrongImportanceMeanC / len(saliency1Ds) 
            
            for f in saliency1Ds:
                wrongImportance = np.zeros(len(res['saliency'][k]['outTest'][0][0]))
                for r in f:
                    baseline = np.max(r[-1* (res['params']['nrEmpty']):])
                    for l in range(len(res['saliency'][k]['outTest'][0][0])-res['params']['nrEmpty']):
                        if r[l] < baseline:
                            wrongImportance[l] += 1
                        for p in range(res['params']['nrEmpty']):
                            if r[l] < r[-1*(p+1)]:
                                wrongImportance[-1*(p+1)] += 1
                finalResults[k]['wrongImportancePercent'].append(wrongImportance/ len(f)) 

            for c in res['saliency'][k]['classes'].keys():
                for f in saliency1Ds[:,np.argmax(tl, axis=1)== int(c)]:
                    wrongImportanceC = np.zeros(len(res['saliency'][k]['outTest'][0][0]))
                    for r in f:
                        baseline = np.max(r[-1* (res['params']['nrEmpty']):])
                        for l in range(len(res['saliency'][k]['outTest'][0][0])-res['params']['nrEmpty']):
                            if r[l] < baseline:
                                wrongImportanceC[l] += 1
                            for p in range(res['params']['nrEmpty']):
                                if r[l] < r[-1*(p+1)]:
                                    wrongImportanceC[-1*(p+1)] += 1
                    finalResults[k][int(c)]['wrongImportancePercent'].append(wrongImportanceC/ len(f))
                    
            

            for f in saliency1Ds:
                wongMeaningImportanceTemp, countImportanceMeaningTemp = sh.getPredictionSaliency(res['params']['nrEmpty'], f, gt, res['saliency'][k]['classes'].keys(), tl, nrSymbols, andStack, orStack, xorStack, nrAnds, nrOrs, nrxor, orOffSet, xorOffSet, trueIndexes)
                for c in wongMeaningImportanceTemp.keys():
                    finalResults[k][c]['generalInformationBelowBaseline'].append(wongMeaningImportanceTemp[c]/countImportanceMeaningTemp[c])

                brokenImportanceTemp, brokenStackImportanceTemp = sh.getStackImportanceBreak(res['params']['nrEmpty'], f, gt, res['saliency'][k]['classes'].keys(), tl, nrSymbols, andStack, orStack, xorStack, nrAnds, nrOrs, nrxor, orOffSet, xorOffSet, trueIndexes, topLevel)
                for c in brokenImportanceTemp.keys():
                    finalResults[k][c]['neededInformationBelowBaseline'].append(brokenImportanceTemp[c])
        
        #per threshold results
        for t in thresholds:
            self.thresholdsProcess(res, t, finalResults)

        saveName = pt.getWeightName(self.dsName, self.dataName, batch_size, epochs, numOfLayers, header, dmodel, dfff, dropout, att_dropout, doSkip, doBn, doClsTocken,learning = False, results = True, resultsPath=filteredResults)
        helper.save_obj(finalResults, str(saveName))

        return saveName


    
    def thresholdsProcess(self, res, thold, finalResults):
        
        gt = res['testData']
        tl = res['testTarget']
        num_of_classes = len(set(list(tl.flatten())))
        trueIndexes = res['params']['trueIndexes']
        nrSymbols = res['params']['symbols']
        symbolA = helper.getMapValues(nrSymbols)
        symbolA = np.array(symbolA)
        trueSymbols = symbolA[trueIndexes]
        falseSymbols = np.delete(symbolA, trueIndexes)

        andStack = res['params']['andStack']
        orStack = res['params']['orStack']
        xorStack = res['params']['xorStack']
        nrAnds = res['params']['nrAnds']
        nrOrs = res['params']['nrOrs']
        nrxor = res['params']['nrxor']
        orOffSet = res['params']['orOffSet']
        xorOffSet = res['params']['xorOffSet']
        topLevel = res['params']['toplevel']


        for k in res['saliency'].keys():
            sds = [] #saliencyMaps

            for f in range(len(res['saliency'][k]['outTest'])):
                sd = res['saliency'][k]['outTest'][f]
                sds.append(sd)
                do3DData = False
                do2DData = False
                if len(sd.shape) > 3:
                    do3DData = True
                elif len(sd.shape) > 2:
                    do2DData = True

                for t in [thold]:
                    t2 = 't'+str(t)
                    finalResults[k][t2]['lasa acc'].append(res['saliency'][k]['Infidelity'][str(t)]['testAcc'][f])
                    finalResults[k][t2]['lasa red'].append(res['saliency'][k]['Infidelity'][str(t)]['testReduction'][f])
                    finalResults[k][t2]['treeScores'].append(res['saliency'][k]['Infidelity'][str(t)]['treeScores'][f])

                    if t == 'baseline':
                        newTest, testReduction = sh.doSimpleLasaROAR(sd, gt, res['params']['nrEmpty'], doBaselineT=True, doFidelity=False, do3DData=do3DData, do3rdStep=do2DData)
                    else:
                        newTest, testReduction = sh.doSimpleLasaROAR(sd, gt, t, doFidelity=False, do3DData=do3DData, do3rdStep=do2DData)

                    logicMax, logicPred = sh.logicAcc(newTest, tl, nrSymbols, topLevel, trueIndexes=trueIndexes, andStack = andStack, orStack = orStack, xorStack = xorStack, nrAnds = nrAnds, nrOrs = nrOrs, nrxor = nrxor, orOffSet=orOffSet, xorOffSet=xorOffSet)
                    logicMaxStatistics, logicPredStatistics = sh.logicAccGuess(newTest, tl, nrSymbols, topLevel, trueIndexes=trueIndexes, andStack = andStack, orStack = orStack, xorStack = xorStack, nrAnds = nrAnds, nrOrs = nrOrs, nrxor = nrxor, orOffSet=orOffSet, xorOffSet=xorOffSet)
                    

                    finalResults[k][t2]['LogicalAcc'].append(logicMax)
                    finalResults[k][t2]['LogicalAccStatistics'].append(logicMaxStatistics)

                    lableMatches = np.argmax(res['results']['testPred'][f], axis=1) == np.argmax(res['saliency'][k]['Infidelity'][str(t)]['testPred'][f], axis=1)

                    tlM = res['saliency'][k]['Infidelity'][str(t)]['testPred'][f][lableMatches]
                    newTestM = newTest[lableMatches]
                    gtM = gt[lableMatches]
                    sdM = sd[lableMatches]
                    
                    
                    fullTestValueMap, _, _, _ = sh.getPredictionMaps('', '', gtM, newTestM, tlM, num_of_classes, topLevel, nrSymbols, 1, 0, 0, len(newTest[0].flatten())-res['params']['nrEmpty'],0 , 0, 0, 0, trueIndexes)

                    k2 = 'and0'
                    A = np.unique(np.array(fullTestValueMap[k2][0]), axis=0)
                    if len(A.shape) >= 3:
                        A = A.squeeze(axis=2)
                    B = np.unique(np.array(fullTestValueMap[k2][1]), axis=0)
                    if len(B.shape) >= 3:
                        B = B.squeeze(axis=2)


                    #NOTE happens because of only taking correct outputs and because sometimes some classes are not included inside a train subset!
                    if(len(A) == 0 or len(B) == 0):
                        m = []
                    else:
                        m = (A[:, None] == B).all(-1).any(1)
                    if k2 not in finalResults[k][t2]['DoubleAssigmentFull'].keys():
                        finalResults[k][t2]['DoubleAssigmentFull'][k2] = []
                        finalResults[k][t2]['DoubleAssigmentFullPercent'][k2] = []
                    if len(m) == 0:
                        finalResults[k][t2]['DoubleAssigmentFull'][k2].append(-1)     
                        finalResults[k][t2]['DoubleAssigmentFullPercent'][k2].append(-1)     
                    else:
                        finalResults[k][t2]['DoubleAssigmentFull'][k2].append(len(A[m]))
                        if len(A) < len(B):
                            finalResults[k][t2]['DoubleAssigmentFullPercent'][k2].append(len(A[m]) / len(A))
                        else:
                            finalResults[k][t2]['DoubleAssigmentFullPercent'][k2].append(len(A[m]) / len(B))

                    testValueMap, testConditionMap, classConValueMap, minConValueMap = sh.getPredictionMaps('', '', gtM, newTestM, tlM, num_of_classes, topLevel, nrSymbols, andStack, orStack, xorStack, nrAnds, nrOrs, nrxor, orOffSet, xorOffSet, trueIndexes)


                    for k2 in minConValueMap.keys():
                        A = np.unique(np.array(minConValueMap[k2][0]), axis=0)
                        if len(A.shape) >= 3:
                            A = A.squeeze(axis=2)
                        B = np.unique(np.array(minConValueMap[k2][1]), axis=0)
                        if len(B.shape) >= 3:
                            B = B.squeeze(axis=2)

                        if(len(A) == 0 or len(B) == 0):
                            m = []
                        else:
                            m = (A[:, None] == B).all(-1).any(1)
                        if k2 not in finalResults[k][t2]['DoubleAssigmentTruthTableMin'].keys():
                            finalResults[k][t2]['DoubleAssigmentTruthTableMin'][k2] = []
                            finalResults[k][t2]['DoubleAssigmentTruthTableMinPercent'][k2] = []
                        if len(m) == 0:
                            finalResults[k][t2]['DoubleAssigmentTruthTableMin'][k2].append(-1)    
                            finalResults[k][t2]['DoubleAssigmentTruthTableMinPercent'][k2].append(-1)
                        else:
                            finalResults[k][t2]['DoubleAssigmentTruthTableMin'][k2].append(len(A[m]))
                            if len(A) < len(B):
                                finalResults[k][t2]['DoubleAssigmentTruthTableMinPercent'][k2].append(len(A[m]) / len(A))
                            else:
                                finalResults[k][t2]['DoubleAssigmentTruthTableMinPercent'][k2].append(len(A[m]) / len(B))

# We can call this command, e.g., from a Jupyter notebook with init_all=False to get an "empty" experiment wrapper,
# where we can then for instance load a pretrained model to inspect the performance.
@ex.command(unobserved=True)
def get_experiment(init_all=False):
    print('get_experiment')
    experiment = ExperimentWrapper(init_all=init_all)
    return experiment


# This function will be called by default. Note that we could in principle manually pass an experiment instance,
# e.g., obtained by loading a model from the database or by calling this from a Jupyter notebook.
@ex.automain
def train(experiment=None):
    if experiment is None:
        experiment = ExperimentWrapper()
    return experiment.trainExperiment()

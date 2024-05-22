from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM, EigenCAM, EigenGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from captum.attr import IntegratedGradients, Saliency,DeepLift,InputXGradient,GuidedBackprop,GuidedGradCam,FeatureAblation,KernelShap,Deconvolution,FeaturePermutation
import torch
import numpy as np
from modules import helper
import shap
from modules import pytorchTrain as pt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score


#Different LRP options
def getRelpropSaliency(device, data, model, method=None, outputO = None, batchSize=5000):
        outRel = None
        for batch_start in range(0, len(data), batchSize):
            batchEnd = batch_start + batchSize      
            input_ids = torch.from_numpy(data[batch_start:batchEnd]).to(device) 
            input_ids.requires_grad_()
            output = model(input_ids)

            if outputO is None:
                outputOut = output.cpu().data.numpy()#[0]
                index = np.argmax(outputOut, axis=-1)
                one_hot = np.zeros((outputOut.shape[0], outputOut.shape[-1]), dtype=np.float32)
                for h in range(len(one_hot)):
                    one_hot[h, index[h]] = 1
            else:
                outputOut = outputO[batch_start:batchEnd]
                index = np.argmax(outputOut, axis=-1)
                one_hot = outputOut

            one_hot_vector = one_hot
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * output)

            model.zero_grad()
            one_hot.backward(retain_graph=True)
            one_hot.shape
            kwargs = {"alpha": 1}

            if method:
                outRelB = model.relprop(torch.tensor(one_hot_vector).to(input_ids.device), method=method, **kwargs).cpu().detach().numpy()
            else:
                outRelB = model.relprop(torch.tensor(one_hot_vector).to(input_ids.device) , **kwargs).cpu().detach().numpy()
            
            if outRel is None:
                outRel = outRelB
            else:
                outRel = np.vstack([outRel, outRelB])
        return outRel

#Reshapring for some cam methods
def reshape_transform(tensor):
    result = tensor.reshape(tensor.size(0), tensor.size(1), 1, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


#different Saliency methods selection
def interpret_dataset(sal, data, targets, package='PytGradCam', smooth=False):
    if package == 'captum':
        attributions_ig = sal.attribute(data, target=targets)
        return attributions_ig.cpu().squeeze().detach().numpy()
    elif package == 'PytGradCam':
        grayscale_cam = sal(input_tensor=data, targets=targets, eigen_smooth=smooth)

        grayscale_cam = grayscale_cam#[0, :]
        return grayscale_cam


# reduce saliency dimension
def reduceMap(saliencyMap, do3DData=False, do3rdStep=False, axis1= 2, axis2=0, axis3=1, op1='max',op2='sum',op3='sum'):#NOTE op3 is sum
    if do3DData:
        saliencyMap = helper.doCombiStep(op1, saliencyMap, axis1)
        saliencyMap = helper.doCombiStep(op2, saliencyMap, axis2) 
        saliencyMap = helper.doCombiStep(op3, saliencyMap, axis3) 
    elif do3rdStep:
        saliencyMap = helper.doCombiStep(op3, saliencyMap, axis3) 

    return saliencyMap

# split saliency per logic gate
def splitSaliencyPerStack(saliencyStacks, saliency, x_train1, newTrain, y_train1, num_of_classes, nrSymbols, andStack, orStack, xorStack, nrAnds, nrOrs, nrxor, orOffSet, xorOffSet, trueIndexes, do3DData=False, do3rdStep=False):
    preds = y_train1
    targets = np.argmax(preds, axis=1)
    inputIds = x_train1.squeeze()  
    symbolA = helper.getMapValues(nrSymbols)
    symbolA = np.array(symbolA)
    trueSymbols = symbolA[trueIndexes]

    saliency1Ds = reduceMap(saliency, do3DData=do3DData, do3rdStep=do3rdStep)

    if 'and' not in saliencyStacks.keys():
        saliencyStacks['and'] = dict()
        saliencyStacks['or'] = dict()
        saliencyStacks['xor'] = dict()
        for c in range(num_of_classes):
            saliencyStacks['and c' + str(c)] = dict()
            saliencyStacks['or c'+ str(c)] = dict()
            saliencyStacks['xor c'+ str(c)] = dict()
            saliencyStacks['and rank c' + str(c)] = dict()
            saliencyStacks['or rank c'+ str(c)] = dict()
            saliencyStacks['xor rank c'+ str(c)] = dict()
    

    for n in range(len(inputIds)):
        for j in range(andStack):
            combi = str(newTrain[n,nrAnds*j: nrAnds*(j+1)])
            saliencyVs = saliency[n,nrAnds*j: nrAnds*(j+1)]
            saliency1Vs = saliency1Ds[n,nrAnds*j: nrAnds*(j+1)]
            if combi not in saliencyStacks['and'].keys():
                saliencyStacks['and'][combi] = []
                for c in range(num_of_classes):
                    saliencyStacks['and c' + str(c)][combi] = []
                    saliencyStacks['and rank c'+ str(c)][combi] = []
            saliencyStacks['and'][combi].append(saliencyVs)
            saliencyStacks['and c'+str(targets[n])][combi].append(saliencyVs)
            temp = saliency1Vs.argsort()

            ranks = np.empty_like(temp)

            ranks[temp] = np.arange(len(saliency1Vs))
            saliencyStacks['and rank c'+str(targets[n])][combi].append(ranks)

            
        maxAnds = (nrAnds * andStack)

        for j in range(orStack):
            combi = str(newTrain[n,maxAnds + (nrOrs *j - orOffSet * j): maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))])
            saliencyVs = saliency[n,maxAnds + (nrOrs *j - orOffSet * j): maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))]
            saliency1Vs = saliency1Ds[n,maxAnds + (nrOrs *j - orOffSet * j): maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))]
            if combi not in saliencyStacks['or'].keys():
                saliencyStacks['or'][combi] = []
                for c in range(num_of_classes):
                    saliencyStacks['or c'+ str(c)][combi] = []
                    saliencyStacks['or rank c'+ str(c)][combi] = []
            saliencyStacks['or'][combi].append(saliencyVs)
            saliencyStacks['or c'+str(targets[n])][combi].append(saliencyVs)
            temp = saliency1Vs.argsort()

            ranks = np.empty_like(temp)

            ranks[temp] = np.arange(len(saliency1Vs))
            saliencyStacks['or rank c'+str(targets[n])][combi].append(ranks)

        maxOrs = nrOrs * orStack - orOffSet * orStack

        for j in range(xorStack):
            combi = str(newTrain[n,maxAnds+maxOrs+(nrxor *j - xorOffSet * j): maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))])
            saliencyVs = saliency[n,maxAnds+maxOrs+(nrxor *j - xorOffSet * j): maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))]
            saliency1Vs = saliency1Ds[n,maxAnds+maxOrs+(nrxor *j - xorOffSet * j): maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))]
            if combi not in saliencyStacks['xor'].keys():
                saliencyStacks['xor'][combi] = []
                for c in range(num_of_classes):
                    saliencyStacks['xor c'+ str(c)][combi] = []
                    saliencyStacks['xor rank c'+ str(c)][combi] = []


            saliencyStacks['xor'][combi].append(saliencyVs)
            saliencyStacks['xor c'+str(targets[n])][combi].append(saliencyVs)
            temp = saliency1Vs.argsort()

            ranks = np.empty_like(temp)

            ranks[temp] = np.arange(len(saliency1Vs))
            saliencyStacks['xor rank c'+str(targets[n])][combi].append(ranks)
            
    return saliencyStacks

#show how often irrelevant data is more important than any must have inforamtion (Calculating NIB)
def getStackImportanceBreak(nrEmpty, saliencys, x_train1, classes, y_train1, nrSymbols, andStack, orStack, xorStack, nrAnds, nrOrs, nrxor, orOffSet, xorOffSet, trueIndexes, topLevel):
    targets = np.argmax(y_train1, axis=1)
    
    inputIds = x_train1#.squeeze()            
    symbolA = helper.getMapValues(nrSymbols)
    symbolA = np.array(symbolA)
    trueSymbols = symbolA[trueIndexes]       

    brokenImportance = dict()
    brokenStackImportance = dict()

    for c in classes: 
        brokenImportance[int(c)] =  0
        brokenImportance[int(c)] =  0
        brokenStackImportance[int(c)] =  np.zeros(andStack + orStack + xorStack)
        brokenStackImportance[int(c)] =  np.zeros(andStack + orStack + xorStack)
       
    for n in range(len(inputIds)):
        baseline = np.max(saliencys[n][-1* nrEmpty:])
        broken = False
        if targets[n] == 1 and topLevel == 'and':
            for j in range(andStack):
                for k in range((nrAnds*j), nrAnds*(j+1)):
                    if saliencys[n][k] < baseline:
                        brokenStackImportance[targets[n]][j] += 1
                        broken = True
                        break
            maxAnds = (nrAnds * andStack)
            for j in range(orStack):
                maxOr = -1
                orK = 0
                for k in range(maxAnds + (nrOrs *j - orOffSet * j),maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))):
                    if round(float(inputIds[n][k]), 4) in trueSymbols:
                        if saliencys[n][k] > maxOr:
                            maxOr = saliencys[n][k]
                            orK = k
                if maxOr < baseline:
                    brokenStackImportance[targets[n]][j+ andStack] += 1
                    broken = True
            maxOrs = nrOrs * orStack - orOffSet * orStack

            for j in range(xorStack):
                for k in range(maxAnds+maxOrs+(nrxor *j - xorOffSet * j),maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))):
                    if saliencys[n][k] < baseline:
                        broken = True
                        brokenStackImportance[targets[n]][j+ andStack + orStack] += 1
                        break

        elif targets[n] == 0 and topLevel == 'and': #One of the breakpoint conditions must be met!!
            strongestBreak = -1
            breakK = 0
            for j in range(andStack):
                andT = True
                for k in range((nrAnds*j), nrAnds*(j+1)):
                    if round(float(inputIds[n][k]), 4) not in trueSymbols:
                        andT = False
                        if saliencys[n][k] > strongestBreak:
                            strongestBreak = saliencys[n][k]
                            breakK = j
            maxAnds = (nrAnds * andStack)
            for j in range(orStack):
                orT = False
                for k in range(maxAnds + (nrOrs *j - orOffSet * j),maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))):
                    if round(float(inputIds[n][k]), 4) in trueSymbols:
                        orT = True
                        break
                if not orT: #Take min because one is enough to break it!
                    minOr = np.min(saliencys[n][maxAnds + (nrOrs *j - orOffSet * j) : maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))])
                    if minOr > strongestBreak:
                        strongestBreak = minOr
                        breakK = j + andStack
            maxOrs = nrOrs * orStack - orOffSet * orStack

            for j in range(xorStack): #Take min because one is enough to break it!
                xorT = False
                all0s = True
                ones = []
                for k in range(maxAnds+maxOrs+(nrxor *j - xorOffSet * j),maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))):
                    if round(float(inputIds[n][k]), 4) in trueSymbols:
                        ones.append(saliencys[n][k])
                        if all0s:
                            xorT = True
                            all0s = False
                        elif xorT == True:
                            xorT = False
                            
                if not xorT:
                    if all0s:
                        minXOr =np.min(saliencys[n][maxAnds+maxOrs+(nrxor *j - xorOffSet * j) : maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))])
                        if minXOr > strongestBreak:
                            strongestBreak = minXOr
                            breakK = j + andStack + orStack
                    else:
                        maxKs = np.argsort(ones)
                        maxk2 = maxKs[-2]
                        if ones[maxk2] > strongestBreak:
                            strongestBreak = ones[maxk2]
                            breakK = j + andStack + orStack

            if strongestBreak < baseline:
                broken = True
                brokenStackImportance[targets[n]][breakK] += 1



        elif targets[n] == 0 and topLevel == 'or':
            for j in range(andStack):
                maxAnd = -1
                for k in range((nrAnds*j), nrAnds*(j+1)):
                    if round(float(inputIds[n][k]), 4) not in trueSymbols:
                        if saliencys[n][k] > maxAnd:
                            maxAnd = saliencys[n][k]
                if maxAnd < baseline:
                    brokenStackImportance[targets[n]][j] += 1
                    broken = True
            maxAnds = (nrAnds * andStack)
            for j in range(orStack):
                minOr = np.min(saliencys[n][maxAnds + (nrOrs *j - orOffSet * j) : maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))])
                if minOr < baseline:
                    brokenStackImportance[targets[n]][j+ andStack] += 1
                    broken = True
            maxOrs = nrOrs * orStack - orOffSet * orStack

            for j in range(xorStack):
                all0s = True
                ones = []
                for k in range(maxAnds+maxOrs+(nrxor *j - xorOffSet * j),maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))):
                    if round(float(inputIds[n][k]), 4) in trueSymbols:
                        all0s = False
                        ones.append(saliencys[n][k])
                        
                if all0s:
                    minXOr =np.min(saliencys[n][maxAnds+maxOrs+(nrxor *j - xorOffSet * j) : maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))])
                    if minXOr < baseline:
                        broken = True
                        brokenStackImportance[targets[n]][j+ andStack + orStack] += 1
                        
                else:
                    maxKs = np.argsort(ones)
                    maxk2 = maxKs[-2]
                    if ones[maxk2] < baseline:
                        broken = True
                        brokenStackImportance[targets[n]][j+ andStack + orStack] += 1
                        

        elif targets[n] == 1 and topLevel == 'or': #One of the breakpoint conditions must be met!!
            strongestBreak = -1
            breakK = 0
            for j in range(andStack):
                andT = True
                for k in range((nrAnds*j), nrAnds*(j+1)):
                    if round(float(inputIds[n][k]), 4) not in trueSymbols:
                        andT = False
                        break
                if andT:
                    andnMin = np.min(saliencys[n][(nrAnds*j): nrAnds*(j+1)])
                    if andnMin > strongestBreak:
                        strongestBreak = andnMin
                        breakK = j
            maxAnds = (nrAnds * andStack)
            for j in range(orStack):
                biggestOr = -1
                orT = False
                for k in range(maxAnds + (nrOrs *j - orOffSet * j),maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))):
                    if round(float(inputIds[n][k]), 4) in trueSymbols:
                        orT = True
                        if saliencys[n][k] > biggestOr:
                            biggestOr = saliencys[n][k]
                if orT: #Take min because one is enough to break it!
                    if biggestOr > strongestBreak:
                        strongestBreak = biggestOr
                        breakK = j + andStack
            maxOrs = nrOrs * orStack - orOffSet * orStack

            for j in range(xorStack): #Take min because one is enough to break it!
                xorT = False
                for k in range(maxAnds+maxOrs+(nrxor *j - xorOffSet * j),maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))):
                    if round(float(inputIds[n][k]), 4) in trueSymbols:
                        if xorT == True:
                            xorT = False
                            break
                        else:
                            xorT = True
                if xorT:
                    minXOr = np.min(saliencys[n][maxAnds+maxOrs+(nrxor *j - xorOffSet * j) : maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))])
                    if minXOr > strongestBreak:
                        strongestBreak = minXOr
                        breakK = j + andStack + orStack

            if strongestBreak < baseline:
                broken = True
                brokenStackImportance[targets[n]][breakK] += 1


        elif targets[n] == 0 and topLevel == 'xor':
            all0 = True
            maxK1 = -1
            maxK2 = -1
            maxV = -1
            maxV2 = -1
            xorMin = []
            for j in range(andStack):
                andT = True
                andMax= 0 
                for k in range((nrAnds*j), nrAnds*(j+1)):
                    if round(float(inputIds[n][k]), 4) not in trueSymbols:
                        andT = False
                        if all0:
                            if saliencys[n][k] > andMax:
                                andMax = saliencys[n][k]
                        else:
                            break
                        
                if andT:
                    all0 = False
                    andMax = np.min(saliencys[n][(nrAnds*j): nrAnds*(j+1)])

                    if andMax > maxV:
                        maxV2 = maxV
                        maxV = andMax
                        maxK2 = maxK1
                        maxK1 = j
                    elif andMax > maxV2:
                        maxV2 = andMax
                        maxK2 = j
                else:
                    xorMin.append(andMax)
                
            maxAnds = (nrAnds * andStack)
            for j in range(orStack):
                orMax = -1
                orT = False
                for k in range(maxAnds + (nrOrs *j - orOffSet * j),maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))):
                    if round(float(inputIds[n][k]), 4) in trueSymbols:
                        orT = True
                        if saliencys[n][k] > orMax:
                            orMax = saliencys[n][k]

                if orT:
                    all0 = False
                    if orMax > maxV:
                        maxV2 = maxV
                        maxV = orMax
                        maxK2 = maxK1
                        maxK1 = j + andStack
                    elif orMax > maxV2:
                        maxV2 = orMax
                        maxK2 = j + andStack
                else:
                    orMax = np.min(saliencys[n][maxAnds + (nrOrs *j - orOffSet * j): maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))])
                    xorMin.append(orMax)

            maxOrs = nrOrs * orStack - orOffSet * orStack


            for j in range(xorStack): #Take min because one is enough to break it!
                xorT = False
                internHighest = -1
                internNdHighest = -1
                all0Intern = True

                for k in range(maxAnds+maxOrs+(nrxor *j - xorOffSet * j),maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))):
                    if round(float(inputIds[n][k]), 4) in trueSymbols:
                        if saliencys[n][k] > internHighest:
                            internHighest = internNdHighest
                            internHighest = saliencys[n][k]
                        elif saliencys[n][k] > internNdHighest:
                            internNdHighest = saliencys[n][k]

                        if all0Intern:
                            all0Intern = False
                            xorT = True
                        elif xorT == True:
                            xorT = False


                if xorT:
                    all0 = False
                    minXOr = np.min(saliencys[n][maxAnds+maxOrs+(nrxor *j - xorOffSet * j) : maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))])
                    if minXOr > maxV:
                        maxV2 = maxV
                        maxV = minXOr
                        maxK2 = maxK1
                        maxK1 = j + andStack + orStack
                    elif minXOr > maxV2:
                        maxV2 = minXOr
                        maxK2 = j + andStack + orStack
                else:
                    
                    if all0Intern:
                        minXOr = np.min(saliencys[n][maxAnds+maxOrs+(nrxor *j - xorOffSet * j) : maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))])
                    else:
                        minXOr = internNdHighest
                    xorMin.append(minXOr)




            if all0:
                for j, v in enumerate(xorMin):
                    if v < baseline:
                        brokenStackImportance[targets[n]][j] += 1
                        broken = True

            else:
                if maxV2 < baseline:
                    brokenStackImportance[targets[n]][maxK2] += 1
                    brokenStackImportance[targets[n]][maxK1] += 1
                    broken = True


        elif targets[n] == 1 and topLevel == 'xor': #One of the breakpoint conditions must be met!!
            for j in range(andStack):
                andMax = -1
                andT = True
                for k in range((nrAnds*j), nrAnds*(j+1)):
                    if round(float(inputIds[n][k]), 4) not in trueSymbols:
                        andT = False
                        if saliencys[n][k] > andMax:
                            andMax = saliencys[n][k]
                if andT:
                    andMax = np.min(saliencys[n][(nrAnds*j): nrAnds*(j+1)])

                if andMax < baseline:
                    brokenStackImportance[targets[n]][j] += 1
                    broken = True
                    
                
            maxAnds = (nrAnds * andStack)
            for j in range(orStack):
                orMax = -1
                orT = False
                for k in range(maxAnds + (nrOrs *j - orOffSet * j),maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))):
                    if round(float(inputIds[n][k]), 4) in trueSymbols:
                        orT = True
                        if saliencys[n][k] > orMax:
                            orMax = saliencys[n][k]
                if not orT: #Take min because one is enough to break it!
                    orMax = np.min(saliencys[n][maxAnds + (nrOrs *j - orOffSet * j): maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))])
                
                if orMax < baseline:
                    brokenStackImportance[targets[n]][j + andStack] += 1
                    broken = True
                    
            maxOrs = nrOrs * orStack - orOffSet * orStack


            for j in range(xorStack): #Take min because one is enough to break it!
                xorT = False
                all0Intern = True
                ones = []
                for k in range(maxAnds+maxOrs+(nrxor *j - xorOffSet * j),maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))):
                    if round(float(inputIds[n][k]), 4) in trueSymbols:
                        ones.append(saliencys[n][k])
                        if all0Intern:
                            all0Intern= False
                            xorT = True
                        elif xorT:
                            xorT = False

                if xorT or all0Intern:
                    minXOr = np.min(saliencys[n][maxAnds+maxOrs+(nrxor *j - xorOffSet * j) : maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))])
                else:
                    maxKs = np.argsort(ones)
                    maxk2 = maxKs[-2]
                    minXOr = ones[maxk2]

                if minXOr < baseline:
                    brokenStackImportance[targets[n]][j + andStack + orStack] += 1
                    broken = True
                    

        if broken:
            brokenImportance[targets[n]] += 1

    for k in brokenImportance.keys():
        if brokenImportance[k] == 0:
            brokenStackImportance[k] = 0
        else:
            brokenStackImportance[k] = brokenStackImportance[k]/brokenImportance[k]
        brokenImportance[k] = brokenImportance[k]/np.sum(targets == k)

    return brokenImportance, brokenStackImportance

#gives how often inputs with information is below the irrelevant data (Calculating GIB)
def getPredictionSaliency(nrEmpty, saliencys, x_train1, classes, y_train1, nrSymbols, andStack, orStack, xorStack, nrAnds, nrOrs, nrxor, orOffSet, xorOffSet, trueIndexes):
    targets = np.argmax(y_train1, axis=1)
    
    inputIds = x_train1.squeeze()            
    symbolA = helper.getMapValues(nrSymbols)
    symbolA = np.array(symbolA)
    trueSymbols = symbolA[trueIndexes]       

    wrongImportanceMeaning = dict() 
    countImportanceMeaning = dict() 
    for c in classes:
        wrongImportanceMeaning[int(c)] =  np.zeros(len(x_train1[0]))
        countImportanceMeaning[int(c)] =  np.zeros(len(x_train1[0]))
       
            
    for n in range(len(inputIds)):
        baseline = np.max(saliencys[n][-1* nrEmpty:])
        if targets[n] == 0:
            for j in range(andStack):
                for k in range((nrAnds*j), nrAnds*(j+1)):
                    if round(float(inputIds[n][k]), 4) not in trueSymbols:
                        if saliencys[n][k] < baseline:
                            wrongImportanceMeaning[targets[n]][k] += 1
                        countImportanceMeaning[targets[n]][k] += 1
        if targets[n] == 1:
            for j in range(andStack):
                for k in range((nrAnds*j), nrAnds*(j+1)):
                    if round(float(inputIds[n][k]), 4) in trueSymbols:
                        if saliencys[n][k] < baseline:
                            wrongImportanceMeaning[targets[n]][k] += 1
                        countImportanceMeaning[targets[n]][k] += 1

        maxAnds = (nrAnds * andStack)


        if targets[n] == 1:
            for j in range(orStack):
                for k in range(maxAnds + (nrOrs *j - orOffSet * j),maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))):
                    if round(float(inputIds[n][k]), 4) in trueSymbols:
                        if saliencys[n][k] < baseline:
                            wrongImportanceMeaning[targets[n]][k] += 1
                        countImportanceMeaning[targets[n]][k] += 1
                        
        if targets[n] == 0:
            for j in range(orStack):
                for k in range(maxAnds + (nrOrs *j - orOffSet * j),maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))):
                    if round(float(inputIds[n][k]), 4) not in trueSymbols:
                        if saliencys[n][k] < baseline:
                            wrongImportanceMeaning[targets[n]][k] += 1
                        countImportanceMeaning[targets[n]][k] += 1

        maxOrs = nrOrs * orStack - orOffSet * orStack


        for j in range(xorStack):
            all0 = True
            highest = 0
            ndHighest = 0
            k1 = -1
            k2 = -1
            if targets[n] == 0:
                for k in range(maxAnds+maxOrs+(nrxor *j - xorOffSet * j),maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))):
                    if round(float(inputIds[n][k]), 4) not in trueSymbols:
                        all0 = False
                        break
                    if saliencys[n][k] > highest:
                        ndHighest = highest
                        highest=saliencys[n][k]
                        k2 = k1
                        k1 = k
                    elif saliencys[n][k] > ndHighest:
                        ndHighest = saliencys[n][k]
                        k2 = k

                if not all0:
                    if saliencys[n][k1] < baseline:
                        wrongImportanceMeaning[targets[n]][k1] += 1
                    countImportanceMeaning[targets[n]][k1] += 1  
                    if saliencys[n][k2] < baseline:
                        wrongImportanceMeaning[targets[n]][k2] += 1
                    countImportanceMeaning[targets[n]][k2] += 1       

            for k in range(maxAnds+maxOrs+(nrxor *j - xorOffSet * j),maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))):
                if targets[n] == 1:
                    if saliencys[n][k] < baseline:
                        wrongImportanceMeaning[targets[n]][k] += 1
                    countImportanceMeaning[targets[n]][k] += 1       
                elif targets[n] == 0: 
                    if all0:
                        if saliencys[n][k] < baseline:
                            wrongImportanceMeaning[targets[n]][k] += 1
                        countImportanceMeaning[targets[n]][k] += 1
                    else:
                        break      

    return wrongImportanceMeaning, countImportanceMeaning

# Calc DCAs
def getPredictionMaps(device, model, x_train1, newTrain, y_train1, num_of_classes, toplevel, nrSymbols, andStack, orStack, xorStack, nrAnds, nrOrs, nrxor, orOffSet, xorOffSet, trueIndexes, val_batch_size=1000, batch_size=1000):
    preds = y_train1
    
    targets = np.argmax(preds, axis=1)
    valueMap = dict()
    conValueMap = dict()
    classConValueMap = dict()
    minConValueMap = dict()
    inputIds = x_train1#.squeeze()            
    for c in range(num_of_classes):
        
        for j in range(andStack):
            b = []
            for i, a in enumerate(newTrain[:,nrAnds*j: nrAnds*(j+1)]):
                if targets[i] == c and -2 in a:
                        b.append(a)
            if 'and'+str(j) not in valueMap.keys():
                valueMap['and'+str(j)] = dict()
                conValueMap['and'+str(j)] = dict()
                minConValueMap['and'+str(j)] = dict()
                classConValueMap['and'+str(j)] = dict()
            valueMap['and'+str(j)][c] =  np.unique(np.array(b), axis=0)
            conValueMap['and'+str(j)][c] = []
            minConValueMap['and'+str(j)][c] = []
            classConValueMap['and'+str(j)][c] = dict()
            for c2 in range(num_of_classes):
                classConValueMap['and'+str(j)][c][c2] = []
            
        maxAnds = (nrAnds * andStack)

        for j in range(orStack):
            b = []
            for i, a in enumerate(newTrain[:,maxAnds + (nrOrs *j - orOffSet): maxAnds+(nrOrs *(j+1) - orOffSet)]):
                if targets[i] == c and -2 in a:
                    b.append(a)
            if 'or'+str(j) not in valueMap.keys():
                valueMap['or'+str(j)] = dict()
                conValueMap['or'+str(j)] = dict()
                minConValueMap['or'+str(j)] = dict()
                classConValueMap['or'+str(j)] = dict()
            valueMap['or'+str(j)][c] =  np.unique(np.array(b), axis=0)
            conValueMap['or'+str(j)][c] = []
            minConValueMap['or'+str(j)][c] = []
            classConValueMap['or'+str(j)][c] = dict()
            for c2 in range(num_of_classes):
                classConValueMap['or'+str(j)][c][c2] = []

        maxOrs = nrOrs * orStack - orOffSet * orStack
        for j in range(xorStack):
            b = []
            for i, a in enumerate(newTrain[:,maxAnds+maxOrs+(nrxor *j - xorOffSet * j): maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet *(j+1))]):
                if targets[i] == c and -2 in a:
                    b.append(a)
            if 'xor'+str(j) not in valueMap.keys():
                valueMap['xor'+str(j)] = dict()
                conValueMap['xor'+str(j)] = dict()
                minConValueMap['xor'+str(j)] = dict()
                classConValueMap['xor'+str(j)] = dict()
            valueMap['xor'+str(j)][c] =  np.unique(np.array(b), axis=0)
            conValueMap['xor'+str(j)][c] = []
            minConValueMap['xor'+str(j)][c] = []
            classConValueMap['xor'+str(j)][c] = dict()
            for c2 in range(num_of_classes):
                classConValueMap['xor'+str(j)][c][c2] = []
            
            
    symbolA = helper.getMapValues(nrSymbols)
    symbolA = np.array(symbolA)
    trueSymbols = symbolA[trueIndexes]       
            
    for n in range(len(inputIds)):
        andTs = []
        for j in range(andStack):
            andT = True
            for k in range((nrAnds*j), nrAnds*(j+1)):
                if round(float(inputIds[n][k]), 4) not in trueSymbols:
                    andT = False
                    break
            if nrAnds == 0:
                andT = True
            if andT:
                conValueMap['and'+str(j)][1].append(newTrain[n,nrAnds*j: nrAnds*(j+1)])
                classConValueMap['and'+str(j)][1][targets[n]].append(newTrain[n,nrAnds*j: nrAnds*(j+1)])
            else:
                conValueMap['and'+str(j)][0].append(newTrain[n,nrAnds*j: nrAnds*(j+1)])
                classConValueMap['and'+str(j)][0][targets[n]].append(newTrain[n,nrAnds*j: nrAnds*(j+1)])

            
            andTs.append(andT)
        maxAnds = (nrAnds * andStack)

        orTs = []
        for j in range(orStack):
            orT = False
            for k in range(maxAnds + (nrOrs *j - orOffSet * j),maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))):
                if round(float(inputIds[n][k]), 4) in trueSymbols:
                    orT = True
                    break
            if nrOrs == 0:
                orT = True
            if orT:
                conValueMap['or'+str(j)][1].append(newTrain[n,maxAnds + (nrOrs *j - orOffSet * j): maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))])
                classConValueMap['or'+str(j)][1][targets[n]].append(newTrain[n,maxAnds + (nrOrs *j - orOffSet * j): maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))])

            else:
                conValueMap['or'+str(j)][0].append(newTrain[n,maxAnds + (nrOrs *j - orOffSet * j): maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))])
                classConValueMap['or'+str(j)][0][targets[n]].append(newTrain[n,maxAnds + (nrOrs *j - orOffSet * j): maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))])


            orTs.append(orT)
        maxOrs = nrOrs * orStack - orOffSet * orStack

        xorTs = []
        for j in range(xorStack):
            xorT = False
            for k in range(maxAnds+maxOrs+(nrxor *j - xorOffSet * j),maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))):
                if round(float(inputIds[n][k]), 4) in trueSymbols:
                    if xorT == True:
                        xorT = False
                        break
                    else:
                        xorT = True
            if nrxor == 0:
                xorT = True
            if xorT:
                conValueMap['xor'+str(j)][1].append(newTrain[n,maxAnds+maxOrs+(nrxor *j - xorOffSet * j): maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))])
                classConValueMap['xor'+str(j)][1][targets[n]].append(newTrain[n,maxAnds+maxOrs+(nrxor *j - xorOffSet * j): maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))])

            else: 
                conValueMap['xor'+str(j)][0].append(newTrain[n,maxAnds+maxOrs+(nrxor *j - xorOffSet * j): maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))])
                classConValueMap['xor'+str(j)][0][targets[n]].append(newTrain[n,maxAnds+maxOrs+(nrxor *j - xorOffSet * j): maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))])

            xorTs.append(xorT)
         
        if (toplevel == 'and' and targets[n] == 1) or (toplevel == 'or' and targets[n] == 0):
            for k3 in conValueMap.keys():
                for j in range(len(andTs)):
                        minConValueMap['and'+str(j)][targets[n]].append(conValueMap['and'+str(j)][andTs[j]][-1])
                for j in range(len(orTs)):
                        minConValueMap['or'+str(j)][targets[n]].append(conValueMap['or'+str(j)][orTs[j]][-1])
                for j in range(len(xorTs)):
                        minConValueMap['xor'+str(j)][targets[n]].append(conValueMap['xor'+str(j)][xorTs[j]][-1])
        elif (toplevel == 'and' and targets[n] == 0) or (toplevel == 'or' and targets[n] == 1):
            if (np.sum(andTs == targets[n]) + np.sum(orTs == targets[n]) + np.sum(xorTs == targets[n])) == 1:
                for j in range(len(andTs)):
                    if andTs[j] == targets[n]:
                        minConValueMap['and'+str(j)][targets[n]].append(conValueMap['and'+str(j)][andTs[j]][-1])
                for j in range(len(orTs)):
                    if orTs[j] == targets[n]:
                        minConValueMap['or'+str(j)][targets[n]].append(conValueMap['or'+str(j)][orTs[j]][-1])
                for j in range(len(xorTs)):
                    if xorTs[j] == targets[n]:
                        minConValueMap['xor'+str(j)][targets[n]].append(conValueMap['xor'+str(j)][xorTs[j]][-1])
        elif (toplevel == 'xor' and targets[n] == 1):
            for j in range(len(andTs)):
                    minConValueMap['and'+str(j)][andTs[j]].append(conValueMap['and'+str(j)][andTs[j]][-1])
            for j in range(len(orTs)):
                    minConValueMap['or'+str(j)][orTs[j]].append(conValueMap['or'+str(j)][orTs[j]][-1])
            for j in range(len(xorTs)):
                    minConValueMap['xor'+str(j)][xorTs[j]].append(conValueMap['xor'+str(j)][xorTs[j]][-1])
        elif (toplevel == 'xor' and targets[n] == 0):
            if (np.sum(andTs == targets[n]) + np.sum(orTs == targets[n]) + np.sum(xorTs == targets[n])) == (nrAnds + nrOrs + nrxor):
                for j in range(len(andTs)):
                    minConValueMap['and'+str(j)][targets[n]].append(conValueMap['and'+str(j)][andTs[j]][-1])
                for j in range(len(orTs)):
                    minConValueMap['or'+str(j)][targets[n]].append(conValueMap['or'+str(j)][orTs[j]][-1])
                for j in range(len(xorTs)):
                    minConValueMap['xor'+str(j)][targets[n]].append(conValueMap['xor'+str(j)][xorTs[j]][-1])
            elif(np.sum(andTs == 1) + np.sum(orTs == 1) + np.sum(xorTs == 1)) == 2:
                for j in range(len(andTs)):
                    if andTs[j] == 1:
                        minConValueMap['and'+str(j)][1].append(conValueMap['and'+str(j)][andTs[j]][-1])
                for j in range(len(orTs)):
                    if orTs[j] == 1:
                        minConValueMap['or'+str(j)][1].append(conValueMap['or'+str(j)][orTs[j]][-1])
                for j in range(len(xorTs)):
                    if xorTs[j] == 1:
                        minConValueMap['xor'+str(j)][1].append(conValueMap['xor'+str(j)][xorTs[j]][-1])

    return valueMap, conValueMap, classConValueMap, minConValueMap

# calc statistical logical acc!
def logicAccGuess(data, target, nrSymbols, topLevel, andStack = 1, orStack = 1, xorStack = 1, nrAnds = 2, nrOrs = 2, nrxor = 2, trueIndexes=[3,1], orOffSet=0, xorOffSet=0):
    predCur = []
    symbolA = helper.getMapValues(nrSymbols)
    symbolA = np.array(symbolA)
    trueSymbols = symbolA[trueIndexes]
    falseSymbols = np.delete(symbolA, trueIndexes)

    for s in data:

        andTs = []
        for j in range(andStack):
            maskCOunt = 0
            andT = 1
            for k in range((nrAnds*j), nrAnds*(j+1)):
                if s[k] == -2 and (andT == 1 or andT == -1):
                    andT = -1
                    maskCOunt += 1
                elif round(float(s[k]), 4) not in trueSymbols:
                    andT = 0
                    break
            if nrAnds == 0:
                andT = 1
            if maskCOunt >= 2:
                andT = 0
            andTs.append(andT)
        maxAnds = (nrAnds * andStack)

        orTs = []
        for j in range(orStack):
            orT = 0
            maskCOunt = 0
            for k in range(maxAnds + (nrOrs *j - orOffSet * j),maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))):
                if s[k] == -2:
                    maskCOunt += 1
                if round(float(s[k]), 4) in trueSymbols:
                    orT = 1
                    break
                elif s[k] == -2 and orT==0:
                    orT = -1
            if nrOrs == 0:
                orT = 1
            if maskCOunt >= 2:
                orT = 1
            orTs.append(orT)
        maxOrs = nrOrs * orStack - orOffSet * orStack

        xorTs = []
        for j in range(xorStack):
            xorT = 0
            maskCOunt = 0
            broken = False
            for k in range(maxAnds+maxOrs+(nrxor *j - xorOffSet * j),maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))):
                if round(float(s[k]), 4) in trueSymbols:
                    if xorT == 1:
                        xorT = 0
                        broken = False
                        break
                    else:
                        xorT = 1
                elif s[k] == -2:
                    broken = True
                    maskCOunt += 1
            if nrxor == 0:
                xorT = 1
            if broken:
                if maskCOunt >= 3:
                    xorT = 0
                else:
                    xorT = -1
            xorTs.append(xorT)

        andTs = np.array(andTs)
        orTs = np.array(orTs)
        xorTs = np.array(xorTs)


        if topLevel == 'and':
            if (np.sum(andTs == 0) + np.sum(orTs == 0) + np.sum(xorTs == 0)) > 0:
                    predCur.append(0)
            elif (np.sum(andTs == -1) + np.sum(orTs == -1) + np.sum(xorTs == -1)) >= 2:
                predCur.append(0)
            elif (np.sum(andTs == -1) + np.sum(orTs==-1) + np.sum(xorTs==-1)) > 0:
                predCur.append(-1)
            else:
                predCur.append(1)
        elif topLevel == 'or':
            if (np.sum([np.sum(andTs == 1), np.sum(orTs == 1), np.sum(xorTs == 1)]) >= 1):
                predCur.append(1)
            elif (np.sum(andTs == -1) + np.sum(orTs == -1) + np.sum(xorTs == -1)) >= 2:
                predCur.append(1)
            elif (np.sum(andTs == -1) + np.sum(orTs==-1) + np.sum(xorTs==-1)) > 0:
                predCur.append(-1)
            else:
                predCur.append(0)
        elif topLevel == 'xor':
            if (np.sum(andTs == 1) + np.sum(orTs == 1) + np.sum(xorTs == 1) == 2):
                predCur.append(0)
            elif (np.sum(andTs == -1) + np.sum(orTs == -1) + np.sum(xorTs == -1)) >= 3:
                predCur.append(0)
            elif (np.sum(andTs == -1) + np.sum(orTs==-1) + np.sum(xorTs==-1)) > 0:
                predCur.append(-1)
            elif (np.sum(andTs == 1) + np.sum(orTs == 1) + np.sum(xorTs == 1) == 1):
                predCur.append(1)
            else:
                predCur.append(0)
        else:
                raise ValueError("Need a valid top level")

    acc = accuracy_score(np.array(predCur),target.argmax(axis=1))
    return acc, predCur

#calc logical acc 
def logicAcc(data, target, nrSymbols, topLevel, andStack = 1, orStack = 1, xorStack = 1, nrAnds = 2, nrOrs = 2, nrxor = 2, trueIndexes=[3,1], orOffSet=0, xorOffSet=0):
    predCur = []
    symbolA = helper.getMapValues(nrSymbols)
    symbolA = np.array(symbolA)
    trueSymbols = symbolA[trueIndexes]
    falseSymbols = np.delete(symbolA, trueIndexes)

    for s in data:

        andTs = []
        for j in range(andStack):
            andT = 1
            for k in range((nrAnds*j), nrAnds*(j+1)):
                if s[k] == -2 and (andT == 1 or andT == -1):
                    andT = -1
                elif round(float(s[k]), 4) not in trueSymbols:
                    andT = 0
                    break
            if nrAnds == 0:
                andT = 1
            andTs.append(andT)
        maxAnds = (nrAnds * andStack)

        orTs = []
        for j in range(orStack):
            orT = 0
            for k in range(maxAnds + (nrOrs *j - orOffSet * j),maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))):
                if round(float(s[k]), 4) in trueSymbols:
                    orT = 1
                    break
                elif s[k] == -2 and orT==0:
                    orT = -1
            if nrOrs == 0:
                orT = 1
            orTs.append(orT)
        maxOrs = nrOrs * orStack - orOffSet * orStack

        xorTs = []
        for j in range(xorStack):
            xorT = 0
            broken = False
            for k in range(maxAnds+maxOrs+(nrxor *j - xorOffSet * j),maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))):
                if round(float(s[k]), 4) in trueSymbols:
                    if xorT == 1:
                        xorT = 0
                        broken = False
                        break
                    else:
                        xorT = 1
                elif s[k] == -2:
                    broken = True
            if nrxor == 0:
                xorT = 1
            if broken:
                xorT = -1
            xorTs.append(xorT)

        andTs = np.array(andTs)
        orTs = np.array(orTs)
        xorTs = np.array(xorTs)

        
        if topLevel == 'and':
            if (np.sum(andTs == 0) + np.sum(orTs == 0) + np.sum(xorTs == 0)) > 0:
                    predCur.append(0)
            elif (np.sum(andTs == -1) + np.sum(orTs==-1) + np.sum(xorTs==-1)) > 0:
                predCur.append(-1)
            else:
                predCur.append(1)
        elif topLevel == 'or':
            if (np.sum([np.sum(andTs == 1), np.sum(orTs == 1), np.sum(xorTs == 1)]) >= 1):
                predCur.append(1)
            elif (np.sum(andTs == -1) + np.sum(orTs==-1) + np.sum(xorTs==-1)) > 0:
                predCur.append(-1)
            else:
                predCur.append(0)
        elif topLevel == 'xor':
            if (np.sum(andTs == 1) + np.sum(orTs == 1) + np.sum(xorTs == 1) == 2):
                predCur.append(0)
            elif (np.sum(andTs == -1) + np.sum(orTs==-1) + np.sum(xorTs==-1)) > 0:
                predCur.append(-1)
            elif (np.sum(andTs == 1) + np.sum(orTs == 1) + np.sum(xorTs == 1) == 1):
                predCur.append(1)
            else:
                predCur.append(0)
        else:
                raise ValueError("Need a valid top level")

    acc = accuracy_score(np.array(predCur),target.argmax(axis=1))
    return acc, predCur

#Get transformer attention
def getAttention(device, model, x_train1, y_train1, val_batch_size=1000, batch_size=1000, doClsTocken=False):
        attention = pt.validate_model(device, model, x_train1, y_train1, val_batch_size, batch_size, '', output_attentions=True)
        fullAttention = []
        for n in range(model.num_hidden_layers):
            fullAttention.append([])
        for a in attention:
                for layers in range(len(list(a))):
                        fullAttention[layers] = fullAttention[layers] + list(a[layers])


        def makecpu(inputList):
                if (isinstance(inputList, list)):
                        for a in range(len(inputList)):
                                inputList[a] = np.array(makecpu(inputList[a].cpu()))
                        return np.array(inputList)
                elif (isinstance(inputList, tuple)):
                        for a in range(len(inputList)):
                                inputList[a] = np.array(makecpu(inputList[a].cpu()))
                        return np.array(inputList)
                elif (isinstance(inputList, torch.Tensor)):
                        return inputList.cpu()
                else:
                        return inputList.cpu()

        for a in range(len(fullAttention)):
            fullAttention[a]= np.array(makecpu(fullAttention[a]))
        
        if doClsTocken:
            fullAttention = np.array(fullAttention)[:, 0, 1:]
        else:
            fullAttention = np.array(fullAttention)
        return fullAttention

#Method to get one out of multiple saliency maps!
def getSaliencyMap(outMap, saveKey, device, numberOfLables, modelType: str, method: str, submethod: str, model, x_train, x_val, x_test, y_train, y_val, y_test, smooth=False, batches=True, batchSize=500, doClassBased=True):
    outTrain = []
    outVal = []
    outTest = []
    print('methods:')
    print(method)
    print(submethod)

     #argsV = np.argwhere(np.argmax(y_train1, axis=1)==1).flatten()
    if not batches:
        batchSize = len(y_train)
    
    if method == 'LRP':
            outTrain = getRelpropSaliency(device, x_train, model, method=submethod, batchSize=batchSize)
            outVal = getRelpropSaliency(device, x_val, model, method=submethod, batchSize=batchSize)
            outTest = getRelpropSaliency(device, x_test, model, method=submethod, batchSize=batchSize)
            for lable in range(numberOfLables):
                targets = np.zeros((x_train.shape[0], numberOfLables), dtype=np.float32)
                for t in range(len(targets)):
                    targets[t, lable] = 1
                outTrainC = getRelpropSaliency(device, x_train, model, method=submethod, outputO=targets, batchSize=batchSize)
                targets = np.zeros((x_val.shape[0], numberOfLables), dtype=np.float32)
                for t in range(len(targets)):
                    targets[t, lable] = 1
                outValC = getRelpropSaliency(device, x_val, model, method=submethod, outputO=targets, batchSize=batchSize)
                targets = np.zeros((x_test.shape[0], numberOfLables), dtype=np.float32)
                for t in range(len(targets)):
                    targets[t, lable] = 1
                outTestC = getRelpropSaliency(device, x_test, model, method=submethod, outputO=targets, batchSize=batchSize)
                outMap['classes'][str(lable)][saveKey + 'Train'].append(outTrainC)
                outMap['classes'][str(lable)][saveKey + 'Val'].append(outValC)
                outMap['classes'][str(lable)][saveKey + 'Test'].append(outTestC)

    elif method ==  'captum':
            if submethod == "IntegratedGradients":
                    lig = IntegratedGradients(model)
            elif submethod == "Saliency":
                    lig = Saliency(model)
            elif submethod == "DeepLift":
                    lig = DeepLift(model)
            elif submethod == "KernelShap":
                    lig = KernelShap(model)
            elif submethod == "InputXGradient":
                    lig = InputXGradient(model)
            elif submethod == "GuidedBackprop":
                    lig = GuidedBackprop(model)
            elif submethod == "GuidedGradCam":
                    if modelType == "Transformer":
                        lig = GuidedGradCam(model, layer=model.encoder.layer[-1])
                    elif modelType == "CNN":
                        lig = GuidedGradCam(model, layer=model.lastConv)
                    else:
                        raise ValueError("Not a valid model type for gradcam")
            elif submethod == "FeatureAblation":
                    lig = FeatureAblation(model)
            elif submethod == "FeaturePermutation":
                    lig = FeaturePermutation(model)
            elif submethod == "Deconvolution":
                    lig = Deconvolution(model)
            else:
                    raise ValueError("Not a valid captum submethod")

            if doClassBased:
                maxGoal = numberOfLables
            else:
                maxGoal = 0
            for lable in range(-1, maxGoal):
                outTrainA = None
                outValA = None
                outTestA = None

                for batch_start in range(0, len(y_train), batchSize):
                    batchEnd = batch_start + batchSize
                    if lable == -1:           
                        targets = torch.from_numpy(y_train[batch_start:batchEnd].argmax(axis=1)).to(device) 
                    else:
                        targets = torch.from_numpy(np.zeros((len(y_train[batch_start:batchEnd])), dtype=np.int64) + lable).to(device) 
                    input_ids = torch.from_numpy(x_train[batch_start:batchEnd]).to(device) 

                    outTrainB = interpret_dataset(lig, input_ids, targets, package=method, smooth=smooth)
                    if outTrainA is None:
                        outTrainA = outTrainB
                    else:
                        outTrainA = np.vstack([outTrainA,outTrainB])

                for batch_start in range(0, len(y_val), batchSize):
                    batchEnd = batch_start + batchSize
                    if lable == -1:           
                        targets = torch.from_numpy(y_val[batch_start:batchEnd].argmax(axis=1)).to(device) 
                    else:
                        targets = torch.from_numpy(np.zeros((len(y_val[batch_start:batchEnd])), dtype=np.int64) + lable).to(device) 
                    input_ids = torch.from_numpy(x_val[batch_start:batchEnd]).to(device) 
                    outValB = interpret_dataset(lig, input_ids, targets, package=method, smooth=smooth)
                    if outValA is None:
                        outValA = outValB
                    else:
                        outValA = np.vstack([outValA,outValB])


                for batch_start in range(0, len(y_test), batchSize):
                    batchEnd = batch_start + batchSize
                    if lable == -1:           
                        targets = torch.from_numpy(y_test[batch_start:batchEnd].argmax(axis=1)).to(device) 
                    else:
                        targets = torch.from_numpy(np.zeros((len(y_test[batch_start:batchEnd])), dtype=np.int64) + lable).to(device) 
                    input_ids = torch.from_numpy(x_test[batch_start:batchEnd]).to(device) 
                    outTestb = interpret_dataset(lig, input_ids, targets, package=method, smooth=smooth)
                    if outTestA is None:
                        outTestA = outTestb
                    else:
                        outTestA = np.vstack([outTestA, outTestb])
                if lable == -1:
                    outTrain = outTrainA
                    outVal = outValA
                    outTest = outTestA
                else:
                    outMap['classes'][str(lable)][saveKey + 'Train'].append(outTrainA)
                    outMap['classes'][str(lable)][saveKey + 'Val'].append(outValA)
                    outMap['classes'][str(lable)][saveKey + 'Test'].append(outTestA)

    elif method ==  'PytGradCam':
            if modelType == "Transformer":
                layer=model.encoder.layer[-1]
                reshapeMethod = reshape_transform
            elif modelType == "CNN":
                layer=model.lastConv
                reshapeMethod = None
            else:
                raise ValueError("Not a valid model type for PytGradCam")

            if submethod == "EigenCAM":
                    lig = EigenCAM(model, target_layers=[layer], use_cuda=True, reshape_transform=reshapeMethod)
            elif submethod == "GradCAMPlusPlus":
                    lig = GradCAMPlusPlus(model, target_layers=[layer], use_cuda=True, reshape_transform=reshapeMethod)
            elif submethod == "XGradCAM":
                    lig = XGradCAM(model, target_layers=[layer], use_cuda=True, reshape_transform=reshapeMethod)
            elif submethod == "GradCAM":
                    lig = GradCAM(model, target_layers=[layer], use_cuda=True, reshape_transform=reshapeMethod)
            elif submethod == "EigenGradCAM":
                    lig = EigenGradCAM(model, target_layers=[layer], use_cuda=True, reshape_transform=reshapeMethod)

            else:
                    raise ValueError("Not a valid PytGradCam submethod")

            if not batches:
                batchSize = len(y_train)

            for lable in range(-1, numberOfLables):

                outTrainA = None
                outValA = None
                outTestA = None
                for batch_start in range(0, len(y_train), batchSize):
                    batchEnd = batch_start + batchSize
                    input_ids = torch.from_numpy(x_train[batch_start:batchEnd]).to(device) 

                    
                    target_categories = torch.from_numpy(y_train[batch_start:batchEnd].argmax(axis=1)).to(device) 
                    if lable != -1:           
                        target_categories = target_categories * 0 + lable


                    target_categories = target_categories.squeeze()
                    targets = [ClassifierOutputTarget(category) for category in target_categories]

                    outTrainB = interpret_dataset(lig, input_ids, targets, package=method, smooth=smooth)
                    if outTrainA is None:
                        outTrainA = outTrainB
                    else:
                        outTrainA = np.vstack([outTrainA,outTrainB])

                for batch_start in range(0, len(y_val), batchSize):
                    batchEnd = batch_start + batchSize
                    target_categories = torch.from_numpy(y_val[batch_start:batchEnd].argmax(axis=1)).to(device) 
                    if lable != -1:           
                        target_categories = target_categories * 0 + lable

                    target_categories = target_categories.squeeze()
                    targets = [ClassifierOutputTarget(category) for category in target_categories]

                    input_ids = torch.from_numpy(x_val[batch_start:batchEnd]).to(device) 
                    
                    outValB = interpret_dataset(lig, input_ids, targets, package=method, smooth=smooth)
                    if outValA is None:
                        outValA = outValB
                    else:
                        outValA = np.vstack([outValA,outValB])

                for batch_start in range(0, len(y_test), batchSize):
                    batchEnd = batch_start + batchSize
                    target_categories = torch.from_numpy(y_test[batch_start:batchEnd].argmax(axis=1)).to(device) 
                    if lable != -1:           
                        target_categories = target_categories * 0 + lable

                    target_categories = target_categories.squeeze()
                    targets = [ClassifierOutputTarget(category) for category in target_categories]

                    input_ids = torch.from_numpy(x_test[batch_start:batchEnd]).to(device) 

                    
                    outTestb = interpret_dataset(lig, input_ids, targets, package=method, smooth=smooth)
                    if outTestA is None:
                        outTestA = outTestb
                    else:
                        outTestA = np.vstack([outTestA, outTestb])
                if lable == -1:
                    outTrain = outTrainA
                    outVal = outValA
                    outTest = outTestA
                else:
                    outMap['classes'][str(lable)][saveKey + 'Train'].append(outTrainA)
                    outMap['classes'][str(lable)][saveKey + 'Val'].append(outValA)
                    outMap['classes'][str(lable)][saveKey + 'Test'].append(outTestA)


    elif method ==  'SHAP':
            explainer = shap.TreeExplainer(model, x_train)
            outTrain = explainer.shap_values(x_train)
            
            explainer = shap.TreeExplainer(model, x_val)
            outVal = explainer.shap_values(x_val)
            
            explainer = shap.TreeExplainer(model, x_test)
            outTest = explainer.shap_values(x_test)
            
    elif method == 'Attention':

            outTrain = getAttention(device, model, x_val, y_val)
            outVal = getAttention(device,model, x_val, y_val)
            outTest = getAttention(device, model, x_test, y_test)
    else:
        print('unknown saliency method: ' + method)

    outTrain = np.array(outTrain).squeeze()
    outVal = np.array(outVal).squeeze()
    outTest = np.array(outTest).squeeze()

    outMap[saveKey + 'Train'].append(outTrain)
    outMap[saveKey + 'Val'].append(outVal)
    outMap[saveKey + 'Test'].append(outTest)
    outMap['means'][saveKey + 'Train'].append(np.mean(outTrain.squeeze(), axis=0))
    outMap['means'][saveKey + 'Val'].append(np.mean(outVal.squeeze(), axis=0))
    outMap['means'][saveKey + 'Test'].append(np.mean(outTest.squeeze(), axis=0))

    do3DData = False
    do2DData = False
    if len(outTrain.shape) > 3:
        do3DData = True
    elif len(outTrain.shape) > 2:
        do2DData = True


    return outTrain, outVal, outTest, do3DData, do2DData
    

# Do ROAR retrain evaluation, but consider a more flexible abstraction
def doSimpleLasaROAR(saliencyMap, data, threshold, doBaselineT=False, doFidelity=False, do3DData=False, do3rdStep=False, axis1= 2, axis2=0, axis3=1, op1='max',op2='sum',op3='max'):
    print('new ROAR start')
    newX = []
    reduction = []

    if do3DData:
        saliencyMap = helper.doCombiStep(op1, saliencyMap, axis1)
        saliencyMap = helper.doCombiStep(op2, saliencyMap, axis2) 
        saliencyMap = helper.doCombiStep(op3, saliencyMap, axis3) 
    elif do3rdStep:
        saliencyMap = helper.doCombiStep(op3, saliencyMap, axis3) 

    X_sax = np.array(data).squeeze()
    heats = saliencyMap.squeeze()
    heats = preprocessing.minmax_scale(heats, axis=1)

    if doBaselineT:
        cutOff = threshold
        threshold = np.max(np.array(heats)[:,-1 * cutOff:], axis=1)

    for index in range(len(saliencyMap)):
                
            X_ori = X_sax[index]
            heat = heats[index] 
    
            if doBaselineT:
                borderHeat = threshold[index]
            else:
                maxHeat = np.average(heat)
                borderHeat = maxHeat*threshold
        
            fitleredSet = []
            skips = 0 
            for h in range(len(heat)):
                if validataHeat(heat[h], borderHeat, doFidelity):
                    fitleredSet.append(X_ori[h])
                else:
                    fitleredSet.append(-2)
                    skips += 1

            reduction.append(skips/len(heat))
            newX.append([fitleredSet])

    newX = np.array(newX, dtype=np.float32)
    newX = np.moveaxis(newX, 1,2)

    return newX, reduction


def validataHeat(value, heat, doFidelity):
    if doFidelity:
        return value <= heat
    else:
        return value > heat
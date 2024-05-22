import numpy as np
from modules import helper
from sklearn.preprocessing import StandardScaler
from pyts.approximation import SymbolicAggregateApproximation
from sklearn.metrics import accuracy_score
import random
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformer_encoder.utils import WarmupOptimizer
import logging

from collections import OrderedDict

def validate_model(device, model, data, lables, val_batch_size, batch_size, name, showLoss = False, output_attentions=False):
    criterion = nn.MSELoss()
    attention = []
    with torch.no_grad():
        epoch_val_loss = 0
        epoch_val_acc = 0
        epoch_val_data = range(len(data)) 
        model.eval()
        for batch_start in range(0, len(epoch_val_data), val_batch_size):
            batch_index = epoch_val_data[batch_start:min(batch_start + batch_size, len(epoch_val_data))]
            batch_data = data[batch_index]
            targets = lables[batch_index]
            targets = torch.from_numpy(targets)
            input_ids = torch.from_numpy(batch_data).to(device) 
            if output_attentions:
                preds, atts = model(input_ids.double(), singleOutput=False, output_attentions=output_attentions)
                atts = atts
                attention.append(atts)
            else:
                preds = model(input_ids.double())
            epoch_val_acc += accuracy_score(preds.argmax(dim=1).cpu(),targets.argmax( dim=1).cpu(), normalize=False)
            loss = criterion(preds, targets.to(device=preds.device)).sum()
            epoch_val_loss += loss.item()

        epoch_val_loss /= len(data)
        epoch_val_acc /= len(data)

        if showLoss:
            print(f'Final '+ name +f' loss {epoch_val_loss}')
        print(f'Final ' + name +f' acc {epoch_val_acc}')
        if(output_attentions):
            return attention
        return preds

def trainBig(device, model, config, classLabels, x_train, y_train, x_val, y_val, x_test, y_test,batch_size=50, epochs=500, fileAdd="", mask=None, useSaves=False):
    val_batch_size = batch_size
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.99), eps=1e-9)
    scheduler = WarmupOptimizer(optimizer, d_model=model.dmodel, scale_factor=1, warmup_steps=10000) 

    print(f'Beginning training classifier')
    save_dir = './savePT/'
    evidence_classifier_output_dir = os.path.join(save_dir, 'classifier')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(evidence_classifier_output_dir, exist_ok=True)
    model_save_file = os.path.join(evidence_classifier_output_dir, 'classifier.pt')
    epoch_save_file = os.path.join(evidence_classifier_output_dir, 'classifier_epoch_data'+fileAdd+'.pt')

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    results = {
        'train_loss': [],
        'train_f1': [],
        'train_acc': [],
        'val_loss': [],
        'val_f1': [],
        'val_acc': [],
    }
    best_epoch = -1
    best_val_acc = 0
    best_val_loss = float('inf')
    best_model_state_dict = None
    start_epoch = 0
    epoch_data = {}

    if os.path.exists(epoch_save_file) and useSaves:
        print(f'Restoring model from {model_save_file}')
        model.load_state_dict(torch.load(model_save_file))
        epoch_data = torch.load(epoch_save_file)
        start_epoch = epoch_data['epoch'] + 1
        # handle finishing because patience was exceeded or we didn't get the best final epoch
        if bool(epoch_data.get('done', 0)):
            start_epoch = epochs
        results = epoch_data['results']
        best_epoch = start_epoch
        best_model_state_dict = OrderedDict({k: v.cpu() for k, v in model.state_dict().items()})
        print(f'Restoring training from epoch {start_epoch}')
    print(f'Training evidence classifier from epoch {start_epoch} until epoch {epochs}')
    scheduler.zero_grad()
    for epoch in range(start_epoch, epochs):
        epoch_train_data = random.sample(range(len(x_train)), k=len(x_train))
        epoch_train_loss = 0
        epoch_training_acc = 0
        model.train()
        print(
            f'Training with {len(epoch_train_data) // batch_size} batches with {len(epoch_train_data)} examples')
        for batch_start in range(0, len(epoch_train_data), batch_size):
            batch_index = epoch_train_data[batch_start:min(batch_start + batch_size, len(epoch_train_data))]
            batch_data = x_train[batch_index]
            targets = y_train[batch_index]

            targets = torch.from_numpy(targets)
            input_ids = torch.from_numpy(batch_data).to(device) 
            preds = model(input_ids.double())#[0]
            epoch_training_acc += accuracy_score(preds.argmax(dim=1).cpu(), targets.argmax(dim=1).cpu(), normalize=False)
            loss = criterion(preds, targets.to(device=preds.device)).sum()
            epoch_train_loss += loss.item()
            loss.backward()
            assert loss == loss  # for nans
            max_grad_norm = False
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scheduler:
                scheduler.step()
            scheduler.zero_grad()
        epoch_train_loss /= len(epoch_train_data)
        epoch_training_acc /= len(epoch_train_data)
        assert epoch_train_loss == epoch_train_loss  # for nans
        results['train_loss'].append(epoch_train_loss)
        print(f'Epoch {epoch} training loss {epoch_train_loss}')
        print(f'Epoch {epoch} training accuracy {epoch_training_acc}')

        with torch.no_grad():
            epoch_val_loss = 0
            epoch_val_acc = 0
            epoch_val_data = random.sample(range(len(x_val)), k=len(x_val))
            model.eval()
            print(
                f'Validating with {len(epoch_val_data) // val_batch_size} batches with {len(epoch_val_data)} examples')
            for batch_start in range(0, len(epoch_val_data), val_batch_size):
                batch_index = epoch_val_data[batch_start:min(batch_start + batch_size, len(epoch_val_data))]
                batch_data = x_val[batch_index]
                targets = y_val[batch_index]
                targets = torch.from_numpy(targets)
                input_ids = torch.from_numpy(batch_data).to(device) 
                preds = model(input_ids.double())
                
                epoch_val_acc += accuracy_score(preds.argmax(dim=1).cpu(),targets.argmax( dim=1).cpu(), normalize=False)
                loss = criterion(preds, targets.to(device=preds.device)).sum()
                epoch_val_loss += loss.item()

            epoch_val_loss /= len(x_val)
            epoch_val_acc /= len(x_val)
            results["val_acc"].append(epoch_val_acc)
            results["val_loss"] = epoch_val_loss

            print(f'Epoch {epoch} val loss {epoch_val_loss}')
            print(f'Epoch {epoch} val acc {epoch_val_acc}')

            if epoch_val_acc > best_val_acc or (epoch_val_acc == best_val_acc and epoch_val_loss < best_val_loss):
                best_model_state_dict = OrderedDict({k: v.cpu() for k, v in model.state_dict().items()})
                best_epoch = epoch
                best_val_acc = epoch_val_acc
                best_val_loss = epoch_val_loss
                epoch_data = {
                    'epoch': epoch,
                    'results': results,
                    'best_val_acc': best_val_acc,
                    'done': 0,
                }
                torch.save(model.state_dict(), model_save_file)
                torch.save(epoch_data, epoch_save_file)
                logging.debug(f'Epoch {epoch} new best model with val accuracy {epoch_val_acc}')
        if epoch - best_epoch > config.patience:
            print(f'Exiting after epoch {epoch} due to no improvement')
            epoch_data['done'] = 1
            torch.save(epoch_data, epoch_save_file)
            break

    epoch_data['done'] = 1
    epoch_data['results'] = results
    torch.save(epoch_data, epoch_save_file)
    model.load_state_dict(best_model_state_dict)
    model = model.to(device=device)

    #final out
    validate_model(device, model, x_train, y_train, val_batch_size, batch_size, 'train')
    validate_model(device, model, x_val, y_val, val_batch_size, batch_size, 'val')
    validate_model(device, model, x_test, y_test, val_batch_size, batch_size, 'test')

    

    return model

# building saving name for model weights
def getWeightName(name, fold, symbols, layers, abstractionType, header, learning = True, resultsPath = 'presults', results=False, usedp=False, doHeaders=True):

    if usedp:
        if results:
            baseName = "./"+resultsPath+"/results-" +name +' -fold: '+str(fold)+' -symbols: '+str(symbols)+ ' -layers: '+str(layers)+' -abstractionType: '+abstractionType
        else: 
            baseName = "./saves/weights-" +name +' -fold: '+str(fold)+' -symbols: '+str(symbols)+ ' -layers: '+str(layers)+' -abstractionType: '+abstractionType
    else:
        if results:
            baseName = "./"+resultsPath+"/results-" +name +' -fold '+str(fold)+' -symbols '+str(symbols)+ ' -layers '+str(layers)+' -abstractionType '+abstractionType
        else: 
            baseName = "./saves/weights-" +name +' -fold '+str(fold)+' -symbols '+str(symbols)+ ' -layers '+str(layers)+' -abstractionType '+abstractionType
    
    if doHeaders:
        baseName = baseName + ' -headers ' + str(header)
    if learning:
        return baseName + '-learning.tf'
    else:
        return baseName + '.tf'
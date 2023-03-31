

import cfg
import functions
import modules

from modules import BaseDataset

import os
import random
import pickle as pkl
import torch
import numpy as np
from tqdm.auto import tqdm as tq
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
from torchvision.transforms import Compose, RandomAffine, ToTensor, Normalize

def get_augmentation_pipeline():
    return Compose([
        RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])
augmentation_pipeline = get_augmentation_pipeline()

def contrastive(configs, loaders):
    def forward_step(batch):
        x1, x2, y = batch
        x1 = x1.to(configs.device)
        x2 = x2.to(configs.device)
        y  = y.to(configs.device)
        
        yhat = model(x1, x2, )
        loss = criterion(yhat, y)
        
        return yhat, loss
    
    # load data, modules, and settings
    train_loader, valid_loader, test_loader = loaders
    model = configs.model
    criterion = configs.criterion
    optimizer = configs.optimizer
    scheduler = configs.scheduler

    # loss tracker
    train_loss_tracker = []
    valid_loss_tracker = []
    valid_acc_tracker  = []
    valid_f1_tracker   = []

    best_loss = 999999
    best_acc  = 0.0
    best_f1   = 0.0
    best_model = None

    # Deploying to devices
    if configs.num_gpus > 1:
        model = torch.nn.DataParallel(model)
    else :
        model = model.to(configs.device)
    
    criterion = criterion.to(configs.device)
    
    # training 
    for epoch in range(1, (configs.epochs + 1)):
        # train stage
        model.train()
        train_loss = []
        train_iterator = tq(train_loader) if configs.tqdm else train_loader
        
        for batch in train_iterator:
            optimizer.zero_grad()
            _, loss = forward_step(batch)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            
        if scheduler is not None:
            scheduler.step()
    
        # validation stage
        model.eval()
        valid_loss = []
        labels = []
        preds  = []
        
        valid_iterator = tq(valid_loader) if configs.tqdm else valid_loader
        with torch.no_grad():
            for batch in valid_iterator:
                yhat, loss = forward_step(batch)
                valid_loss.append(loss.item())

                # result
                y = batch[2].detach().cpu().numpy()
                yhat =  yhat.argmax(1).detach().cpu().numpy()
                
                labels.append(y)
                preds.append(yhat)
                
        labels = np.concatenate(labels, axis=0)
        preds  = np.concatenate(preds,  axis=0)
        
        # metric
        acc, f1 = accuracy_score(labels, preds), f1_score(labels, preds, average = 'macro')
        
        if loss < best_loss:
            best_loss = loss
            best_model = model
        
        train_loss = round(np.mean(train_loss), 4)
        valid_loss = round(np.mean(valid_loss)  , 4)
        
        # inference : test task !
        print(f"-- EPOCH {epoch} --")
        print(f"training   loss : {train_loss}")
        print(f"validation loss : {valid_loss}")
        train_loss_tracker.append(train_loss)
        valid_loss_tracker.append(valid_loss)
        
    return (best_model, train_loss_tracker, valid_loss_tracker, valid_acc_tracker, valid_f1_tracker)










if __name__ == '__main__':
        
    # load data and setting
    configs = cfg.configs
    loaders = functions.get_loader(configs)
    
    # training (train time task : full model)
    if configs.num_gpus >= 1:
        print("--current device : CUDA")
    if configs.num_gpus > 1:
        print(f"--distributed training : {['cuda:'+str(i) for i in range(torch.cuda.device_count())]}")
    outputs = functions.train(configs, loaders)

    # outputs of the training
    best_model         = outputs[0] 
    train_loss_tracker = outputs[1]
    valid_loss_tracker = outputs[2]
    valid_acc_tracker  = outputs[3] 
    valid_f1_tracker   = outputs[4]
    
    # inference (test time task : digit recognizer)
    infer_model = best_model.digit_recognizer
    predicts = functions.inference(configs, infer_model, loaders[-1])
    test_acc = predicts[0] 
    test_f1  = predicts[1] 
    labels   = predicts[2]
    preds    = predicts[3]
    
    
    
    



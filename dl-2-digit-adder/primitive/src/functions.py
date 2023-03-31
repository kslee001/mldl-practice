from modules import BaseDataset

import os
import random
import pickle as pkl
import torch
import numpy as np
from tqdm.auto import tqdm as tq
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


def get_loader(configs):
    # data I/O
    with open(f'{configs.data_dir}/training_tuple.pkl', 'rb') as f:
        training_tuple = pkl.load(f)
    with open(f'{configs.data_dir}/training_dict.pkl', 'rb') as f:
        training_dict  = pkl.load(f)
    with open(f'{configs.data_dir}/test.pkl', 'rb') as f:
        test = pkl.load(f)
        
    # train, valid, and test dataset
    X_train = training_tuple[0]
    Y_train = training_tuple[1]
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=configs.val_rate, random_state=configs.seed)
    X_test  = test[0]
    Y_test  = test[1]
    train_dataset = BaseDataset(X_train, Y_train, mode='train')
    valid_dataset = BaseDataset(X_train, Y_train, mode='valid')
    test_dataset  = BaseDataset(X_test,  Y_test,  mode='test')
    
    # data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=configs.batch_size, num_workers = configs.num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=configs.batch_size, num_workers = configs.num_workers,shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=configs.batch_size, num_workers = configs.num_workers,shuffle=False)
    
    return (train_loader, valid_loader, test_loader)
    

def train(configs, loaders):
    mse = torch.nn.MSELoss()
    ce  = torch.nn.CrossEntropyLoss()
    
    def mse_forward_step(batch):
        x1, x2, y = batch
        x1 = x1.to(configs.device)
        x2 = x2.to(configs.device)
        y_mse  = y.to(configs.device).float().reshape(-1, 1)
        yhat = model(x1, x2)
        loss = mse(yhat, y_mse)
        return yhat, loss
    
    def ce_forward_step(batch):
        x1, x2, y = batch
        x1 = x1.to(configs.device)
        x2 = x2.to(configs.device)
        y_ce  = y.to(configs.device)
        yhat = model(x1, x2)
        loss = criterion(yhat, y_ce)
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
        reconstruction_loss = []
        train_iterator = tq(train_loader) if configs.tqdm else train_loader
        
        for batch in train_iterator:
            # classification
            optimizer.zero_grad()
            if configs.mse :
                _, loss = mse_forward_step(batch)
            else:
                _, loss = ce_forward_step(batch)
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
                if configs.mse :
                    yhat, loss = mse_forward_step(batch)
                else:
                    yhat, loss = ce_forward_step(batch)
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
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
        
        train_loss = round(np.mean(train_loss), 4)
        reconstruction_loss = round(np.mean(reconstruction_loss), 4)
        valid_loss = round(np.mean(valid_loss)  , 4)
        valid_acc  = round(acc, 4)
        valid_f1   = round(f1, 4)
        
        # inference : test task !
        print(f"-- EPOCH {epoch} --")
        print(f"training   loss : {train_loss}")
        print(f"validation loss : {valid_loss}")
        print(f"current val acc : {valid_acc}")
        print(f"current val f1  : {valid_f1}") 
        print(f"best val acc    : {round(best_acc, 4)}")
        print(f"best val f1     : {round(best_f1, 4)}")
        print(f"labels (first 5 items)  : {labels[:5]}")
        print(f"preds  (first 5 items)  : {preds[:5]}")
        train_loss_tracker.append(train_loss)
        valid_loss_tracker.append(valid_loss)
        valid_acc_tracker.append(valid_acc)
        valid_f1_tracker.append(valid_f1)
        
        print(" -- test result ")
        inference(configs, model.digit_recognizer, loaders[-1])
        
    return (best_model, train_loss_tracker, valid_loss_tracker, valid_acc_tracker, valid_f1_tracker)



def inference(configs, infer_model, test_loader):
    def forward_step(batch):
        x, y = batch
        x = x.to(configs.device)
        yhat = model(x)
        return yhat
    
    model = infer_model.to(configs.device)
    # test stage
    model.eval()
    labels = []
    preds  = []

    test_iterator = tq(test_loader) if configs.tqdm else test_loader
    with torch.no_grad():
        for batch in test_iterator:
            yhat = forward_step(batch)

            # result
            y = batch[1].detach().cpu().numpy()
            yhat =  yhat.argmax(1).detach().cpu().numpy()

            labels.append(y)
            preds.append(yhat)

    labels = np.concatenate(labels, axis=0)
    preds  = np.concatenate(preds,  axis=0)
    # metric
    acc, f1 = accuracy_score(labels, preds), f1_score(labels, preds, average = 'macro')
    test_acc  = round(acc, 4)
    test_f1   = round(f1, 4)
    
    print(f" test acc & f1 score    : {[test_acc, test_f1]}")
    print(f"labels (first 10 items)  : {labels[:10]}")
    print(f"preds  (first 10 items)  : {preds[:10]}")
    return test_acc, test_f1, labels, preds



def autoencode_train(configs, loaders):
    mse = torch.nn.MSELoss()
    ce  = torch.nn.CrossEntropyLoss()

    def autoencoder_forward_step(batch):
        x1, x2, _ = batch
        x1 = x1.to(configs.device)
        x2 = x2.to(configs.device)
        
        cur_reconstruction_loss = 0
        
        # autoencoding : x1
        optimizer.zero_grad()
        x1_embed = model.digit_recognizer(x1, project=True)
        loss = mse(x1, x1_embed)
        cur_reconstruction_loss += loss.item()
        loss.backward()
        optimizer.step()

        # autoencoding : x2
        optimizer.zero_grad()
        x2_embed = model.digit_recognizer(x2, project=True)
        loss = mse(x2, x2_embed)
        loss.backward()
        cur_reconstruction_loss += loss.item()
        optimizer.step()
        
        return cur_reconstruction_loss
        
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
        reconstruction_loss = []
        train_iterator = tq(train_loader) if configs.tqdm else train_loader
        
        for batch in train_iterator:
            # autoencoding
            cur_reconstruction_loss = autoencoder_forward_step(batch)
            reconstruction_loss.append(cur_reconstruction_loss)
            
        if scheduler is not None:
            scheduler.step()
            
        reconstruction_loss = round(np.mean(reconstruction_loss), 4)
        print(f"-- EPOCH {epoch} --")
        print(f"recon loss      : {reconstruction_loss}")
        
    return (best_model, train_loss_tracker, valid_loss_tracker, valid_acc_tracker, valid_f1_tracker)
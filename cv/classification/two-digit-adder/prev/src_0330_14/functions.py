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
    train_dataset = BaseDataset(X_train, Y_train, configs, mode='train')
    valid_dataset = BaseDataset(X_valid, Y_valid, configs, mode='valid')
    test_dataset  = BaseDataset(X_test,  Y_test,  configs, mode='test')
    
    # data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=configs.batch_size, num_workers = configs.num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=configs.batch_size, num_workers = configs.num_workers,shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=configs.batch_size, num_workers = configs.num_workers,shuffle=False)
    
    return (train_loader, valid_loader, test_loader)
    

def fit(configs, loaders):
    def forward_step(batch):
        x1, x2, y = batch
        x1 = x1.to(configs.device)
        x2 = x2.to(configs.device)
        y  = y.to(configs.device)
        yhat, cur_zero_vectors = model(x1, x2)
        loss = criterion(yhat, y)
        return yhat, cur_zero_vectors, loss
    
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
        cur_zero_vectors_list = []
        # train stage
        model.train()
        train_loss = []
        train_iterator = tq(train_loader) if configs.tqdm else train_loader
        
        for batch in train_iterator:
            # classification
            optimizer.zero_grad()
            _, cur_zero_vectors, loss = forward_step(batch)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if cur_zero_vectors is not None:
                cur_zero_vectors_list.append(cur_zero_vectors)

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
                yhat, _, loss = forward_step(batch)
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
        
        train_loss = round(np.mean(train_loss), 4)
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
        
        print("-- test result ")

        if valid_f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_zero_vectors = torch.vstack(cur_zero_vectors_list)
            model.zero_vector = best_zero_vectors.mean(axis=0) # (10, )
            
        inference(configs, model, test_loader)
            
    return (best_model, train_loss_tracker, valid_loss_tracker, valid_acc_tracker, valid_f1_tracker)


def inference(configs, infer_model, test_loader):
    def forward_step(batch):
        x, _ = batch
        x = x.to(configs.device)
        yhat = model(x, None)
        return yhat
    
    model = infer_model.to(configs.device)
    # test stage
    model.eval()
    labels = []
    preds  = []

    # test_iterator = tq(test_loader) if configs.tqdm else test_loader
    test_iterator = test_loader
    with torch.no_grad():
        for batch in test_iterator:
            yhat = forward_step(batch)

            # result
            y = batch[1].detach().cpu().numpy()
            yhat = yhat.argmax(1).detach().cpu().numpy()
            # yhat = yhat//2  # <- key point !!! novelty
        
            labels.append(y)
            preds.append(yhat)

    labels = np.concatenate(labels, axis=0)
    preds  = np.concatenate(preds,  axis=0)
    # metric
    acc, f1  = accuracy_score(labels, preds), f1_score(labels, preds, average = 'macro')
    test_acc = round(acc, 4)
    test_f1  = round(f1,  4)
    
    print(f"-- test acc & f1 score    : {[test_acc, test_f1]}")
    print(f"labels (first 10 items) : {labels[:10]}")
    print(f"preds  (first 10 items) : {preds[:10]}")
    return test_acc, test_f1, labels, preds


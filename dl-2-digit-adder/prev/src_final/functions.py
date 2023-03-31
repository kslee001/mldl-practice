from modules import BaseDataset

import os
import random
import pickle as pkl
import torch
import numpy as np
from tqdm.auto import tqdm as tq
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import argparse
from torchinfo import summary

def initiate(configs):
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', "--seed", dest="seed", action="store", default=1203)
    parser.add_argument('-t', "--tqdm", dest="tqdm", action="store_true")
    parser.add_argument('-e', "--epochs", dest="epochs", action='store', default=20)
    args = parser.parse_args()
    configs.seed = int(args.seed)
    configs.tqdm = args.tqdm
    configs.epochs = int(args.epochs)
    print(f"-- current seed : {configs.seed}")
    
    # training (train time task : full model)
    if configs.num_gpus >= 1:
        print("-- current device : CUDA")
    if configs.num_gpus > 1:
        print(f"-- distributed training : {['cuda:'+str(i) for i in range(torch.cuda.device_count())]}")
    
    # model summary
    model_summary = summary(
        model=configs.model, 
        input_size=(configs.batch_size, 1, 28, 28),
        verbose=1 # 0 : no output / 1 : print model summary / 2 : full detail(weight, bias layers)
    ).__repr__()
    return model_summary # string

def get_loader(configs):
    def random_affine(x):
        return A.Affine(rotate=(-8, 8), translate_percent=(0.05, 0.05), scale=(0.95, 1.05), shear=(-8, 8))(image=x)['image']

    def oversampling_with_augmentation(X_train, y_train):
        indices_for_each_labels = []
        for label in range(0, 19):
            label_indices = torch.nonzero(y_train == label, as_tuple = False).numpy().squeeze()
            indices_for_each_labels.append(label_indices)

        X_train_labels = []
        y_train_labels = []
        for label in range(0, 19):
            X_train_labels.append(X_train[indices_for_each_labels[label]].numpy())
            y_train_labels.append(torch.tensor([label for _ in range(len(indices_for_each_labels[label]))]))

        max_count = max([len(y_train_labels[idx]) for idx in range(len(y_train_labels))])

        X_train_np = X_train.numpy()
        
        X_train_augmented = []
        y_train_augmented = []
        for label in range(0, 19):    
            # 최대 label의 데이터 개수 - 현재 label의 데이터 개수 만큼 random sampling
            target_num_sampling = max_count - len(indices_for_each_labels[label]) 

            # sampling -> augmentation -> append to list
            cnt = 0
            while (cnt < target_num_sampling):
                random_idx = random.sample(sorted(indices_for_each_labels[label].tolist()), k=1)
                sample = X_train_np[random_idx].squeeze()
                left = random_affine(sample[0])
                right = random_affine(sample[1])
                augmented = np.stack([left, right])

                X_train_augmented.append(torch.tensor(augmented))
                y_train_augmented.append(torch.tensor(label))
                cnt += 1
                
        # list -> tensor
        X_train_augmented = torch.stack(X_train_augmented)
        y_train_augmented = torch.stack(y_train_augmented)
        
        return X_train_augmented, y_train_augmented
    
    
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
    
    if configs.oversampling:
        X_train_aug, Y_train_aug = oversampling_with_augmentation(X_train, Y_train)
        X_train = torch.vstack([X_train, X_train_aug])
        Y_train = torch.vstack([Y_train.unsqueeze(1), Y_train_aug.unsqueeze(1)]).squeeze(1)
    
    print("-- shape of data")
    print(f"X_train : {X_train.shape}")
    print(f"Y_train : {Y_train.shape}")
    print(f"X_valid : {X_valid.shape}")
    print(f"Y_valid : {Y_valid.shape}")
    
    
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

    def forward_step_switch(batch):
        x2, x1, y = batch # switched
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
    train_acc_tracker  = []
    train_f1_tracker   = []
    valid_loss_tracker = []
    valid_acc_tracker  = []
    valid_f1_tracker   = []
    test_acc_tracker   = []
    test_f1_tracker    = []

    # best model setting 
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
        train_labels = []
        train_preds  = []
        train_iterator = tq(train_loader) if configs.tqdm else train_loader
        
        for batch in train_iterator:
            # first forward-backward
            optimizer.zero_grad()
            yhat, cur_zero_vectors, loss = forward_step(batch)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if cur_zero_vectors is not None:
                cur_zero_vectors_list.append(cur_zero_vectors)
            
            # result
            y    = batch[2].detach().cpu().numpy()
            yhat = yhat.argmax(1).detach().cpu().numpy()
            
            train_labels.append(y)
            train_preds.append(yhat)
            
            # second forward-backward
            optimizer.zero_grad()
            yhat, cur_zero_vectors, loss = forward_step_switch(batch)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if cur_zero_vectors is not None:
                cur_zero_vectors_list.append(cur_zero_vectors)
            
            # result
            y    = batch[2].detach().cpu().numpy()
            yhat =  yhat.argmax(1).detach().cpu().numpy()
            
            train_labels.append(y)
            train_preds.append(yhat)
            
        train_labels = np.concatenate(train_labels, axis=0)
        train_preds  = np.concatenate(train_preds,  axis=0)
        
        # metric
        train_acc, train_f1 = round(accuracy_score(train_labels, train_preds), 4), round(f1_score(train_labels, train_preds, average = 'macro'), 4)
        train_loss = round(np.mean(train_loss), 4)
          
        if scheduler is not None:
            scheduler.step()
    
        # validation stage
        model.eval()
        valid_loss = []
        valid_labels = []
        valid_preds  = []
        
        valid_iterator = tq(valid_loader) if configs.tqdm else valid_loader
        with torch.no_grad():
            for batch in valid_iterator:
                yhat, _, loss = forward_step(batch)
                valid_loss.append(loss.item())
                
                # result
                y    = batch[2].detach().cpu().numpy()
                yhat =  yhat.argmax(1).detach().cpu().numpy()
                
                valid_labels.append(y)
                valid_preds.append(yhat)
                
        valid_labels = np.concatenate(valid_labels, axis=0)
        valid_preds  = np.concatenate(valid_preds,  axis=0)
        
        # metric
        valid_acc, valid_f1 = round(accuracy_score(valid_labels, valid_preds), 4), round(f1_score(valid_labels, valid_preds, average = 'macro'), 4)
        valid_loss = round(np.mean(valid_loss), 4)

        if valid_f1 > best_f1:
            best_f1 = valid_f1
            
        if valid_acc > best_acc:
            best_acc = valid_acc
            
        if valid_loss < best_loss:
            best_loss = valid_loss
            # update best model
            best_model = model
            best_zero_vectors = torch.vstack(cur_zero_vectors_list)
            model.zero_vector = best_zero_vectors.mean(axis=0) # (10, )
            

        print(f"-- EPOCH {epoch} --")
        print(f"current train loss : {train_loss}")
        print(f"current train acc  : {train_acc}")
        print(f"current train f1   : {train_f1}") 

        print(f"current valid loss : {valid_loss}")
        print(f"current val acc    : {valid_acc}")
        print(f"current val f1     : {valid_f1}") 
        
        print(f"best val loss      : {round(best_loss, 4)}")
        print(f"best val acc       : {round(best_acc, 4)}")
        print(f"best val f1        : {round(best_f1, 4)}")
        print(f"valid labels (first 5 items)  : {valid_labels[:5]}")
        print(f"valid preds  (first 5 items)  : {valid_preds[:5]}")
        train_loss_tracker.append(train_loss)
        train_acc_tracker.append(train_acc)
        train_f1_tracker.append(train_f1)        

        valid_loss_tracker.append(valid_loss)
        valid_acc_tracker.append(valid_acc)
        valid_f1_tracker.append(valid_f1)
        
        print("-- test result ")
        test_result = inference(configs, model, test_loader)
        test_acc = test_result[0]
        test_f1  = test_result[1] 
        
        test_acc_tracker.append(test_acc)
        test_f1_tracker.append(test_f1)
            
    return (best_model, train_loss_tracker, train_acc_tracker, train_f1_tracker, valid_loss_tracker, valid_acc_tracker, valid_f1_tracker, test_acc_tracker, test_f1_tracker)


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
    
    print(f"acc & f1 score          : {[test_acc, test_f1]}")
    print(f"labels (first 10 items) : {labels[:10]}")
    print(f"preds  (first 10 items) : {preds[:10]}\n")
    return (test_acc, test_f1, labels, preds)


def get_final_results(configs, best_model, loaders):
    def forward_step(batch):
        x1, x2, y = batch
        x1 = x1.to(configs.device)
        x2 = x2.to(configs.device)
        y  = y.to(configs.device)
        yhat, _ = model(x1, x2)
        return yhat
    
    train_loader, valid_loader, test_loader = loaders
    model = best_model.to(configs.device)
    model.eval()
    
    with torch.no_grad():
        # 1. train dataset    
        train_labels = []
        train_preds  = []
        train_iterator = tq(train_loader) if configs.tqdm else train_loader
        
        for batch in train_iterator:
            # first forward-backward
            yhat = forward_step(batch)
            
            # result
            y    = batch[2].detach().cpu().numpy()
            yhat = yhat.argmax(1).detach().cpu().numpy()
            train_labels.append(y)
            train_preds.append(yhat)
                        
        train_labels = np.concatenate(train_labels, axis=0)
        train_preds  = np.concatenate(train_preds,  axis=0)
        train_acc, train_f1 = round(accuracy_score(train_labels, train_preds), 4), round(f1_score(train_labels, train_preds, average = 'macro'), 4)
          
          
        # 2. validation dataset
        valid_labels = []
        valid_preds  = []
        
        valid_iterator = tq(valid_loader) if configs.tqdm else valid_loader
        for batch in valid_iterator:
            yhat = forward_step(batch)
            
            # result
            y    = batch[2].detach().cpu().numpy()
            yhat =  yhat.argmax(1).detach().cpu().numpy()
            
            valid_labels.append(y)
            valid_preds.append(yhat)
                
        valid_labels = np.concatenate(valid_labels, axis=0)
        valid_preds  = np.concatenate(valid_preds,  axis=0)
        valid_acc, valid_f1 = round(accuracy_score(valid_labels, valid_preds), 4), round(f1_score(valid_labels, valid_preds, average = 'macro'), 4)


        # 3. test dataset
        test_result = inference(configs, model, test_loader)
        test_acc = test_result[0]
        test_f1  = test_result[1] 

    return (train_acc, train_f1, valid_acc, valid_f1, test_acc, test_f1)

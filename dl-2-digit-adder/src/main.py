import cfg
import functions
import modules

import random
import os
import numpy as np
import torch
import pandas as pd
from datetime import datetime

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    # load data & setting
    configs = cfg.configs
    seed_everything(configs.seed)
    loaders = functions.get_loader(configs)
    
    # parse arguments & check model information
    model_summary = functions.initiate(configs)

    # training & outputs of the training
    outputs = functions.fit(configs, loaders)
    
    best_model   = outputs[0]
    train_result = outputs[1:] 
    # train_loss_tracker = train_result[1]
    # train_acc_tracker  = train_result[2] 
    # train_f1_tracker   = train_result[3]
    # valid_loss_tracker = train_result[4]
    # valid_acc_tracker  = train_result[5] 
    # valid_f1_tracker   = train_result[6]
    
    # inference (test time task : digit recognizer)
    print("\n=== final prediction of best model ===")
    test_result = functions.get_final_results(configs, best_model, loaders)
    # test_acc = test_result[0] 
    # test_f1  = test_result[1] 
    # labels   = test_result[2]
    # preds    = test_result[3]
    
    # save model information, train / test results and best model (for inference)
    # model summary
    now = datetime.today()
    now = "".join([str(now.year), str(now.month).zfill(2), str(now.day).zfill(2), str(now.hour).zfill(2), str(now.minute).zfill(2)])

    with open(f'./model_summary_{str(configs.seed)}_{now}.txt', 'w', encoding='utf-8') as f:
        f.write(model_summary)
    # train results
    train_result = pd.DataFrame(train_result).T.reset_index()
    train_result.columns = [
        'epoch', 'train_loss', 'train_acc', 'train_f1', 
        'valid_loss', 'valid_acc', 'valid_f1',
        'test_acc', 'test_f1'                    
    ]
    train_result.to_csv(f"./train_result_{str(configs.seed)}_{now}.csv", index=False, encoding='utf-8')

    # test results
    test_result = pd.DataFrame(test_result).T
    test_result.columns = ['train_acc', 'train_f1', 'valid_acc', 'valid_f1', 'test_acc', 'test_f1'] 
    test_result.to_csv(f"./test.result_{str(configs.seed)}_{now}.csv", index=False, encoding='utf-8')

    # best model 
    torch.save(best_model, f'./best_model_{str(configs.seed)}_{now}.pt')
    
    
    
    

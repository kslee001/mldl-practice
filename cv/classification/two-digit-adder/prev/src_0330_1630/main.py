import cfg
import functions
import modules

import torch
import argparse

if __name__ == '__main__':
    
    
    # load data and setting
    configs = cfg.configs
    loaders = functions.get_loader(configs)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', "--seed", dest="seed", action="store", default=1203)
    parser.add_argument('-t', "--tqdm", dest="tqdm", action="store_true")
    args = parser.parse_args()
    configs.seed = int(args.seed)
    configs.tqdm = args.tqdm
    print(f"-- current seed : {configs.seed}")
    
    # training (train time task : full model)
    if configs.num_gpus >= 1:
        print("--current device : CUDA")
    if configs.num_gpus > 1:
        print(f"--distributed training : {['cuda:'+str(i) for i in range(torch.cuda.device_count())]}")
    
    outputs = functions.fit(configs, loaders)

    # outputs of the training
    best_model         = outputs[0] 
    train_loss_tracker = outputs[1]
    valid_loss_tracker = outputs[2]
    valid_acc_tracker  = outputs[3] 
    valid_f1_tracker   = outputs[4]
    
    # inference (test time task : digit recognizer)
    print("\n=== final prediction ===")
    predicts = functions.inference(configs, best_model, loaders[-1])
    test_acc = predicts[0] 
    test_f1  = predicts[1] 
    labels   = predicts[2]
    preds    = predicts[3]
    
    
    
    

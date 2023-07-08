import cfg
import functions
import modules

import torch

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
    predicts = functions.inference(configs, best_model, loaders[-1])
    test_acc = predicts[0] 
    test_f1  = predicts[1] 
    labels   = predicts[2]
    preds    = predicts[3]
    
    
    
    

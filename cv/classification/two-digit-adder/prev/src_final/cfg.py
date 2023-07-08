import os
import random
import numpy as np
import torch
import modules

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
class CFG:
    def __init__(self):
        return
    
configs = CFG()
configs.seed          = 1203
seed_everything(configs.seed)

# directory
configs.data_dir = '/home/gyuseonglee/workspace/2day/data/Paired_MNIST'

# training setting
configs.batch_size    = 128
configs.learning_rate = 0.0007
configs.epochs        = 20
configs.val_rate      = 0.1 #  1.666 -> train 50000 | valid 10000   &   0.1 -> valid ~1400
configs.augmentation  = True
configs.normalization = False
configs.oversampling  = False

# modules setting
configs.model     = modules.DigitAdder()
configs.optimizer = torch.optim.SGD(
    params=configs.model.parameters(), lr=configs.learning_rate, momentum=0.9
) 

configs.scheduler = modules.CosineAnnealingWarmUpRestarts(
    optimizer=configs.optimizer,
    T_0=configs.epochs,
    T_up=8, # warm-up iteration
    T_mult=1,
    eta_max=configs.learning_rate,
    gamma=0.5, # learning rate decay for each restart
)
configs.criterion = torch.nn.CrossEntropyLoss()
configs.warmup = None

# device
configs.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
configs.num_gpus = 1 
configs.num_workers = 2

# utils
configs.tqdm = True
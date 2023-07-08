import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


# ====== Conv Batchnorm Activation block ==================================================
class ConvBnAct(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3,
                 stride=1,
                 padding=1, activation='gelu'):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            bias=False,
        )
        self.bn = torch.nn.BatchNorm2d(
            num_features=out_channels, eps=1e-05, momentum=0.1)
        
        if activation == 'gelu':
            self.activate = torch.nn.GELU(approximate='tanh') 
        else:
            self.activate = torch.nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))
    
    
# ====== Conv Residual Cell ===============================================
class DoubleConvBnAct(torch.nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1, activation='gelu'):
        super().__init__()
        if activation == 'gelu':
            self.activate = torch.nn.GELU(approximate='tanh') 
        else:
            self.activate = torch.nn.ReLU(inplace=True)
            
        self.conv1 = ConvBnAct(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.conv2 = ConvBnAct(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        
    def forward(self, x):
        return x + self.conv2(self.conv1(x))
        
    
    
# ====== Single digit recognizer ==================================================
class DigitRecognizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.convs = torch.nn.Sequential(
            ConvBnAct(in_channels= 1, out_channels=32, activation='relu'),
            DoubleConvBnAct(32),
            ConvBnAct(in_channels=32, out_channels=64, activation='relu'),
            DoubleConvBnAct(64),
            self.pool, # 28 x 28 -> 14 x 14
            ConvBnAct(in_channels=64, out_channels=96, activation='relu'),
            DoubleConvBnAct(96),
            ConvBnAct(in_channels=96, out_channels=128, activation='relu'),
            self.pool, # 14 x 14 -> 7 x 7
            DoubleConvBnAct(128),
        )        
        self.linear = torch.nn.Linear(128 * 7 * 7, 10)
        
    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x.permute(0, 2, 3, 1), 1) # embedding
        return self.linear(x)


# ====== Single digit recognizer ==================================================
class TinyDigitRecognizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.convs = torch.nn.Sequential(
            ConvBnAct(in_channels= 1, out_channels=32, activation='relu'),
            DoubleConvBnAct(32),
            self.pool,
            ConvBnAct(in_channels=32, out_channels=64, activation='relu'),
            DoubleConvBnAct(64),
            self.pool, # 28 x 28 -> 14 x 14
            ConvBnAct(in_channels=64, out_channels=96, activation='relu'),
            DoubleConvBnAct(96),
            ConvBnAct(in_channels=96, out_channels=128, activation='relu'),
            self.pool, # 14 x 14 -> 7 x 7
            DoubleConvBnAct(128),
        )        
        self.linear = torch.nn.Linear(128 * 7 * 7, 10)
        
    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x.permute(0, 2, 3, 1), 1) # embedding
        return self.linear(x)
    
    
# ====== Single digit recognizer ==================================================
class DigitAdder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.digit_recognizer = DigitRecognizer() # single digit recognizer !
        self.classifier       = torch.nn.Linear(20, 19)
        self.zero_vector      = torch.zeros(10)  # for inference time

    def forward(self, x1, x2=None):
        x1_out = self.digit_recognizer(x1)
        
        if x2 is None: # test time 
            x2_out = self.zero_vector.repeat((x1.shape[0], 1)).to(x1.device)  # x1 + 0 
        else:          # train time
            x2_out = self.digit_recognizer(x2)
        
        # (B, 10+10)-> (B, 19)
        out = self.classifier( torch.cat([x1_out, x2_out], 1) ) 
        
        # zero vector
        if x2 is None: # test time
            return out
        else:          # train time
            cur_zero_vectors = x2_out[out.argmax(1)==0] # 0 이 없는 경우도 있음.
            return out, cur_zero_vectors
    
        
        
    
# ====== Dataset ==========================================================
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, configs, mode ='train'):
        self.X = X.float().numpy()
        self.Y = Y
        self.configs = configs
        self.mode = mode
                
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.mode in ['train', 'valid']:
            x1 = self.X[idx][0].reshape(1, 28, 28)
            x2 = self.X[idx][1].reshape(1, 28, 28)
            y  = self.Y[idx] 
            if self.configs.augmentation == True :
                x1 = self.random_affine(x1)
                x2 = self.random_affine(x2)
            if self.configs.normalization == True:
                x1 = self.normalize(x1)
                x2 = self.normalize(x2)
            x1 = torch.from_numpy(x1)
            x2 = torch.from_numpy(x2)    
            return x1, x2, y
        else:
            x = self.X[idx].reshape(1, 28, 28)
            y = self.Y[idx]
            if self.configs.normalization == True:
                x = self.normalize(x)
            x = torch.from_numpy(x)
            return x, y
        
    
    def normalize(self, x):
        return A.Normalize((0.1307,),(0.3081,))(image=x)['image']
        
    def random_affine(self, x):
        return A.Affine(rotate=(-8, 8), translate_percent=(0.05, 0.05), scale=(0.95, 1.05), shear=(-8, 8))(image=x)['image']

        
# ====== Scheduler =========================================================
class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

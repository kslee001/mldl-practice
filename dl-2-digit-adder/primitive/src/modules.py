import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from torchvision.transforms import Compose, RandomAffine, ToTensor, Normalize
def get_augmentation_pipeline():
    return Compose([
        RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])
augmentation_pipeline = get_augmentation_pipeline()


# import torch
# import torch.nn as nn

# class Flatten(nn.Module):
#     def forward(self, input):
#         return input.view(input.size(0), -1)

# class UnFlatten(nn.Module):
#     def forward(self, input):
#         return input.view(input.size(0), 64, 7, 7)

# class DigitRecognizer(nn.Module):
#     def __init__(self):
#         super(DigitRecognizer, self).__init__()

#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             Flatten(),
#             nn.Linear(64 * 7 * 7, 256),
#             nn.ReLU(),
#         )

#         self.decoder = nn.Sequential(
#             nn.Linear(256, 64 * 7 * 7),
#             nn.ReLU(),
#             UnFlatten(),
#             nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2, padding=0),
#             nn.Sigmoid()
#         )
        

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         # x = self.d1(x)
#         return x

# class ArithmeticNN(nn.Module):
#     def __init__(self):
#         super(ArithmeticNN, self).__init__()
#         self.digit_recognizer = DigitRecognizer()
#         self.fc1 = nn.Linear(256 * 2, 128)
#         self.fc2 = nn.Linear(128, 1)  # 19 classes: 0 to 18

#     def forward(self, x1, x2):
        
#         # arithmetic phase
#         x1 = self.digit_recognizer.encoder(x1) # [512,256]
#         x2 = self.digit_recognizer.encoder(x2)
#         x = torch.cat((x1, x2), dim=1)
#         x = nn.ReLU()(self.fc1(x))
#         yhat = self.fc2(x) # [512, 19]
#         # autoencoding phase
#         x1_embed = self.digit_recognizer.decoder(x1)
#         x2_embed = self.digit_recognizer.decoder(x2)

#         return yhat, x1_embed, x2_embed










# ===============================================================
# base module : single conv layer (conv -> batch norm -> activation)
class ConvBnAct(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3,
                 stride=1,
                 padding=0, activation='gelu'):
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

class ConvTransposeBnAct(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3,
                 stride=1,
                 padding=0, 
                 output_padding=0,
                 activation='gelu'):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
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
        return self.activate(self.bn(self.conv_transpose(x)))
    

class UnFlatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 256, 7, 7)
    
class DigitRecognizer(torch.nn.Module):
    def __init__(self, mse=False):
        super().__init__()
        self.mse = mse
        self.maxpool = torch.nn.MaxPool2d(2, 2) # 안쓸듯..?
        self.activate = torch.nn.GELU(approximate='tanh')
        
        self.encoder = torch.nn.Sequential(
            ConvBnAct(in_channels= 1,  out_channels=32, padding=1, activation='gelu'),
            self.activate,
            ConvBnAct(in_channels= 32, out_channels=64, padding=1, activation='gelu'),            
            self.activate,
            self.maxpool, # 28 -> 14
            
            ConvBnAct(in_channels= 64, out_channels=128, padding=1, activation='gelu'),            
            self.activate,
            ConvBnAct(in_channels= 128, out_channels=256, padding=1, activation='gelu'),            
            self.activate,
            self.maxpool, # 14 -> 7
            
            torch.nn.Flatten(),
            torch.nn.Linear(256*7*7, 256),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(256, 256*7*7),
            UnFlatten(),
            ConvTransposeBnAct(256, 128, kernel_size=2, stride=2, padding=0, activation='gelu'),
            self.activate,
            ConvTransposeBnAct(128, 64, kernel_size=3, stride=1, padding=1, activation='gelu'),
            self.activate,
            ConvTransposeBnAct(64, 32, kernel_size=2, stride=2, padding=0, activation='gelu'),
            self.activate,
            ConvTransposeBnAct(32, 1, kernel_size=3, stride=1, padding=1, activation='gelu'),
            self.activate,
        )

        self.linear = torch.nn.Linear(256, 10) # Cross entropy loss
                
    def forward(self, x, project=False):
        x = self.encoder(x)
        if project : 
            return self.decoder(x) 
        elif self.mse:
            return x
        else: # cross entropy
            return self.activate(self.linear(x))
       
       
# single digit recognizer
class DigitRecognizer(torch.nn.Module):
    def __init__(self, mse=False):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(2, 2) # 안쓸듯..?
        
        self.conv1 = ConvBnAct(in_channels= 1, out_channels=32, activation='relu')
        self.conv2 = ConvBnAct(in_channels=32, out_channels=48, activation='relu')
        self.conv3 = ConvBnAct(in_channels=48, out_channels=64, activation='relu')

        self.conv4 = ConvBnAct(in_channels=64, out_channels=80, activation='relu')
        self.conv5 = ConvBnAct(in_channels=80, out_channels=96, activation='relu')
        self.conv6 = ConvBnAct(in_channels=96, out_channels=112, activation='relu')

        self.conv7 = ConvBnAct(in_channels=112, out_channels=118, activation='relu')
        self.conv8 = ConvBnAct(in_channels=118, out_channels=144, activation='relu')
        self.conv9 = ConvBnAct(in_channels=144, out_channels=160, activation='relu')
        self.conv10 = ConvBnAct(in_channels=160, out_channels=176, activation='relu')

        self.convs = [
            self.conv1, self.conv2, self.conv3,
            self.conv4, self.conv5, self.conv6,
            self.conv7, self.conv8, self.conv9, self.conv10,
        ]
        self.linear = torch.nn.Linear(11264, 10)
        
    def forward(self, x):
        for c in self.convs:
            x = c(x)   # [batch_size, channels, height, weight]
        x = torch.flatten(x.permute(0, 2, 3, 1), 1) # embedding
        x = self.linear(x)
        return x
    
            
# full model : add two digits (recognized by DigitRecognizer)
class DigitAdder(torch.nn.Module):
    def __init__(self, mse=False):
        super().__init__()
        self.mse = mse

        self.digit_recognizer = DigitRecognizer(mse=mse)
        
        if self.mse:
            self.add_layer = torch.nn.Sequential(
                torch.nn.Linear(512, 128),
                torch.nn.GELU(approximate='tanh'),
                torch.nn.Linear(128, 1)
            )
        else:
            self.add_layer = torch.nn.Linear(100, 19)


    def forward(self, x1, x2):
        B = x1.shape[0] # batch size
        x1_out = self.digit_recognizer(x1)
        x2_out = self.digit_recognizer(x2)

        if self.mse :
            summation = self.add_layer(torch.cat([x1_out, x2_out], 1))
        else:
            comb = x1_out.unsqueeze(2) + x2_out.unsqueeze(1)
            summation = self.add_layer(comb.view(B, -1))
            
        return summation
        


# ===============================================================
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

# ===== Dataset ==========================================================
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, mode ='train'):
        self.X = X
        self.Y = Y
        self.mode = mode
                
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.mode in ['train', 'valid']:
            x1 = self.X[idx][0].reshape(1, 28, 28).float()
            # x1 = augmentation_pipeline(x1)
            x2 = self.X[idx][1].reshape(1, 28, 28).float() 
            # x2 = augmentation_pipeline(x2)
            y = self.Y[idx] 
            return x1, x2, y
        else:
            x = self.X[idx].reshape(1, 28, 28).float()
            y = self.Y[idx]
            return x, y
        
        
# ===== Loss function for contrastive learning ==========================================================      
class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z = torch.cat((z_i, z_j), dim=0)
        sim_matrix = self.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_ij = torch.diag(sim_matrix, batch_size)
        sim_ji = torch.diag(sim_matrix, -batch_size)
        positive_samples = torch.cat((sim_ij, sim_ji), dim=0)

        nominator = torch.exp(positive_samples)
        denominator = torch.sum(torch.exp(sim_matrix), dim=-1) - 1  # Subtract self-similarity

        loss = -torch.log(nominator / denominator)
        return loss.mean()
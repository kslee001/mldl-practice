import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, img_size):
        self.X = X
        self.Y = Y
        self.img_size = img_size

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        img, bbox = self.transform(
            x=self.X[idx],
            bbox=self.Y[idx]['boxes'],
            img_size=self.img_size,
        )
        labels = torch.tensor(self.Y[idx]['labels'])
        # diffs = self.Y[idx]['difficulties']

        return img, bbox, labels#, diffs


    def transform(self, x, bbox, img_size=(300, 300)):
        resize_func = A.Resize(height=img_size[0], width=img_size[1]) 
        
        # image transformation
        img = cv2.imread(x)
        h, w = img.shape[:2]
        img = resize_func(image=img)['image']

        """Some augmentations applied here"""

        img = ToTensorV2()(image=img)['image']/255.0 # tensor

        # bbox transformation -> fractional form
        bbox = np.array(bbox)/np.array([h, w, h, w])
        bbox = torch.from_numpy(bbox)

        return img, bbox


    def normalize(self, x):
        return 


    def collate_fn(self, batch):
        imgs = list()
        boxes = list()
        labels = list()

        for b in batch:
            imgs.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
        
        imgs = torch.stack(imgs, dim=0)
    
        return imgs, boxes, labels
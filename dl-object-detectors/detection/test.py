import os
import torch
import json
import xml.etree.ElementTree as ET

import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split


voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

# Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}



def main():
    data_directory = "/home/gyuseonglee/workspace/dataset/voc2007"
    directory = "./voc_data"

    # preprocessing
    if not os.path.exists(directory):
        preprocess_data(
            data_directory=data_directory, 
            processed_directory=directory, 
        )

    # load data
    train_data = load_data(directory, split='train')
    train_imgs, train_anns = train_data['images'], train_data['annotations']
    test_data = load_data(directory, split='test')
    test_imgs, test_anns = test_data['images'], test_data['annotations']

    # train valid split
    train_imgs, valid_imgs, train_anns, valid_anns = train_test_split(train_imgs, train_anns, test_size=0.2, random_state=1005)

    dataset = BaseDataset(X=train_imgs, Y=train_anns, img_size=(300,300))
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=4,
        collate_fn=dataset.collate_fn
    )

    batch = next(iter(loader))
    img, bbox, label = batch[0], batch[1], batch[2]
    print(img.shape)
    print(bbox)
    print(label)

    return


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
        labels = self.Y[idx]['labels']
        # diffs = self.Y[idx]['difficulties']

        return img, bbox, labels#, diffs


    def transform(self, x, bbox, img_size=(300, 300)):
        resize_func = A.Resize(height=img_size[0], width=img_size[1]) 
        
        # image transformation
        img = cv2.imread(x)
        h, w = img.shape[:2]
        img = resize_func(image=img)['image']

        """Some augmentations applied here"""

        img = ToTensorV2()(image=img)['image'] # tensor

        # bbox transformation -> fractional form
        bbox = np.array(bbox)/np.array([h, w, h, w])
        bbox = torch.from_numpy(bbox)

        return img, bbox

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



class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        





def load_data(directory, split='train'):
    assert split in ['train', 'test']
    D = {
        'label_map':f"{directory}/label_map.json",
        'images':f"{directory}/{split}_images.json",
        'annotations':f"{directory}/{split}_annotations.json",
    }

    for name, path in D.items():
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)   
            D[name] = data

    return D




def preprocess_data(data_directory, processed_directory="./"):
    # parse a single xml file
    def parse_annotation(path):
        tree = ET.parse(path)
        root = tree.getroot()

        boxes = list()
        labels = list()
        difficulties = list()
        for obj in root.iter('object'):
            difficult = int(obj.find('difficult').text == '1')
            label = obj.find('name').text.lower().strip()

            if label not in label_map:
                continue
            
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text) - 1
            ymin = int(bbox.find('ymin').text) - 1
            xmax = int(bbox.find('xmax').text) - 1
            ymax = int(bbox.find('ymax').text) - 1

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label_map[label])
            difficulties.append(difficult)

        return {'boxes':boxes, 'labels':labels, 'difficulties':difficulties}


    for split in ['train', 'test']:
        if split == 'train':
            ans = f"{data_directory}/{split}/ImageSets/Main/train_trainval.txt"
        elif split == 'test':
            ans = f"{data_directory}/{split}/ImageSets/Main/test.txt"

        # training data
        images = list()
        objects = list()
        n_objects = 0
        with open(ans, 'r') as f:
            indices = f.read().splitlines()

        for idx_line in indices:
            # 000005 -1
            idx = idx_line.split(" ")[0]
            
            # store image
            img_file = f"{data_directory}/{split}/JPEGImages/{str(idx).zfill(6)}.jpg"
            images.append(img_file)

            # store objects
            objs = parse_annotation(path=f"{data_directory}/{split}/Annotations/{str(idx).zfill(6)}.xml")
            n_objects += len(objs)
            objects.append(objs)

        assert len(objects) == len(images)


        # save file
        if not os.path.exists(processed_directory):
            os.makedirs(processed_directory)

        with open(f"{processed_directory}/{split}_images.json", 'w') as f:
            json.dump(images, f)

        with open(f"{processed_directory}/{split}_annotations.json", 'w') as f:
            json.dump(objects, f)

        with open(f"{processed_directory}/label_map.json", 'w') as f:
            json.dump(label_map, f)


    print(f"[INFO] data saved at {processed_directory}")

    





if __name__ == '__main__':


    main()
import os
import json
import xml.etree.ElementTree as ET

import torch


from sklearn.model_selection import train_test_split


# private
from modules import BaseDataset


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




def get_dataloaders(batch_size, num_workers, valid_ratio, seed, raw_data_directory, processed_directory):
    train_imgs, train_anns = load_data(
        raw_data_directory=raw_data_directory, 
        processed_directory=processed_directory, 
        split='train',
    )
    test_imgs, test_anns = load_data(
        raw_data_directory=raw_data_directory, 
        processed_directory=processed_directory, 
        split='test',
    )

    # train valid split
    train_imgs, valid_imgs, train_anns, valid_anns = train_test_split(train_imgs, train_anns, test_size=valid_ratio, random_state=seed)

    # data loaders
    train_dataset = BaseDataset(X=train_imgs, Y=train_anns, img_size=(300,300))
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn
    )

    valid_dataset = BaseDataset(X=valid_imgs, Y=valid_anns, img_size=(300,300))
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=valid_dataset.collate_fn
    )

    test_dataset = BaseDataset(X=test_imgs, Y=test_anns, img_size=(300,300))
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=test_dataset.collate_fn
    )

    return train_loader, valid_loader, test_loader


def load_data(raw_data_directory, processed_directory, split='train'):

    # preprocessing
    if not os.path.exists(processed_directory):
        preprocess_data(
            raw_data_directory=raw_data_directory, 
            processed_directory=processed_directory, 
        )

    assert split in ['train', 'test']
    D = {
        'label_map':f"{processed_directory}/label_map.json",
        'images':f"{processed_directory}/{split}_images.json",
        'annotations':f"{processed_directory}/{split}_annotations.json",
    }

    for name, path in D.items():
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)   
            D[name] = data


    return D['images'], D['annotations']



def preprocess_data(raw_data_directory, processed_directory="./"):
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
            ans = f"{raw_data_directory}/{split}/ImageSets/Main/train_trainval.txt"
        elif split == 'test':
            ans = f"{raw_data_directory}/{split}/ImageSets/Main/test.txt"

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
            img_file = f"{raw_data_directory}/{split}/JPEGImages/{str(idx).zfill(6)}.jpg"
            images.append(img_file)

            # store objects
            objs = parse_annotation(path=f"{raw_data_directory}/{split}/Annotations/{str(idx).zfill(6)}.xml")
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

    
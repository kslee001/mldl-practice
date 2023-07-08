import os
import glob
import random
import numpy as np

import skimage.io as io
import cv2
from pycocotools.coco import COCO
import torch


mode = 'val'
def main():

    dataset_dir = "/home/gyuseonglee/workspace/dataset/coco-2017/coco2017"
    ann_file = f"{dataset_dir}/annotations/instances_{mode}2017.json"


    coco = COCO(ann_file)
    # print(ann_file)
    # print(coco)

    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)

    print()
    print(get_class_name(77, cats))
    img_ids = coco.getImgIds(catIds=77)
    # print(img_ids)
    img_info = coco.loadImgs(img_ids[np.random.randint(0, len(img_ids))])[0]
    img = io.imread(f"{dataset_dir}/{mode}2017/{img_info['file_name']}")/255.0
    # print(img)
    # print(img.shape)

    ann_ids = coco.getAnnIds(imgIds=img_info['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    # print(anns)
    coco.showAnns(anns)

    return



def get_class_name(class_id, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == class_id:
            return cats[i]['name']
    return None



if __name__ == '__main__':
    main()
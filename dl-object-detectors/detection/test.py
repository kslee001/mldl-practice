import os
import torch

# private
import functions


HOME_DIRECTORY = os.path.expanduser("~")
BATCH_SIZE = 4
NUM_WORKERS = 16
VALID_RATIO = 0.2
SEED = 1005

def main():
    raw_data_directory = f"{HOME_DIRECTORY}/workspace/dataset/voc2007"
    processed_directory = "./voc_data"

    # data loaders
    train_loader, valid_loader, test_loader = functions.get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, 
        valid_ratio=VALID_RATIO,
        seed=SEED,
        raw_data_directory=raw_data_directory, 
        processed_directory=processed_directory, 
    )


    batch = next(iter(train_loader))
    img, bbox, label = batch[0], batch[1], batch[2]
    print(img.shape)
    print(bbox)
    print(label)

    return









if __name__ == '__main__':


    main()
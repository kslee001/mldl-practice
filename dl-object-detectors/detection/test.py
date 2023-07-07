import os
import torch

# private
import functions
from modules import SSD300
from loss_fn import MultiBoxLoss


HOME_DIRECTORY = os.path.expanduser("~")
BATCH_SIZE = 4
NUM_WORKERS = 16
VALID_RATIO = 0.2
SEED = 1005

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
    img = img.to(device)

    model = SSD300(num_classes=20)
    model.to(device)
    obtimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    criterion = MultiBoxLoss(model.priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.)

    locs, classes = model(img)
    loss = criterion(predicted_locs=locs, predicted_scores=classes, boxes=bbox, labels=label)
    print(loss)
    loss.backward()

    return









if __name__ == '__main__':


    main()
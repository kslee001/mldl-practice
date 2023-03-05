from loadlibs import *

import modules
import functions

# ========= CONFIGURATAION ===================
backbone = 'resnet18'
project_name = 'clock'

configs = dict()


configs['BATCH_SIZE'] = 128
configs['LEARNING_RATE'] = 0.001
configs['EPOCHS'] = 40
configs['TEST_SIZE'] = 0.2
configs['SEED'] = 1203
configs['WEIGHT_DECAY'] = 0.001
configs['DEVICE'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
configs['NUM_GPUS'] = torch.cuda.device_count()
configs['TQDM'] = True
configs['NUM_WORKERS'] = 2

configs['AUGMENTATION'] = True
configs['SIZE'] = 128 # image size

configs['TRAIN_FOLDER'] =  "/home/gyuseonglee/workspace/project_2days/clock/data/train"
configs['TEST1_FOLDER'] =  "/home/gyuseonglee/workspace/project_2days/clock/data/test1"
configs['TEST2_FOLDER'] =  "/home/gyuseonglee/workspace/project_2days/clock/data/test2"

folder_name = f"./checkpoints/{backbone}_{configs['SEED']}"
# ============================================
           
    
def main():
    train, test1, test2 = functions.prepare_data(configs)
    train_loader, val_loader, test1_loader, test2_loader = functions.prepare_loaders(configs, train, test1, test2)

    # set training environment
    model = modules.ClockClassifier(backbone) # backbone
    optimizer = torch.optim.Adam(model.parameters(), lr = configs['LEARNING_RATE'])
    criterion1 = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.CrossEntropyLoss()
    scheduler = None
    best_model = functions.train_fn(configs, model, criterion1, criterion2, optimizer, scheduler, train_loader, val_loader)    
    
    # Inference
    preds1 = functions.inference(configs, best_model, test1_loader)
    preds2 = functions.inference(configs, best_model, test2_loader)

    
    # save outputs
    test1['preds'] = preds1
    test1['hour_pred'] = test1['preds'].str.split(0,2)
    test1['min_pred']  = test1['preds'].str.split(2,4)    
    test2['preds'] = preds2
    test2['hour_pred'] = test2['preds'].str.split(0,2)
    test2['min_pred']  = test2['preds'].str.split(2,4)
    del test1['preds'], test2['preds']
    
    today = datetime.today()
    today = "-".join([str(today.year), str(today.month).zfill(2), str(today.day).zfill(2), str(today.hour).zfill(2)])
    today = today[:10].replace("-", "")
    
    test1.to_csv(f"output_test1_{today}.csv", encoding='utf8', index=False)
    test2.to_csv(f"output_test2_{today}.csv", encoding='utf8', index=False)
    
    return 
    
if __name__ == "__main__":
    preds1, labels1, preds2, labels2 = main()

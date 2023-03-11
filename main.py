from loadlibs import *
import modules
import functions

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

# ========= CONFIGURATAION ===================
backbone = 'resnet18'
project_name = 'clock'

configs = dict()


configs['BATCH_SIZE'] = 128
configs['LEARNING_RATE'] = 0.001
configs['EPOCHS'] = 10
configs['TEST_SIZE'] = 0.2
configs['SEED'] = 1203
configs['WEIGHT_DECAY'] = 0.001
configs['DEVICE'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
configs['NUM_GPUS'] = torch.cuda.device_count()
configs['TQDM'] = True
configs['NUM_WORKERS'] = 2
configs['AUGMENTATION'] = True
configs['SIZE'] = 128 # image size

configs['TRAIN_FOLDER'] =  "./data/train"
configs['TEST1_FOLDER'] =  "./data/test1"
configs['TEST2_FOLDER'] =  "./data/test2"

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
    hours_preds1, mins_preds1 = functions.inference(configs, best_model, test1_loader)
    hours_preds2, mins_preds2 = functions.inference(configs, best_model, test2_loader)

    
    # save outputs
    test1['hour_pred'] = hours_preds1
    test1['min_pred']  = mins_preds1    
    test1['label'] = test1['hour'].astype(str) + test1['min'].astype(str)
    test1['pred']  = test1['hour_pred'].astype(str) + test1['min_pred'].astype(str)

    test2['hour_pred'] = hours_preds2
    test2['min_pred']  = mins_preds2
    test2['label'] = test2['hour'].astype(str) + test2['min'].astype(str)
    test2['pred']  = test2['hour_pred'].astype(str) + test2['min_pred'].astype(str)
    
    today = datetime.today()
    today = "-".join([str(today.year), str(today.month).zfill(2), str(today.day).zfill(2), str(today.hour).zfill(2)])
    today = today[:10].replace("-", "")
        
    print(f"test1 accuracy : {accuracy_score(test1['label'].values, test1['pred'].values)}")
    print(f"test2 accuracy : {accuracy_score(test2['label'].values, test2['pred'].values)}")
    # print(f"test1 f1 score : {f1_score(test1['label'].values, test1['pred'].values)}")
    # print(f"test2 f1 score : {f1_score(test2['label'].values, test2['pred'].values)}")
    
    test1.to_csv(f"output_test1_{today}.csv", encoding='utf8', index=False)
    test2.to_csv(f"output_test2_{today}.csv", encoding='utf8', index=False)

    return 
    
if __name__ == "__main__":
    main()

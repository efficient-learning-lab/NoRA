import torch
from torch.utils.data import DataLoader
from DataSet.MyDataset import MyDataset, load_and_preprocess

def get_dataset(train_config: dict):
    dataset_name = train_config['dataset_name']
    train_data, train_label, test_data, test_label = load_and_preprocess(train_config['data_dir'], dataset_name, train_config['preprocess'])
    train_set = MyDataset(train_data, train_label)
    test_set = MyDataset(test_data, test_label)
    return train_set, test_set

def get_dataloader(train_config: dict):
    train_set, test_set = get_dataset(train_config)
    train_loader = DataLoader(
        train_set, 
        batch_size=train_config['batch_size'], 
        num_workers=train_config['num_workers'],
        shuffle=True,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=train_config['batch_size'], 
        shuffle=False,
        pin_memory=True
    )

    return train_loader, test_loader
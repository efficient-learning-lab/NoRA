import os
import csv
import numpy as np
import pandas as pd
import glob
from scipy import io
import torch
from torch.utils.data import Dataset
import sklearn.preprocessing
import ipdb

npz_files = glob.glob(os.path.join('./Data', '*.npz'))
npz_datanames = [os.path.splitext(os.path.basename(file))[0] for file in npz_files]

mat_files = glob.glob(os.path.join('./Data', '*.mat'))
mat_datanames = [os.path.splitext(os.path.basename(file))[0] for file in mat_files]

dat_files = glob.glob(os.path.join('./Data', '*.data'))
dat_datanames = [os.path.splitext(os.path.basename(file))[0] for file in dat_files]

arff_files = glob.glob(os.path.join('./Data', '*.arff'))
arff_datanames = [os.path.splitext(os.path.basename(file))[0] for file in arff_files]


class MyDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = torch.Tensor(data)
        self.targets = torch.Tensor(label)

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    def __len__(self):
        return len(self.data)
   

def load_dataset(data_dir, dataset_name):
    if dataset_name in npz_datanames:
        path = os.path.join(data_dir, dataset_name+'.npz')
        data=np.load(path)  
        samples = data['X']
        labels = ((data['y']).astype(int)).reshape(-1)

        inliers = samples[labels == 0]
        outliers = samples[labels == 1]

    elif dataset_name in mat_datanames:
        path = os.path.join(data_dir, dataset_name + '.mat')
        data = io.loadmat(path)
        samples = data['X']
        labels = ((data['y']).astype(int)).reshape(-1)

        inliers = samples[labels == 0]
        outliers = samples[labels == 1]

    elif dataset_name in dat_datanames:
        path = os.path.join(data_dir, dataset_name + '.data')
        data = io.loadmat(path)
        samples = data['X']
        labels = ((data['y']).astype(int)).reshape(-1)

        inliers = samples[labels == 0]
        outliers = samples[labels == 1]

    
    elif dataset_name in arff_datanames:
        path = os.path.join(data_dir, dataset_name + '.arff')
        data, _ = io.arff.loadarff(path)
        data = pd.DataFrame(data)
        samples = pd.get_dummies(data.iloc[:, :-1]).to_numpy()
        labels = data.iloc[:, -1].values
        inliers = samples[labels == b'0']
        outliers = samples[labels == b'1']

    return inliers, outliers


def split_and_preprocess(inliers, outliers, preprocess):
    """
    Shuffle, split, and apply preprocessor. 
    """

    # shuffle and split
    np.random.shuffle(inliers)
    num_split = len(inliers) // 2

    train_data = inliers[:int(num_split)] 
    train_label = np.zeros(int(num_split))
    test_data = np.concatenate([inliers[num_split:], outliers], 0)
    test_label = np.zeros(test_data.shape[0])
    test_label[-len(outliers):] = 1

    # preprocessing
    if preprocess == 'standard':
        processor = sklearn.preprocessing.StandardScaler().fit(train_data)
    elif preprocess == 'minmax':
        processor = sklearn.preprocessing.MinMaxScaler().fit(train_data)
    elif preprocess == 'quantile':
        processor = sklearn.preprocessing.QuantileTransformer().fit(train_data)
    elif preprocess == 'none':
        processor = sklearn.preprocessing.FunctionTransformer().fit(train_data)
    train_data = processor.transform(train_data)
    test_data = processor.transform(test_data)

    return train_data, train_label, test_data, test_label   

def load_and_preprocess(data_dir, dataset_name, preprocess):
    inliers, outliers = load_dataset(data_dir, dataset_name)
    train_data, train_label, test_data, test_label = split_and_preprocess(inliers, outliers, preprocess)

    return train_data, train_label, test_data, test_label
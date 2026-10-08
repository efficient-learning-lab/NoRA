import argparse
import logging
import numpy as np
import glob
import os
from scipy import io
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support

npz_files = glob.glob(os.path.join('./Data', '*.npz'))
npz_datanames = [os.path.splitext(os.path.basename(file))[0] for file in npz_files]

mat_files = glob.glob(os.path.join('./Data', '*.mat'))
mat_datanames = [os.path.splitext(os.path.basename(file))[0] for file in mat_files]

dat_files = glob.glob(os.path.join('./Data', '*.data'))
dat_datanames = [os.path.splitext(os.path.basename(file))[0] for file in dat_files]

arff_files = glob.glob(os.path.join('./Data', '*.arff'))
arff_datanames = [os.path.splitext(os.path.basename(file))[0] for file in arff_files]

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='wbc')
    parser.add_argument('--model_type', type=str, default='RECLR')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=None)

    return parser


def load_yaml(args):
    dict_to_import = args.model_type + '.yaml'
    with open(f'configs/{dict_to_import}', 'r') as f:
        configs = yaml.safe_load(f)

    model_config = configs['default']['model_config']
    train_config = configs['default']['train_config']

    if args.dataname in configs:
        for k, v in configs[args.dataname].items():
            if k in model_config:
                model_config[k] = v
            if k in train_config:
                train_config[k] = v


    # Set required fields
    train_config['model_type'] = args.model_type
    train_config['dataset_name'] = args.dataname
    train_config['base_path'] = args.base_path
    train_config['exp_name'] = args.exp_name
    train_config['runs'] = args.runs
    model_config['num_features'] = get_input_dim(args, train_config)

    return model_config, train_config


def build_trainer(model_config, train_config):
    model_type = train_config['model_type']
    module = __import__(f'models.{model_type}.Trainer', fromlist=['Trainer'])
    Trainer = module.Trainer
    
    return Trainer(model_config, train_config)


def get_input_dim(args, config):
    data_dir = config['data_dir']
    
    if args.dataname in npz_datanames:
        path = os.path.join(data_dir, args.dataname + '.npz')
        data = np.load(path)
        return data['X'].shape[-1]
    
    elif args.dataname in mat_datanames:
        path = os.path.join(data_dir, args.dataname + '.mat')
        data = io.loadmat(path)
        return data['X'].shape[-1]
    
    elif args.dataname in dat_datanames:
        path = os.path.join(data_dir, args.dataname + '.data')
        data = pd.read_csv(path, header=None)
        return data['X'].shape[-1]
    
    elif args.dataname in arff_datanames:
        path = os.path.join(data_dir, args.dataname + '.arff')
        data, _ = io.arff.loadarff(path)
        data = pd.DataFrame(data)
        return pd.get_dummies(data.iloc[:, :-1]).shape[-1]
    
    else:
        raise ValueError(f"Unknown dataset {args.dataname}")


def aucPerformance(score, labels):
    roc_auc = roc_auc_score(labels, score)
    ap = average_precision_score(labels, score)
    return roc_auc, ap


def F1Performance(score, target):
    normal_ratio = (target == 0).sum() / len(target)
    score = np.squeeze(score)
    threshold = np.percentile(score, 100 * normal_ratio)
    pred = (score > threshold).astype(int)
    _, _, f1, _ = precision_recall_fscore_support(target, pred, average='binary')
    return f1


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("%(message)s")
    
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger
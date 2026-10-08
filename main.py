import torch
import numpy as np
import os
import time
from utils import get_parser, get_logger, build_trainer, load_yaml

def train_test(model_config, train_config, run):
    train_config['run'] = run
    trainer = build_trainer(model_config, train_config)   
    metrics = trainer.evaluate()
    
    return metrics


def main(args):
    os.makedirs(args.base_path, exist_ok=True)
    
    all_results = []
    model_config, train_config = load_yaml(args)

    logger = get_logger(os.path.join(args.base_path, 'log.log'))
    train_config['logger'] = logger
    train_config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for seed in range(args.runs):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        result = train_test(model_config, train_config, seed) 
        all_results.append(result)

    print(all_results)
        

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.exp_name = args.model_type if args.exp_name is None else args.exp_name
    args.base_path = f"results/{args.exp_name}/{args.dataname}"

    main(args)
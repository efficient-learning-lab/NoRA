import os
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from DataSet.DataLoader import get_dataloader
from models.NoRA.Model import NoRA
from utils import aucPerformance, F1Performance
import copy

class Trainer(object):
    def __init__(self, model_config: dict, train_config: dict):
        self.train_loader, self.test_loader = get_dataloader(train_config)
        self.device = train_config['device']
        self.model = NoRA(**model_config).to(self.device)

        self.use_amp = train_config.get('use_amp', False)
        self.scaler = GradScaler() if self.use_amp else None

        self.sche_gamma = train_config['sche_gamma']
        self.learning_rate = train_config['learning_rate']
        self.logger = train_config['logger']
        self.epochs = train_config['epochs']
        self.patience = train_config['patience']
        self.min_delta = train_config['min_delta']
        self.run = train_config['run']
        self.dataname = train_config.get('dataset_name', 'unknown')
        self.model_config = model_config
        self.train_config = train_config

        self.path = os.path.join(train_config['base_path'], str(train_config['run']))
        os.makedirs(self.path, exist_ok=True)

    def training(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)
        self.model.train()
        print("Training Start.")

        if self.patience is not None:
            best_loss = float('inf')
            patience_cnt = 0
            best_model_state = None
            min_delta = self.min_delta

        for epoch in range(self.epochs):
            running_loss = 0.0
            running_recon_loss = 0.0
            running_contra_loss = 0.0

            for step, (x_input, y_label) in enumerate(self.train_loader):
                x_input = x_input.to(self.device)
                optimizer.zero_grad(set_to_none=True)

                if self.use_amp:
                    with autocast():
                        output = self.model(x_input)
                        loss = output['loss']
                        recon_loss = output['recon_loss']
                        contra_loss = output['contra_loss']

                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    output = self.model(x_input)
                    loss = output['loss']
                    recon_loss = output['recon_loss']
                    contra_loss = output['contra_loss']

                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                running_recon_loss += recon_loss.item()
                running_contra_loss += contra_loss.item()

            scheduler.step()

            avg_loss = running_loss / len(self.train_loader)
            avg_recon_loss = running_recon_loss / len(self.train_loader)
            avg_contra_loss = running_contra_loss / len(self.train_loader)

            info = 'Epoch:[{}]\t loss={:.6f}\t recon_loss={:.6f}\t contra_loss={:.6f}'
            self.logger.info(info.format(epoch, avg_loss, avg_recon_loss, avg_contra_loss))

            if self.patience is not None:
                if avg_loss < best_loss - min_delta:
                    best_loss = avg_loss
                    patience_cnt = 0
                    best_model_state = copy.deepcopy(self.model.state_dict())
                else:
                    patience_cnt += 1
                    if patience_cnt >= self.patience:
                        print(f"\nEarly Stopping: No Improvement for {self.patience} epochs.")
                        if best_model_state is not None:
                            self.model.load_state_dict(best_model_state)
                        return 
        print("Training complete.")


    @torch.no_grad()
    def evaluate(self):
        model = self.model
        model.eval()

        score_list, test_label_list = [], []

        for step, (x_input, y_label) in enumerate(self.test_loader):
            x_input = x_input.to(self.device)
            output = model(x_input)
            recon = output['recon_loss'].data.cpu()
            score_list.append(recon)
            test_label_list.append(y_label)

        model.train()

        score = torch.cat(score_list, axis=0).numpy()
        test_label = torch.cat(test_label_list, axis=0).numpy()

        rauc, ap = aucPerformance(score, test_label)
        f1 = F1Performance(score, test_label)
        return {
            'rauc': rauc, 
            'ap': ap, 
            'f1': f1,
        }
import streamlit as st
import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import log_loss

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F

from utils import *

train_features = pd.read_csv(f'train_features.csv')
train_targets_scored = pd.read_csv(f'train_targets_scored.csv')
train_targets_nonscored = pd.read_csv(f'train_targets_nonscored.csv')

test_features = pd.read_csv(f'test_features.csv')
sample_submission = pd.read_csv(f'sample_submission.csv')

device = torch.device('cpu')

# ref: https://www.kaggle.com/c/lish-moa/discussion/180165
# check if labels for 'ctl_vehicle' are all 0.
train = train_features.merge(train_targets_scored, on='sig_id')
target_cols = [c for c in train_targets_scored.columns if c not in ['sig_id']]
cols = target_cols + ['cp_type']
train[cols].groupby('cp_type').sum().sum(1)

# constrcut train&test except 'cp_type'=='ctl_vehicle' data

train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)

folds = train.copy()
Fold = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[target_cols])):
    folds.loc[val_index, 'fold'] = int(n)
folds['fold'] = folds['fold'].astype(int)

class TrainDataset(Dataset):
    def __init__(self, df, num_features, cat_features, labels):
        self.cont_values = df[num_features].values
        self.cate_values = df[cat_features].values
        self.labels = labels
        
    def __len__(self):
        return len(self.cont_values)

    def __getitem__(self, idx):
        cont_x = torch.FloatTensor(self.cont_values[idx])
        cate_x = torch.LongTensor(self.cate_values[idx])
        label = torch.tensor(self.labels[idx]).float()
        
        return cont_x, cate_x, label
    

class TestDataset(Dataset):
    def __init__(self, df, num_features, cat_features):
        self.cont_values = df[num_features].values
        self.cate_values = df[cat_features].values
        
    def __len__(self):
        return len(self.cont_values)

    def __getitem__(self, idx):
        cont_x = torch.FloatTensor(self.cont_values[idx])
        cate_x = torch.LongTensor(self.cate_values[idx])
        
        return cont_x, cate_x

cat_features = ['cp_time', 'cp_dose']
num_features = [c for c in train.columns if train.dtypes[c] != 'object']
num_features = [c for c in num_features if c not in cat_features]
num_features = [c for c in num_features if c not in target_cols]
target = train[target_cols].values

def cate2num(df):
    df['cp_time'] = df['cp_time'].map({24: 0, 48: 1, 72: 2})
    df['cp_dose'] = df['cp_dose'].map({'D1': 3, 'D2': 4})
    return df

train = cate2num(train)
test = cate2num(test)

class CFG:
    max_grad_norm=1000
    gradient_accumulation_steps=1
    hidden_size=512
    dropout=0.5
    lr=1e-2
    weight_decay=1e-6
    batch_size=32
    epochs=20
    #total_cate_size=5
    #emb_size=4
    num_features=num_features
    cat_features=cat_features
    target_cols=target_cols

class TabularNN(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.mlp = nn.Sequential(
                          nn.Linear(len(CFG.num_features), CFG.hidden_size),
                          nn.BatchNorm1d(CFG.hidden_size),
                          nn.Dropout(CFG.dropout),
                          nn.PReLU(),
                          nn.Linear(CFG.hidden_size, CFG.hidden_size),
                          nn.BatchNorm1d(CFG.hidden_size),
                          nn.Dropout(CFG.dropout),
                          nn.PReLU(),
                          nn.Linear(CFG.hidden_size, len(CFG.target_cols)),
                          )

    def forward(self, cont_x, cate_x):
        # no use of cate_x yet
        x = self.mlp(cont_x)
        return x

def train_fn(train_loader, model, optimizer, epoch, scheduler, device):
    
    losses = AverageMeter()

    model.train()

    for step, (cont_x, cate_x, y) in enumerate(train_loader):
        
        cont_x, cate_x, y = cont_x.to(device), cate_x.to(device), y.to(device)
        batch_size = cont_x.size(0)

        pred = model(cont_x, cate_x)
        
        loss = nn.BCEWithLogitsLoss()(pred, y)
        losses.update(loss.item(), batch_size)

        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps

        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)

        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
        
    return losses.avg


def validate_fn(valid_loader, model, device):
    
    losses = AverageMeter()

    model.eval()
    val_preds = []

    for step, (cont_x, cate_x, y) in enumerate(valid_loader):
        
        cont_x, cate_x, y = cont_x.to(device), cate_x.to(device), y.to(device)
        batch_size = cont_x.size(0)

        with torch.no_grad():
            pred = model(cont_x, cate_x)
            
        loss = nn.BCEWithLogitsLoss()(pred, y)
        losses.update(loss.item(), batch_size)

        val_preds.append(pred.sigmoid().detach().cpu().numpy())

        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps

    val_preds = np.concatenate(val_preds)
        
    return losses.avg, val_preds


def inference_fn(test_loader, model, device):

    model.eval()
    preds = []

    for step, (cont_x, cate_x) in enumerate(test_loader):

        cont_x,  cate_x = cont_x.to(device), cate_x.to(device)

        with torch.no_grad():
            pred = model(cont_x, cate_x)

        preds.append(pred.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds)

    return preds


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def run_single_nn(CFG, test, num_features, cat_features, device, fold_num, seed=42):
    
    # Set seed
    # logger.info(f'Set seed {seed}')
    # seed_everything(seed=seed)

    # # loader
    # trn_idx = folds[folds['fold'] != fold_num].index
    # val_idx = folds[folds['fold'] == fold_num].index
    # train_folds = train.loc[trn_idx].reset_index(drop=True)
    # valid_folds = train.loc[val_idx].reset_index(drop=True)
    # train_target = target[trn_idx]
    # valid_target = target[val_idx]
    # train_dataset = TrainDataset(train_folds, num_features, cat_features, train_target)
    # valid_dataset = TrainDataset(valid_folds, num_features, cat_features, valid_target)
    # train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, 
    #                           num_workers=4, pin_memory=True, drop_last=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False, 
    #                           num_workers=4, pin_memory=True, drop_last=False)

    # # model
    # model = TabularNN(CFG)
    # model.to(device)
    # optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
    #                                           max_lr=1e-2, epochs=CFG.epochs, steps_per_epoch=len(train_loader))

    # # log
    # log_df = pd.DataFrame(columns=(['EPOCH']+['TRAIN_LOSS']+['VALID_LOSS']) )

    # # train & validate
    # best_loss = np.inf
    # for epoch in range(CFG.epochs):
    #     train_loss = train_fn(train_loader, model, optimizer, epoch, scheduler, device)
    #     valid_loss, val_preds = validate_fn(valid_loader, model, device)
    #     log_row = {'EPOCH': epoch, 
    #                'TRAIN_LOSS': train_loss,
    #                'VALID_LOSS': valid_loss,
    #               }
    #     log_df = log_df.append(pd.DataFrame(log_row, index=[0]), sort=False)
    #     #logger.info(log_df.tail(1))
    #     if valid_loss < best_loss:
    #         logger.info(f'epoch{epoch} save best model... {valid_loss}')
    #         best_loss = valid_loss
    #         oof = np.zeros((len(train), len(CFG.target_cols)))
    #         oof[val_idx] = val_preds
    #         torch.save(model.state_dict(), f"fold{fold_num}_seed{seed}.pth")

    # predictions
    test_dataset = TestDataset(test, num_features, cat_features)
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, 
                             num_workers=4, pin_memory=True)
    model = TabularNN(CFG)
    model.load_state_dict(torch.load(f"models/fold{fold_num}_seed{seed}.pth", map_location=torch.device('cpu')))
    model.to(device)
    predictions = inference_fn(test_loader, model, device)
    
    # del
    torch.cuda.empty_cache()

    return predictions


def run_kfold_nn(CFG, test, num_features, cat_features, device, n_fold=5, seed=42):

    predictions = np.zeros((len(test), len(CFG.target_cols)))

    for _fold in range(n_fold):
        _predictions = run_single_nn(CFG, test, num_features, cat_features, device, fold_num=_fold, seed=seed)
        
        predictions += _predictions / n_fold
    
    return predictions




import re
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def data_shaping(
    X_data,
    y_data,
    seq_len,
    label_len,
    pred_len,
    horizon,
):

    window = seq_len
    length = (X_data.shape[0] - (window + horizon)) // pred_len
    win = window // pred_len
    hor = horizon // pred_len

    X = np.array([X_data[(pred_len * i) : (i + win) * pred_len] for i in range(length)])
    y = np.array(
        [
            y_data[(i + win + hor) * pred_len : (i + 1 + win + hor) * pred_len]
            for i in range(length)
        ]
    )

    return X[:, :-label_len], X[:, -label_len:], y


def train_val_test_split(data: np.ndarray, val_threshold, test_threshold):
    train = data[:-(val_threshold + test_threshold)]
    val = data[-(val_threshold + test_threshold):-test_threshold]
    test = data[-test_threshold:]
    return train, val, test


def get_dataloader(
    x_enc, x_dec, y, batch_size, shuffle=True
):
    x_enc = torch.tensor(x_enc, dtype=torch.float)
    x_dec = torch.tensor(x_dec, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    dataset = TensorDataset(x_enc, x_dec, y)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
    )
    return dataloader

def is_version_lower(installed_version, required_version):
    installed_version = re.sub(r"\+.*$", "", installed_version)
    installed_version = tuple([int(x) for x in installed_version.split(".")])
    required_version = tuple([int(x) for x in required_version.split(".")])
    return installed_version < required_version

def fix_seed(seed):
    # random
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), str(path) + "/" + "checkpoint.pth")
        self.val_loss_min = val_loss

def min_max(x, axis=None):
    x_min = x.min(axis=axis, keepdims=True)
    x_max = x.max(axis=axis, keepdims=True)
    return (x - x_min) / (x_max - x_min)


def mm_to_primal_range(data, X, axis=None):
    x_min = X.min(axis=axis, keepdims=True)
    x_max = X.max(axis=axis, keepdims=True)
    return (x_max - x_min) * data + x_min

def MAE(pred, true):
    return np.mean(np.abs(pred - true))
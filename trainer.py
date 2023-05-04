from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from models.utils import EarlyStopping
from models.transformer import Transformer

class trainer(object):
    def __init__(self, args) -> None:
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.optimizer = self._get_optimizer()
        self.loss_fn = nn.MSELoss()

    def _acquire_device(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Use {device}")
        return device

    def _get_optimizer(self):
        opt = optim.Adam(self.model.parameters(), lr=0.001)
        return opt

    def _build_model(self):
        raise NotImplementedError
        return model

    def _learning_process_for_epoch(self, iter, phase):
        raise NotImplementedError

    def predict(self, data_loader):
        raise NotImplementedError
        return preds

    def fit(self, data_loader, setting):

        early_stopping = EarlyStopping(
            patience=5, verbose=False
        )
        scheduler = ReduceLROnPlateau(
            self.optimizer, "min", patience=3, min_lr=1e-5, factor=0.1
        )
        path = Path(self.args.checkpoints).resolve()
        path.mkdir(parents=True, exist_ok=True)

        train_loss_list = []
        val_loss_list = []

        for epoch in range(self.args.epochs):
            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train()
                elif phase == "val":
                    self.model.eval()

                bar_format = "{desc}: {percentage:3.0f}%, {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"
                with torch.set_grad_enabled(phase == "train"):
                    with tqdm(
                        data_loader[phase], unit="batch", bar_format=bar_format
                    ) as iter:

                        iter.set_description(
                            f"Epoch[{epoch+1}/{self.args.epochs}]({phase})"
                        )

                        loss_result = self._learning_process_for_epoch(iter, phase)

                        if phase == "train":
                            train_loss = loss_result
                        elif phase == "val":
                            val_loss = loss_result

            early_stopping(val_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early Stopping!!!")
                break

            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            scheduler.step(val_loss)

        print("load best model...")
        best_model_path = path.joinpath("checkpoint.pth")
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model, train_loss_list, val_loss_list


class trainer_formers(trainer):
    def __init__(self, args) -> None:
        super(trainer_formers, self).__init__(args)

    def _build_model(self):
        model = Transformer(self.args.enc_in, self.args.dec_in, self.args.dec_out, self.args.pred_len)
        return model

    def _learning_process_for_epoch(self, iter, phase):
        running_loss = 0
        for i, (x_enc, x_dec, t) in enumerate(iter):
            pad = torch.zeros(
                x_dec.shape[0], self.args.pred_len, x_dec.shape[-1]
            ).float()
            x_dec = torch.cat([x_dec, pad], dim=1).float()
            x_enc = x_enc.to(self.device)
            x_dec, t = x_dec.to(self.device), t.to(self.device)

            y, _ = self.model(x_enc, x_dec)
            loss_t = self.loss_fn(y, t)
            running_loss += loss_t.item()
            if phase == "train":
                self.optimizer.zero_grad()
                loss_t.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

            iter.set_postfix(
                {
                    "loss": (running_loss / (i + 1)),
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
            )

        running_loss /= i + 1
        return running_loss

    def predict(self, data_loader, setting, do_train=True):
        if not do_train:
            print("loading best model...")
            self.model.load_state_dict(
                torch.load(
                    Path(self.args.checkpoints)
                    .resolve()
                    .joinpath(f"{setting}/checkpoint.pth")
                )
            )

        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for (x_enc, x_dec, t) in data_loader:
                pad = torch.zeros(
                    x_dec.shape[0], self.args.pred_len, x_dec.shape[-1]
                ).float()
                x_dec = torch.cat([x_dec, pad], dim=1).float()
                x_enc, x_dec = x_enc.to(self.device), x_dec.to(self.device)

                y, _ = self.model(x_enc, x_dec)

                pred = y.cpu().numpy()
                preds.append(pred)
                trues.append(t.numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        return preds, trues
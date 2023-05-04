import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from models.utils import (
    data_shaping,
    get_dataloader,
    train_val_test_split,
)
from models.utils import (
    fix_seed,
    is_version_lower,
    min_max,
    mm_to_primal_range,
    MAE,
)

from trainer import trainer_formers

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--seq_len', type=int, default=48)

    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--label_len', type=int, default=24)
    parser.add_argument('--horizon', type=int, default=0)

    parser.add_argument('--enc_in', type=int, default=1)
    parser.add_argument('--dec_in', type=int, default=1)
    parser.add_argument('--dec_out', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()
    fix_seed(2022)

    df = pd.read_csv(
        "./ETDataset/ETT-small/ETTh1.csv",
        usecols=["date", "HUFL"],
    )
    consumption = df["HUFL"].values
    original_data = df.drop("date", 1).values
    print(original_data.shape)
    data_scaled = min_max(original_data, axis=0)

    x_enc, x_dec, y = data_shaping(
        data_scaled,
        data_scaled[:, :1],
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        horizon=args.horizon,
    )

    val_threshold, test_threshold = 100, 100

    X_train_enc, X_val_enc, X_test_enc = train_val_test_split(
        x_enc, val_threshold, test_threshold
    )
    X_train_dec, X_val_dec, X_test_dec = train_val_test_split(
        x_dec, val_threshold, test_threshold
    )
    y_train, y_val, y_test = train_val_test_split(y, val_threshold, test_threshold)
    print(X_train_enc.shape, X_train_dec.shape, y_train.shape)
    train_loader = get_dataloader(
        X_train_enc,
        X_train_dec,
        y_train,
        args.batch_size,
    )
    val_loader = get_dataloader(
        X_val_enc,
        X_val_dec,
        y_val,
        args.batch_size,
    )
    test_loader = get_dataloader(
        X_test_enc,
        X_test_dec,
        y_test,
        args.batch_size,
        shuffle=False,
    )

    trainer = trainer_formers(args)

    data_loader = {"train": train_loader, "val": val_loader}

    model, train_loss_list, val_loss_list = trainer.fit(data_loader, "test_transformer")

    preds = trainer.predict(test_loader, "test_transformer")[0]
    correct_prime = mm_to_primal_range(y_test.reshape(-1, 1), consumption, axis=0)
    preds_prime = mm_to_primal_range(preds.reshape(-1, 1), consumption, axis=0)

    mape = MAE(preds_prime, correct_prime)
    print(f"MAE: {mape:.4f}")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
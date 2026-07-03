import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "global_active_power"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def infer_feature_columns(df, target_column=TARGET_COLUMN):
    excluded = {"date", "observed_minutes", "day_of_week", "month", "day_of_year"}
    numeric_cols = [
        column
        for column in df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
    ]
    if target_column not in numeric_cols:
        raise ValueError(f"Target column {target_column!r} is missing from numeric data.")
    return numeric_cols


def load_daily_dataframe(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. Run `python preprocess.py` first."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    return df


def make_supervised_arrays(
    df,
    input_len,
    horizon,
    train_ratio=0.7,
    target_column=TARGET_COLUMN,
    feature_columns=None,
    stride=1,
):
    if feature_columns is None:
        feature_columns = infer_feature_columns(df, target_column=target_column)

    split_idx = int(len(df) * train_ratio)
    if split_idx <= input_len + horizon:
        raise ValueError("Training split is too short for the selected input/output lengths.")

    scaler = StandardScaler()
    scaler.fit(df.loc[: split_idx - 1, feature_columns])
    scaled = scaler.transform(df[feature_columns]).astype(np.float32)
    target_index = feature_columns.index(target_column)

    train_x, train_y, test_x, test_y = [], [], [], []
    train_dates, test_dates = [], []
    max_start = len(df) - input_len - horizon + 1
    for start in range(0, max_start, stride):
        input_end = start + input_len
        target_end = input_end + horizon
        x = scaled[start:input_end]
        y = scaled[input_end:target_end, target_index]
        dates = df.loc[input_end : target_end - 1, "date"].dt.strftime("%Y-%m-%d").tolist()

        if target_end <= split_idx:
            train_x.append(x)
            train_y.append(y)
            train_dates.append(dates)
        elif input_end >= split_idx:
            test_x.append(x)
            test_y.append(y)
            test_dates.append(dates)

    if not train_x or not test_x:
        raise ValueError(
            "Could not create both train and test windows. "
            "Try a smaller --train-ratio or shorter --horizon."
        )

    metadata = {
        "feature_columns": feature_columns,
        "target_column": target_column,
        "target_mean": float(scaler.mean_[target_index]),
        "target_scale": float(scaler.scale_[target_index]),
        "split_idx": split_idx,
        "train_rows": split_idx,
        "test_rows": len(df) - split_idx,
        "train_windows": len(train_x),
        "test_windows": len(test_x),
        "first_test_dates": test_dates[0],
    }
    return (
        np.stack(train_x),
        np.stack(train_y),
        np.stack(test_x),
        np.stack(test_y),
        train_dates,
        test_dates,
        metadata,
    )


def inverse_target(values, metadata):
    return values * metadata["target_scale"] + metadata["target_mean"]


def compute_metrics(y_true_scaled, y_pred_scaled, metadata):
    y_true = inverse_target(np.asarray(y_true_scaled), metadata)
    y_pred = inverse_target(np.asarray(y_pred_scaled), metadata)
    return {
        "mse": float(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1))),
        "mae": float(mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))),
    }


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_prediction_plot(path, dates, y_true_scaled, y_pred_scaled, metadata, title):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y_true = inverse_target(np.asarray(y_true_scaled), metadata)
    y_pred = inverse_target(np.asarray(y_pred_scaled), metadata)
    x = pd.to_datetime(dates)

    plt.figure(figsize=(12, 5))
    plt.plot(x, y_true, label="Ground Truth", linewidth=2)
    plt.plot(x, y_pred, label="Prediction", linewidth=2)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Daily global active power")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True

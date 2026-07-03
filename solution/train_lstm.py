import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import (
    compute_metrics,
    load_daily_dataframe,
    make_supervised_arrays,
    save_json,
    save_prediction_plot,
    set_seed,
)


def sample_std(values):
    values = list(values)
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def split_train_validation(train_x, train_y, val_ratio):
    if not 0.0 < val_ratio < 0.5:
        raise ValueError("--val-ratio must be between 0 and 0.5.")
    val_size = max(1, int(len(train_x) * val_ratio))
    if len(train_x) - val_size < 1:
        raise ValueError("Validation split leaves no training samples.")
    return (
        train_x[:-val_size],
        train_y[:-val_size],
        train_x[-val_size:],
        train_y[-val_size:],
    )


def save_loss_curve(path, history):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(epochs, train_loss, label="Train loss", linewidth=2)
    ax1.plot(epochs, val_loss, label="Validation loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("SmoothL1 loss")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper right")

    ax2 = ax1.twinx()
    ax2.plot(epochs, [row["lr"] for row in history], label="Learning rate", linestyle="--", alpha=0.6)
    ax2.set_ylabel("Learning rate")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        return torch, nn, DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for LSTM training. Install it with `pip install -r requirements.txt`."
        ) from exc


def build_model(input_size, horizon, hidden_size, num_layers, dropout):
    torch, nn, _, _ = require_torch()

    class LSTMForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            lstm_dropout = dropout if num_layers > 1 else 0.0
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=lstm_dropout,
                batch_first=True,
            )
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, max(8, hidden_size // 2)),
                nn.Tanh(),
                nn.Linear(max(8, hidden_size // 2), 1),
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_size * 2),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, horizon),
            )

        def forward(self, x):
            sequence, (hidden, _) = self.lstm(x)
            attention_weights = torch.softmax(self.attention(sequence), dim=1)
            context = torch.sum(sequence * attention_weights, dim=1)
            features = torch.cat([hidden[-1], context], dim=1)
            return self.head(features)

    return LSTMForecaster()


def evaluate_loss(model, tensor_x, tensor_y, loss_fn):
    model.eval()
    with require_torch()[0].no_grad():
        pred = model(tensor_x)
        return float(loss_fn(pred, tensor_y).item())


def train_one_run(args, horizon, run_id, df):
    torch, nn, DataLoader, TensorDataset = require_torch()
    seed = args.seed + run_id
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    arrays = make_supervised_arrays(
        df,
        input_len=args.input_len,
        horizon=horizon,
        train_ratio=args.train_ratio,
        stride=args.stride,
    )
    train_x, train_y, test_x, test_y, _, test_dates, metadata = arrays
    fit_x, fit_y, val_x, val_y = split_train_validation(train_x, train_y, args.val_ratio)

    train_ds = TensorDataset(torch.from_numpy(fit_x), torch.from_numpy(fit_y))
    val_tensor = torch.from_numpy(val_x).to(device)
    val_target = torch.from_numpy(val_y).to(device)
    test_tensor = torch.from_numpy(test_x).to(device)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(
        input_size=train_x.shape[-1],
        horizon=horizon,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    loss_fn = nn.SmoothL1Loss(beta=args.huber_beta)

    history = []
    best_val_loss = float("inf")
    best_state = None
    patience_left = args.patience
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            if args.input_noise > 0:
                batch_x = batch_x + torch.randn_like(batch_x) * args.input_noise

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(loss.item())

        train_loss = float(np.mean(losses))
        val_loss = evaluate_loss(model, val_tensor, val_target, loss_fn)
        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
            }
        )

        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred = model(test_tensor).cpu().numpy()

    metrics = compute_metrics(test_y, pred, metadata)
    metrics.update(
        {
            "run": run_id + 1,
            "seed": seed,
            "horizon": horizon,
            "best_val_loss": best_val_loss,
            "final_train_loss": history[-1]["train_loss"],
            "final_val_loss": history[-1]["val_loss"],
            "epochs_trained": len(history),
            "train_windows": int(len(fit_x)),
            "val_windows": int(len(val_x)),
            "test_windows": int(len(test_x)),
            "num_parameters": count_parameters(model),
            "device": str(device),
        }
    )

    out_dir = Path(args.output_dir) / f"lstm_h{horizon}" / f"run_{run_id + 1}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "metadata.json", metadata)
    save_json(out_dir / "metrics.json", metrics)
    save_json(
        out_dir / "config.json",
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    )
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    save_loss_curve(out_dir / "loss_curve.png", history)
    np.save(out_dir / "pred_scaled.npy", pred)
    np.save(out_dir / "true_scaled.npy", test_y)

    plot_indices = sorted(set([0, len(test_dates) // 2, len(test_dates) - 1]))
    for plot_idx in plot_indices:
        save_prediction_plot(
            out_dir / f"prediction_sample_{plot_idx + 1}.png",
            test_dates[plot_idx],
            test_y[plot_idx],
            pred[plot_idx],
            metadata,
            title=f"LSTM horizon={horizon}, run={run_id + 1}, sample={plot_idx + 1}",
        )
    save_prediction_plot(
        out_dir / "prediction.png",
        test_dates[-1],
        test_y[-1],
        pred[-1],
        metadata,
        title=f"LSTM horizon={horizon}, run={run_id + 1}, last test sample",
    )
    return metrics


def run_experiments(args):
    df = load_daily_dataframe(args.data)
    all_metrics = []
    for horizon in args.horizons:
        horizon_metrics = []
        for run_id in range(args.runs):
            metrics = train_one_run(args, horizon, run_id, df)
            horizon_metrics.append(metrics)
            all_metrics.append(metrics)
            print(
                f"LSTM horizon={horizon} run={run_id + 1}: "
                f"MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, "
                f"best_val_loss={metrics['best_val_loss']:.4f}, "
                f"epochs={metrics['epochs_trained']}"
            )

        summary = {
            "model": "LSTM",
            "horizon": horizon,
            "runs": args.runs,
            "mse_mean": float(np.mean([m["mse"] for m in horizon_metrics])),
            "mse_std": sample_std([m["mse"] for m in horizon_metrics]),
            "mae_mean": float(np.mean([m["mae"] for m in horizon_metrics])),
            "mae_std": sample_std([m["mae"] for m in horizon_metrics]),
            "best_val_loss_mean": float(np.mean([m["best_val_loss"] for m in horizon_metrics])),
            "epochs_trained_mean": float(np.mean([m["epochs_trained"] for m in horizon_metrics])),
            "num_parameters": int(horizon_metrics[0]["num_parameters"]),
        }
        save_json(Path(args.output_dir) / f"lstm_h{horizon}" / "summary.json", summary)
        print(
            f"LSTM horizon={horizon} summary: "
            f"MSE={summary['mse_mean']:.4f}+/-{summary['mse_std']:.4f}, "
            f"MAE={summary['mae_mean']:.4f}+/-{summary['mae_std']:.4f}"
        )

    metrics_df = pd.DataFrame(all_metrics)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(Path(args.output_dir) / "lstm_metrics.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Question 1: LSTM forecasting.")
    parser.add_argument("--data", type=Path, default=Path("processed") / "daily_power.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "experiments")
    parser.add_argument("--horizons", type=int, nargs="+", default=[90, 365])
    parser.add_argument("--input-len", type=int, default=90)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=24)
    parser.add_argument("--lr-patience", type=int, default=6)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--huber-beta", type=float, default=0.5)
    parser.add_argument("--input-noise", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiments(parse_args())

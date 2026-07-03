import argparse
import math
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
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset

        return torch, nn, F, DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for improved model training. Install it with `pip install -r requirements.txt`."
        ) from exc


def build_model(input_size, input_len, horizon, d_model, nhead, num_layers, dropout):
    torch, nn, F, _, _ = require_torch()
    if d_model % nhead != 0:
        raise ValueError("--d-model must be divisible by --nhead.")

    class PositionalEncoding(nn.Module):
        def __init__(self):
            super().__init__()
            pe = torch.zeros(input_len, d_model)
            position = torch.arange(0, input_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32)
                * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, : x.size(1)]

    class DepthwiseCausalConv1d(nn.Module):
        def __init__(self, channels, kernel_size, dilation):
            super().__init__()
            self.left_padding = (kernel_size - 1) * dilation
            self.depthwise = nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
                groups=channels,
            )
            self.pointwise = nn.Conv1d(
                channels,
                channels,
                kernel_size=1,
            )

        def forward(self, x):
            x = F.pad(x, (self.left_padding, 0))
            return self.pointwise(self.depthwise(x))

    class MultiScaleCausalBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.branches = nn.ModuleList(
                [
                    DepthwiseCausalConv1d(d_model, kernel_size=3, dilation=1),
                    DepthwiseCausalConv1d(d_model, kernel_size=5, dilation=2),
                    DepthwiseCausalConv1d(d_model, kernel_size=7, dilation=3),
                ]
            )
            self.projection = nn.Conv1d(d_model * len(self.branches), d_model, kernel_size=1)
            self.norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            self.gate = nn.Parameter(torch.tensor(-2.0))

        def forward(self, x):
            x_t = x.transpose(1, 2)
            branches = [torch.relu(branch(x_t)) for branch in self.branches]
            merged = torch.cat(branches, dim=1)
            local = self.projection(merged).transpose(1, 2)
            out = x + torch.sigmoid(self.gate) * self.dropout(local)
            return self.norm(out)

    class MSConvTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_norm = nn.LayerNorm(input_size)
            self.input_projection = nn.Linear(input_size, d_model)
            self.local_encoder = MultiScaleCausalBlock()
            self.position = PositionalEncoding()
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 3,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.global_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model * 3),
                nn.Linear(d_model * 3, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, horizon),
            )

        def forward(self, x):
            x = self.input_norm(x)
            x = self.input_projection(x)
            x = self.local_encoder(x)
            x = self.position(x)
            encoded = self.global_encoder(x)
            pooled = torch.cat(
                [
                    encoded[:, -1, :],
                    encoded.mean(dim=1),
                    encoded.max(dim=1).values,
                ],
                dim=1,
            )
            return self.head(pooled)

    return MSConvTransformer()


def evaluate_loss(model, tensor_x, tensor_y, loss_fn):
    torch = require_torch()[0]
    model.eval()
    with torch.no_grad():
        pred = model(tensor_x)
        return float(loss_fn(pred, tensor_y).item())


def train_one_run(args, horizon, run_id, df):
    torch, nn, _, DataLoader, TensorDataset = require_torch()
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
        input_len=args.input_len,
        horizon=horizon,
        d_model=args.d_model,
        nhead=args.nhead,
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
        val_pred = model(val_tensor).cpu().numpy()
        pred = model(test_tensor).cpu().numpy()

    val_metrics = compute_metrics(val_y, val_pred, metadata)
    metrics = compute_metrics(test_y, pred, metadata)
    metrics.update(
        {
            "run": run_id + 1,
            "seed": seed,
            "horizon": horizon,
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
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

    out_dir = Path(args.output_dir) / f"improved_h{horizon}" / f"run_{run_id + 1}"
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
            title=f"MSConvTransformer horizon={horizon}, run={run_id + 1}, sample={plot_idx + 1}",
        )
    save_prediction_plot(
        out_dir / "prediction.png",
        test_dates[-1],
        test_y[-1],
        pred[-1],
        metadata,
        title=f"MSConvTransformer horizon={horizon}, run={run_id + 1}, last test sample",
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
                f"MSConvTransformer horizon={horizon} run={run_id + 1}: "
                f"MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, "
                f"val_MAE={metrics['val_mae']:.4f}, "
                f"epochs={metrics['epochs_trained']}"
            )

        summary = {
            "model": "MSConvTransformer",
            "horizon": horizon,
            "runs": args.runs,
            "mse_mean": float(np.mean([m["mse"] for m in horizon_metrics])),
            "mse_std": sample_std([m["mse"] for m in horizon_metrics]),
            "mae_mean": float(np.mean([m["mae"] for m in horizon_metrics])),
            "mae_std": sample_std([m["mae"] for m in horizon_metrics]),
            "val_mse_mean": float(np.mean([m["val_mse"] for m in horizon_metrics])),
            "val_mae_mean": float(np.mean([m["val_mae"] for m in horizon_metrics])),
            "best_val_loss_mean": float(np.mean([m["best_val_loss"] for m in horizon_metrics])),
            "epochs_trained_mean": float(np.mean([m["epochs_trained"] for m in horizon_metrics])),
            "num_parameters": int(horizon_metrics[0]["num_parameters"]),
        }
        save_json(Path(args.output_dir) / f"improved_h{horizon}" / "summary.json", summary)
        print(
            f"MSConvTransformer horizon={horizon} summary: "
            f"MSE={summary['mse_mean']:.4f}+/-{summary['mse_std']:.4f}, "
            f"MAE={summary['mae_mean']:.4f}+/-{summary['mae_std']:.4f}"
        )

    metrics_df = pd.DataFrame(all_metrics)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(Path(args.output_dir) / "improved_metrics.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Question 3: improved MSConvTransformer forecasting.")
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
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
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

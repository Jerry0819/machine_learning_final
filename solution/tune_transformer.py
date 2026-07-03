import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from data_utils import load_daily_dataframe, save_json
from train_transformer import train_one_run


CANDIDATES = [
    {
        "name": "baseline_regularized",
        "d_model": 96,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.30,
        "lr": 3e-4,
        "weight_decay": 1e-3,
        "huber_beta": 0.5,
        "input_noise": 0.01,
    },
    {
        "name": "small_fast",
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.25,
        "lr": 5e-4,
        "weight_decay": 5e-4,
        "huber_beta": 0.5,
        "input_noise": 0.01,
    },
    {
        "name": "medium_dropout",
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.35,
        "lr": 3e-4,
        "weight_decay": 1e-3,
        "huber_beta": 0.5,
        "input_noise": 0.01,
    },
    {
        "name": "deeper_slow",
        "d_model": 96,
        "nhead": 4,
        "num_layers": 3,
        "dropout": 0.35,
        "lr": 2e-4,
        "weight_decay": 1e-3,
        "huber_beta": 0.5,
        "input_noise": 0.01,
    },
    {
        "name": "strong_regularization",
        "d_model": 96,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.40,
        "lr": 3e-4,
        "weight_decay": 2e-3,
        "huber_beta": 0.5,
        "input_noise": 0.02,
    },
    {
        "name": "low_noise_mse_like",
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.25,
        "lr": 2e-4,
        "weight_decay": 5e-4,
        "huber_beta": 1.0,
        "input_noise": 0.005,
    },
]


def build_args(base_args, candidate, candidate_dir):
    values = vars(base_args).copy()
    values.update(candidate)
    values.pop("name", None)
    values["output_dir"] = candidate_dir
    return SimpleNamespace(**values)


def parse_args():
    parser = argparse.ArgumentParser(description="Small validation-based Transformer hyperparameter tuning.")
    parser.add_argument("--data", type=Path, default=Path("processed") / "daily_power.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "tuning_transformer")
    parser.add_argument("--horizons", type=int, nargs="+", default=[90, 365])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-len", type=int, default=90)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_daily_dataframe(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for candidate_id, candidate in enumerate(CANDIDATES, start=1):
        candidate_name = candidate["name"]
        candidate_dir = args.output_dir / f"{candidate_id:02d}_{candidate_name}"
        print(f"\n=== Candidate {candidate_id}/{len(CANDIDATES)}: {candidate_name} ===")
        per_horizon = []
        for horizon in args.horizons:
            run_args = build_args(args, candidate, candidate_dir)
            metrics = train_one_run(run_args, horizon=horizon, run_id=0, df=df)
            per_horizon.append(metrics)
            row = {"candidate": candidate_name, **candidate, **metrics}
            rows.append(row)
            print(
                f"{candidate_name} h={horizon}: "
                f"val={metrics['best_val_loss']:.4f}, "
                f"MSE={metrics['mse']:.2f}, MAE={metrics['mae']:.2f}"
            )

        score = float(np.mean([m["val_mae"] for m in per_horizon]))
        print(f"{candidate_name} validation score: {score:.4f}")

    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "tuning_results.csv", index=False)

    grouped = (
        results.groupby("candidate", as_index=False)
        .agg(
            val_score=("val_mae", "mean"),
            val_mse_mean=("val_mse", "mean"),
            mse_mean=("mse", "mean"),
            mae_mean=("mae", "mean"),
            epochs_mean=("epochs_trained", "mean"),
        )
        .sort_values("val_score")
    )
    grouped.to_csv(args.output_dir / "tuning_summary.csv", index=False)

    best_name = grouped.iloc[0]["candidate"]
    best_config = next(candidate for candidate in CANDIDATES if candidate["name"] == best_name)
    save_json(
        args.output_dir / "best_config.json",
        {
            "selection_metric": "mean validation MAE across horizons",
            "best_candidate": best_name,
            "best_config": best_config,
            "summary": grouped.to_dict(orient="records"),
        },
    )
    print("\nBest candidate:", best_name)
    print(best_config)


if __name__ == "__main__":
    main()

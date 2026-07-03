# Household Power Forecasting

This directory contains the preprocessing pipeline and code for the first two
questions of the homework:

1. LSTM forecasting.
2. Transformer forecasting.
3. Improved MSConvTransformer forecasting.

The models use the latest 90 days as input and predict either the next 90 days
or the next 365 days. The two horizons are trained separately.

## Files

- `preprocess.py`: aggregates the minute-level power data into daily rows and
  merges monthly weather features.
- `data_utils.py`: scaling, sliding-window generation, metrics, and plotting.
- `train_lstm.py`: question 1 implementation.
- `train_transformer.py`: question 2 implementation.
- `train_improved.py`: question 3 implementation.
- `third_question_model.md`: improved model description, rationale, and references.
- `requirements.txt`: Python dependencies.

## Usage

```bash
python preprocess.py
python train_lstm.py --runs 5 --horizons 90 365
python train_transformer.py --runs 5 --horizons 90 365
python train_improved.py --runs 5 --horizons 90 365
```

If the data is not in this directory, pass paths explicitly:

```bash
python preprocess.py ^
  --power-path "C:\path\to\household_power_consumption.txt" ^
  --weather-path "C:\path\to\MENSQ_01_previous-1950-2024.csv"
```

Outputs are written under:

- `processed/daily_power.csv`
- `processed/daily_train.csv`
- `processed/daily_test.csv`
- `outputs/experiments/*/metrics.json`
- `outputs/experiments/*/summary.json`
- `outputs/experiments/*/prediction.png`

The summary files contain the mean and standard deviation of MSE and MAE over
the requested runs.

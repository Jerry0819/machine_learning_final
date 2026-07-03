from pathlib import Path

import tune_transformer


tune_transformer.CANDIDATES = [
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
        "name": "dropout_025",
        "d_model": 96,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.25,
        "lr": 3e-4,
        "weight_decay": 1e-3,
        "huber_beta": 0.5,
        "input_noise": 0.01,
    },
    {
        "name": "dropout_035",
        "d_model": 96,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.35,
        "lr": 3e-4,
        "weight_decay": 1e-3,
        "huber_beta": 0.5,
        "input_noise": 0.01,
    },
    {
        "name": "lr_00025",
        "d_model": 96,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.30,
        "lr": 2.5e-4,
        "weight_decay": 1e-3,
        "huber_beta": 0.5,
        "input_noise": 0.01,
    },
    {
        "name": "lr_0004",
        "d_model": 96,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.30,
        "lr": 4e-4,
        "weight_decay": 1e-3,
        "huber_beta": 0.5,
        "input_noise": 0.01,
    },
    {
        "name": "wd_0007",
        "d_model": 96,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.30,
        "lr": 3e-4,
        "weight_decay": 7e-4,
        "huber_beta": 0.5,
        "input_noise": 0.01,
    },
    {
        "name": "wd_0015",
        "d_model": 96,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.30,
        "lr": 3e-4,
        "weight_decay": 1.5e-3,
        "huber_beta": 0.5,
        "input_noise": 0.01,
    },
]


if __name__ == "__main__":
    tune_transformer.main()

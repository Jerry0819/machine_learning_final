import argparse
from pathlib import Path

import numpy as np
import pandas as pd


POWER_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

RENAME_COLUMNS = {
    "Global_active_power": "global_active_power",
    "Global_reactive_power": "global_reactive_power",
    "Voltage": "voltage",
    "Global_intensity": "global_intensity",
    "Sub_metering_1": "sub_metering_1",
    "Sub_metering_2": "sub_metering_2",
    "Sub_metering_3": "sub_metering_3",
}

WEATHER_COLUMNS = ["RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]


def first_existing(candidates):
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return None


def find_default_power_path(base_dir):
    desktop_final = Path.home() / "Desktop" / "machine-learning-homework" / "final"
    candidates = [
        base_dir / "household_power_consumption.txt",
        base_dir / "data" / "household_power_consumption.txt",
        base_dir
        / "individual+household+electric+power+consumption"
        / "household_power_consumption.txt",
        desktop_final
        / "individual+household+electric+power+consumption"
        / "household_power_consumption.txt",
    ]
    return first_existing(candidates)


def find_default_weather_path(base_dir):
    desktop_final = Path.home() / "Desktop" / "machine-learning-homework" / "final"
    candidates = [
        base_dir / "MENSQ_94_previous-1950-2024.csv",
        base_dir / "MENSQ_94_previous-1950-2024.csv" / "MENSQ_94_previous-1950-2024.csv",
        base_dir / "data" / "MENSQ_94_previous-1950-2024.csv",
        base_dir / "data" / "MENSQ_94_previous-1950-2024.csv" / "MENSQ_94_previous-1950-2024.csv",
        desktop_final / "MENSQ_94_previous-1950-2024.csv",
        desktop_final / "MENSQ_94_previous-1950-2024.csv" / "MENSQ_94_previous-1950-2024.csv",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return path
        if path.is_dir():
            inner = path / path.name
            if inner.is_file():
                return inner
    return None


def aggregate_power_by_day(power_path, chunksize=500_000, min_observed_minutes=1000):
    daily_parts = []
    usecols = ["Date"] + POWER_COLUMNS
    for chunk in pd.read_csv(
        power_path,
        sep=";",
        usecols=usecols,
        na_values=["?"],
        chunksize=chunksize,
        low_memory=False,
    ):
        # Daily aggregation only needs the Date field; parsing Time for every
        # minute makes the preprocessing several times slower.
        chunk = chunk.assign(
            date=pd.to_datetime(chunk["Date"], format="%d/%m/%Y", errors="coerce")
        )
        chunk = chunk.dropna(subset=["date"])
        for column in POWER_COLUMNS:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")

        grouped = chunk.groupby("date").agg(
            {
                "Global_active_power": ["sum", "count"],
                "Global_reactive_power": "sum",
                "Voltage": "mean",
                "Global_intensity": "mean",
                "Sub_metering_1": "sum",
                "Sub_metering_2": "sum",
                "Sub_metering_3": "sum",
            }
        )
        grouped.columns = [
            "Global_active_power",
            "observed_minutes",
            "Global_reactive_power",
            "Voltage",
            "Global_intensity",
            "Sub_metering_1",
            "Sub_metering_2",
            "Sub_metering_3",
        ]
        grouped = grouped.reset_index()
        daily_parts.append(grouped)

    if not daily_parts:
        raise ValueError("No rows were read from the power data file.")

    daily = pd.concat(daily_parts, ignore_index=True)
    daily = daily.groupby("date").agg(
        {
            "Global_active_power": "sum",
            "Global_reactive_power": "sum",
            "Voltage": "mean",
            "Global_intensity": "mean",
            "Sub_metering_1": "sum",
            "Sub_metering_2": "sum",
            "Sub_metering_3": "sum",
            "observed_minutes": "sum",
        }
    )
    daily = daily.reset_index()
    daily = daily.rename(columns=RENAME_COLUMNS)
    daily = daily[daily["observed_minutes"] >= min_observed_minutes].copy()
    daily["sub_metering_remainder"] = (
        daily["global_active_power"] * 1000.0 / 60.0
        - daily["sub_metering_1"]
        - daily["sub_metering_2"]
        - daily["sub_metering_3"]
    )
    daily["sub_metering_remainder"] = daily["sub_metering_remainder"].clip(lower=0)
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


def load_monthly_weather(weather_path, station_id=None):
    if weather_path is None:
        return None
    weather_path = Path(weather_path)
    if not weather_path.is_file():
        return None

    needed = ["NUM_POSTE", "NOM_USUEL", "AAAAMM"] + WEATHER_COLUMNS
    weather = pd.read_csv(
        weather_path,
        sep=";",
        usecols=lambda name: name in needed,
        low_memory=False,
    )
    if weather.empty or "AAAAMM" not in weather.columns:
        return None

    if station_id is not None and "NUM_POSTE" in weather.columns:
        weather = weather[weather["NUM_POSTE"].astype(str) == str(station_id)]
    elif "NUM_POSTE" in weather.columns and weather["NUM_POSTE"].nunique() > 1:
        first_station = weather["NUM_POSTE"].dropna().iloc[0]
        weather = weather[weather["NUM_POSTE"] == first_station]

    weather["AAAAMM"] = weather["AAAAMM"].astype(str).str.slice(0, 6)
    weather["year_month"] = pd.to_datetime(
        weather["AAAAMM"], format="%Y%m", errors="coerce"
    ).dt.to_period("M")
    for column in WEATHER_COLUMNS:
        if column in weather.columns:
            weather[column] = pd.to_numeric(weather[column], errors="coerce")
        else:
            weather[column] = np.nan
    weather = weather.groupby("year_month", as_index=False)[WEATHER_COLUMNS].mean()
    weather = weather.rename(columns={column: column.lower() for column in WEATHER_COLUMNS})
    return weather


def add_time_features(daily):
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["day_of_year"] = daily["date"].dt.dayofyear
    daily["dow_sin"] = np.sin(2 * np.pi * daily["day_of_week"] / 7.0)
    daily["dow_cos"] = np.cos(2 * np.pi * daily["day_of_week"] / 7.0)
    daily["month_sin"] = np.sin(2 * np.pi * daily["month"] / 12.0)
    daily["month_cos"] = np.cos(2 * np.pi * daily["month"] / 12.0)
    daily["doy_sin"] = np.sin(2 * np.pi * daily["day_of_year"] / 366.0)
    daily["doy_cos"] = np.cos(2 * np.pi * daily["day_of_year"] / 366.0)
    return daily


def fill_daily_gaps(daily):
    daily = daily.sort_values("date").set_index("date")
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index)
    daily.index.name = "date"
    numeric_columns = daily.select_dtypes(include=[np.number]).columns
    daily[numeric_columns] = daily[numeric_columns].interpolate(
        method="time", limit_direction="both"
    )
    daily[numeric_columns] = daily[numeric_columns].ffill().bfill()
    return daily.reset_index()


def preprocess(
    power_path,
    weather_path,
    output_dir,
    train_ratio=0.7,
    chunksize=500_000,
    min_observed_minutes=1000,
    station_id=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily = aggregate_power_by_day(
        power_path,
        chunksize=chunksize,
        min_observed_minutes=min_observed_minutes,
    )

    weather = load_monthly_weather(weather_path, station_id=station_id)
    daily["year_month"] = daily["date"].dt.to_period("M")
    if weather is not None:
        daily = daily.merge(weather, on="year_month", how="left")
    else:
        for column in WEATHER_COLUMNS:
            daily[column.lower()] = np.nan
    daily = daily.drop(columns=["year_month"])
    daily = fill_daily_gaps(daily)

    weather_feature_cols = [column.lower() for column in WEATHER_COLUMNS]
    for column in weather_feature_cols:
        if column not in daily.columns:
            daily[column] = 0.0
        if daily[column].isna().all():
            daily[column] = 0.0
        else:
            daily[column] = daily[column].ffill().bfill()
            daily[column] = daily[column].fillna(daily[column].median())

    daily = add_time_features(daily)
    daily = daily.sort_values("date").reset_index(drop=True)

    split_idx = int(len(daily) * train_ratio)
    daily_path = output_dir / "daily_power.csv"
    train_path = output_dir / "daily_train.csv"
    test_path = output_dir / "daily_test.csv"
    daily.to_csv(daily_path, index=False)
    daily.iloc[:split_idx].to_csv(train_path, index=False)
    daily.iloc[split_idx:].to_csv(test_path, index=False)

    print(f"Saved daily data: {daily_path} ({len(daily)} rows)")
    print(f"Saved train split: {train_path} ({split_idx} rows)")
    print(f"Saved test split: {test_path} ({len(daily) - split_idx} rows)")
    return daily


def parse_args():
    base_dir = Path.cwd()
    default_power = find_default_power_path(base_dir)
    default_weather = find_default_weather_path(base_dir)
    parser = argparse.ArgumentParser(description="Preprocess household power data.")
    parser.add_argument("--power-path", type=Path, default=default_power)
    parser.add_argument("--weather-path", type=Path, default=default_weather)
    parser.add_argument("--output-dir", type=Path, default=base_dir / "processed")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--min-observed-minutes", type=int, default=1000)
    parser.add_argument("--station-id", type=str, default=94034001)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.power_path is None or not Path(args.power_path).is_file():
        raise FileNotFoundError(
            "Power data file was not found. Pass --power-path household_power_consumption.txt"
        )
    preprocess(
        power_path=args.power_path,
        weather_path=args.weather_path,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        chunksize=args.chunksize,
        min_observed_minutes=args.min_observed_minutes,
        station_id=args.station_id,
    )


if __name__ == "__main__":
    main()

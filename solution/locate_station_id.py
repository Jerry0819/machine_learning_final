import numpy as np
import pandas as pd
from pathlib import Path

WEATHER_COLUMNS = ["RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]

# Sceaux 大致坐标
SCEAUX_LAT = 48.7788
SCEAUX_LON = 2.2906

weather_files = [
    "MENSQ_75_previous-1950-2024.csv.gz",
    "MENSQ_78_previous-1950-2024.csv.gz",
    "MENSQ_91_previous-1950-2024.csv.gz",
    "MENSQ_92_previous-1950-2024.csv.gz",
    "MENSQ_94_previous-1950-2024.csv.gz",
]

usecols = ["NUM_POSTE", "NOM_USUEL", "LAT", "LON", "AAAAMM"] + WEATHER_COLUMNS

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(a))

parts = []

for file in weather_files:
    path = Path(file)
    if not path.exists():
        print(f"Missing file: {path}")
        continue

    df = pd.read_csv(
        path,
        sep=";",
        usecols=lambda col: col in usecols,
        compression="infer",
        low_memory=False,
    )

    df["AAAAMM"] = df["AAAAMM"].astype(str).str.slice(0, 6)
    df["AAAAMM"] = pd.to_numeric(df["AAAAMM"], errors="coerce")

    # 只保留你的电力数据时间段：2006-12 到 2010-11
    df = df[(df["AAAAMM"] >= 200612) & (df["AAAAMM"] <= 201011)]

    parts.append(df)

weather = pd.concat(parts, ignore_index=True)

# 转成数值
weather["LAT"] = pd.to_numeric(weather["LAT"], errors="coerce")
weather["LON"] = pd.to_numeric(weather["LON"], errors="coerce")

for col in WEATHER_COLUMNS:
    weather[col] = pd.to_numeric(weather[col], errors="coerce")

# 按气象站统计覆盖情况
stations = (
    weather
    .groupby(["NUM_POSTE", "NOM_USUEL", "LAT", "LON"], as_index=False)
    .agg(
        months=("AAAAMM", "nunique"),
        rr_months=("RR", lambda s: s.notna().sum()),
        fog_months=("NBJBROU", lambda s: s.notna().sum()),
    )
)

stations["distance_km"] = haversine_km(
    SCEAUX_LAT,
    SCEAUX_LON,
    stations["LAT"],
    stations["LON"],
)

# 47个月：2006-12 到 2010-11
# 可以要求至少有 40 个月数据，避免选到缺失太多的站
candidates = stations[
    (stations["months"] >= 40)
    & (stations["rr_months"] >= 40)
].sort_values("distance_km")

print(candidates.head(10))

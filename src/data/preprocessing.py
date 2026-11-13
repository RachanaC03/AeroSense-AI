
from pathlib import Path

import numpy as np
import pandas as pd


# Project dataset location
RAW_DIR = Path("data/raw")

# NASA C-MAPSS dataset has 26 columns:
# 2 engine identifiers + 3 operating settings + 21 sensors
INDEX_COLS = ["engine_id", "cycle"]

SETTING_COLS = [
    "setting_1",
    "setting_2",
    "setting_3",
]

SENSOR_COLS = [f"s{i}" for i in range(1, 22)]

COLUMNS = INDEX_COLS + SETTING_COLS + SENSOR_COLS


def load_split(
    subset: str = "FD001",
    split: str = "train",
) -> pd.DataFrame:
    """Load a training or test file for one C-MAPSS subset."""

    if subset not in {"FD001", "FD002", "FD003", "FD004"}:
        raise ValueError(f"Invalid subset: {subset}")

    if split not in {"train", "test"}:
        raise ValueError(f"Invalid split: {split}")

    path = RAW_DIR / f"{split}_{subset}.txt"

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )

    return df


def load_test_rul(subset: str = "FD001") -> np.ndarray:
    """Load true RUL values for the final observed cycle of each test engine."""

    if subset not in {"FD001", "FD002", "FD003", "FD004"}:
        raise ValueError(f"Invalid subset: {subset}")

    path = RAW_DIR / f"RUL_{subset}.txt"

    if not path.exists():
        raise FileNotFoundError(f"RUL file not found: {path}")

    rul = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0]

    return rul.to_numpy(dtype=np.int64)


def add_rul(
    df: pd.DataFrame,
    clip: int | None = 125,
) -> pd.DataFrame:
    """
    Add training RUL labels.

    RUL = engine's final training cycle - current cycle.

    Clipping at 125 is an optional modelling choice.
    Use clip=None to retain the original calculated values.
    """

    df = df.copy()

    max_cycle = df.groupby("engine_id")["cycle"].transform("max")

    df["RUL"] = max_cycle - df["cycle"]

    if clip is not None:
        if clip < 0:
            raise ValueError("RUL clip must be non-negative or None.")

        df["RUL"] = df["RUL"].clip(upper=clip)

    return df


def inspect_dataset() -> None:
    """Load and print basic information about the FD001 dataset."""

    train = load_split("FD001", "train")
    test = load_split("FD001", "test")
    test_rul = load_test_rul("FD001")

    train = add_rul(train)

    print("========== AEROSENSE DATASET INSPECTION ==========")

    print("\nTRAINING DATA")
    print("Shape:", train.shape)
    print("Number of engines:", train["engine_id"].nunique())
    print("Number of sensors:", len(SENSOR_COLS))
    print("Cycle range:", train["cycle"].min(), "-", train["cycle"].max())

    print("\nTEST DATA")
    print("Shape:", test.shape)
    print("Number of engines:", test["engine_id"].nunique())
    print("RUL labels:", len(test_rul))

    if test["engine_id"].nunique() != len(test_rul):
        raise ValueError(
            "Number of test engines does not match number of RUL labels."
        )

    print("\nMISSING VALUES")
    print("Training missing values:", train.isna().sum().sum())
    print("Test missing values:", test.isna().sum().sum())

    print("\nTRAINING DATA SAMPLE")
    print(train.head())

    print("\nSENSOR STANDARD DEVIATION")
    sensor_std = train[SENSOR_COLS].std()

    print(sensor_std.round(4))

    print("\nCONSTANT OR NEAR-CONSTANT SENSORS")
    constant_sensors = sensor_std[sensor_std < 1e-3]

    if constant_sensors.empty:
        print("No sensors below the selected threshold.")
    else:
        print(constant_sensors.index.tolist())

    print("\nRUL LABEL SUMMARY")
    print(train["RUL"].describe())

    print("\n========== INSPECTION COMPLETE ==========")


if __name__ == "__main__":
    inspect_dataset()

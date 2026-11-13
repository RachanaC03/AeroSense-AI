
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


def split_by_engine(
    df: pd.DataFrame,
    validation_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split complete engines into training and validation sets.

    This prevents windows from the same engine appearing in both sets.
    """

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    engine_ids = np.sort(df["engine_id"].unique())

    train_ids, validation_ids = train_test_split(
        engine_ids,
        test_size=validation_fraction,
        random_state=random_state,
    )

    train_df = df[df["engine_id"].isin(train_ids)].copy()
    validation_df = df[df["engine_id"].isin(validation_ids)].copy()

    return train_df, validation_df


def create_train_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = 30,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create overlapping sequences from labelled training engines.

    Returns:
        X: Sensor sequences, shape (samples, window_size, features).
        y: RUL target for the final cycle of each sequence.
        engine_ids: Engine ID associated with each sequence.
    """

    if window_size < 1:
        raise ValueError("window_size must be at least 1.")

    if stride < 1:
        raise ValueError("stride must be at least 1.")

    required_cols = {"engine_id", "cycle", "RUL", *feature_cols}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    X_windows = []
    y_values = []
    window_engine_ids = []

    for engine_id, engine_df in df.groupby("engine_id", sort=True):
        engine_df = engine_df.sort_values("cycle")

        features = engine_df[feature_cols].to_numpy(dtype=np.float32)
        rul_values = engine_df["RUL"].to_numpy(dtype=np.float32)

        if len(engine_df) < window_size:
            continue

        for start in range(0, len(engine_df) - window_size + 1, stride):
            end = start + window_size

            X_windows.append(features[start:end])
            y_values.append(rul_values[end - 1])
            window_engine_ids.append(engine_id)

    if not X_windows:
        raise ValueError(
            "No training windows were created. "
            "Check the window size and engine sequence lengths."
        )

    return (
        np.stack(X_windows),
        np.asarray(y_values, dtype=np.float32),
        np.asarray(window_engine_ids),
    )


def create_test_windows(
    df: pd.DataFrame,
    rul_values: np.ndarray,
    feature_cols: list[str],
    window_size: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create one final window per test engine.

    Each test RUL label corresponds to the final observed cycle
    of the matching engine, ordered by engine ID.
    """

    if window_size < 1:
        raise ValueError("window_size must be at least 1.")

    required_cols = {"engine_id", "cycle", *feature_cols}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    engine_groups = list(df.groupby("engine_id", sort=True))
    engine_ids = np.asarray(
        [engine_id for engine_id, _ in engine_groups]
    )

    rul_values = np.asarray(rul_values, dtype=np.float32)

    if len(engine_ids) != len(rul_values):
        raise ValueError(
            f"Found {len(engine_ids)} test engines but "
            f"{len(rul_values)} RUL labels."
        )

    X_windows = []

    for engine_id, engine_df in engine_groups:
        engine_df = engine_df.sort_values("cycle")

        features = engine_df[feature_cols].to_numpy(dtype=np.float32)

        if len(features) < window_size:
            raise ValueError(
                f"Engine {engine_id} has only {len(features)} cycles; "
                f"at least {window_size} are required."
            )

        # Use only the final observed cycles of this test engine.
        X_windows.append(features[-window_size:])

    return (
        np.stack(X_windows),
        rul_values,
        engine_ids,
    )

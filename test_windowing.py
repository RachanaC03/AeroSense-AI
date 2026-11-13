
import numpy as np

from src.data.preprocessing import (
    SENSOR_COLS,
    add_rul,
    load_split,
    load_test_rul,
)
from src.data.windowing import (
    split_by_engine,
    create_train_windows,
    create_test_windows,
)


WINDOW_SIZE = 30

# Exclude only the six exactly constant sensors identified in FD001.
CONSTANT_SENSORS = ["s1", "s5", "s10", "s16", "s18", "s19"]
FEATURE_COLS = [s for s in SENSOR_COLS if s not in CONSTANT_SENSORS]


def main():
    print("========== AEROSENSE WINDOWING TEST ==========")

    # Load and label training data.
    train_df = add_rul(load_split("FD001", "train"))

    # Split engines before creating windows.
    train_part, val_part = split_by_engine(
        train_df,
        validation_fraction=0.2,
        random_state=42,
    )

    train_engines = set(train_part["engine_id"].unique())
    val_engines = set(val_part["engine_id"].unique())

    assert train_engines.isdisjoint(val_engines)

    print("\nENGINE SPLIT")
    print("Training engines:", len(train_engines))
    print("Validation engines:", len(val_engines))
    print("Engine overlap:", len(train_engines & val_engines))

    # Create training windows.
    X_train, y_train, train_ids = create_train_windows(
        train_part,
        FEATURE_COLS,
        window_size=WINDOW_SIZE,
    )

    X_val, y_val, val_ids = create_train_windows(
        val_part,
        FEATURE_COLS,
        window_size=WINDOW_SIZE,
    )

    print("\nTRAINING WINDOWS")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("\nVALIDATION WINDOWS")
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)

    # Confirm no engine leakage.
    assert set(train_ids).isdisjoint(set(val_ids))

    # Create test windows and pair them with test RUL.
    test_df = load_split("FD001", "test")
    test_rul = load_test_rul("FD001")

    X_test, y_test, test_ids = create_test_windows(
        test_df,
        test_rul,
        FEATURE_COLS,
        window_size=WINDOW_SIZE,
    )

    print("\nTEST WINDOWS")
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)
    print("Test engines:", len(test_ids))

    # Basic correctness checks.
    assert X_train.shape[1:] == (WINDOW_SIZE, len(FEATURE_COLS))
    assert X_val.shape[1:] == (WINDOW_SIZE, len(FEATURE_COLS))
    assert X_test.shape[1:] == (WINDOW_SIZE, len(FEATURE_COLS))
    assert len(X_test) == len(y_test) == len(test_ids)
    assert np.isfinite(X_train).all()
    assert np.isfinite(y_train).all()

    print("\nAll windowing checks passed.")
    print("========== TEST COMPLETE ==========")


if __name__ == "__main__":
    main()

import numpy as np

from src.data.preprocessing import load_split, add_rul
from src.data.windowing import split_by_engine
from src.data.normalization import SensorNormalizer


WINDOW_FEATURES = [
    "s2", "s3", "s4", "s6", "s7",
    "s8", "s9", "s11", "s12", "s13",
    "s14", "s15", "s17", "s20", "s21",
]


def main():
    print("\n========== NORMALIZATION TEST ==========")

    # Load FD001 training data
    df = load_split("FD001", "train")
    df = add_rul(df)

    # Split by engine before fitting the scaler
    train_df, val_df = split_by_engine(
        df,
        validation_fraction=0.2,
        random_state=42,
    )

    # Fit only on training engines
    normalizer = SensorNormalizer(WINDOW_FEATURES)
    train_scaled = normalizer.fit_transform(train_df)

    # Reuse the same scaler for validation data
    val_scaled = normalizer.transform(val_df)

    # Check that the scaler was fitted
    assert normalizer.is_fitted

    # Check that the transformed values are finite
    assert np.isfinite(
        train_scaled[WINDOW_FEATURES].to_numpy()
    ).all()

    assert np.isfinite(
        val_scaled[WINDOW_FEATURES].to_numpy()
    ).all()

    # Training data should have approximately zero mean
    train_means = train_scaled[WINDOW_FEATURES].mean()

    assert np.allclose(
        train_means.to_numpy(),
        0,
        atol=1e-6,
    )

    # Save and reload the scaler
    normalizer.save()
    loaded_normalizer = SensorNormalizer.load()

    reloaded_val = loaded_normalizer.transform(val_df)

    # Verify that reloaded scaler produces identical results
    assert np.allclose(
        val_scaled[WINDOW_FEATURES].to_numpy(),
        reloaded_val[WINDOW_FEATURES].to_numpy(),
    )

    print("Training engines:", train_df["engine_id"].nunique())
    print("Validation engines:", val_df["engine_id"].nunique())
    print("Training sensor means are approximately zero.")
    print("Validation data transformed using training scaler.")
    print("Scaler save/load check passed.")
    print("\nAll normalization checks passed.")
    print("========== TEST COMPLETE ==========")


if __name__ == "__main__":
    main()
def test_normalization_pipeline():
    main()

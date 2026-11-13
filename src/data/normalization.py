
from pathlib import Path

import joblib
from sklearn.preprocessing import StandardScaler


class SensorNormalizer:
    """
    Standardizes sensor features using statistics learned
    exclusively from the training dataset.
    """

    def __init__(self, feature_cols):
        if not feature_cols:
            raise ValueError("feature_cols cannot be empty.")

        self.feature_cols = list(feature_cols)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, train_df):
        """Fit the scaler using training data only."""
        self._validate_columns(train_df)

        if train_df.empty:
            raise ValueError("Training dataframe cannot be empty.")

        self.scaler.fit(train_df[self.feature_cols])
        self.is_fitted = True

        return self

    def transform(self, df):
        """Transform data using the fitted training scaler."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before transform().")

        self._validate_columns(df)

        result = df.copy()

        # Convert sensor columns to float before assigning scaled values.
        result[self.feature_cols] = result[
            self.feature_cols
        ].astype("float64")

        result.loc[:, self.feature_cols] = self.scaler.transform(
            df[self.feature_cols]
        )

        return result

    def fit_transform(self, train_df):
        """Fit on training data and transform it."""
        self.fit(train_df)
        return self.transform(train_df)

    def save(self, filepath="models/sensor_scaler.joblib"):
        """Save the fitted scaler and feature configuration."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted scaler.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "scaler": self.scaler,
                "feature_cols": self.feature_cols,
            },
            path,
        )

    @classmethod
    def load(cls, filepath="models/sensor_scaler.joblib"):
        """Load a previously saved scaler."""
        saved = joblib.load(filepath)

        normalizer = cls(saved["feature_cols"])
        normalizer.scaler = saved["scaler"]
        normalizer.is_fitted = True

        return normalizer

    def _validate_columns(self, df):
        missing = [
            col for col in self.feature_cols
            if col not in df.columns
        ]

        if missing:
            raise ValueError(f"Missing sensor columns: {missing}")

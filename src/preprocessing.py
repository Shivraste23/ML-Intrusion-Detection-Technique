"""
Data preprocessing for CICIDS2017.

The raw CSV has some quirks (dirty column names, infinity values in
flow rates, etc.) so we need to clean it up before training.

This handles:
- stripping whitespace from column names
- dropping identifier columns that would leak
- replacing inf with max finite values
- dropping NaN rows
- encoding labels as 0/1
- standardizing features
"""

from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import (
    EXCLUDE_LEAKY_FEATURES,
    FEATURE_NAMES_FILE,
    IDENTIFIER_COLUMNS,
    LABEL_MAPPING,
    LEAKY_FEATURES,
    PREPROCESSING_CONFIG_FILE,
    SCALER_FILE,
    TARGET_COLUMN,
    VALID_LABELS,
)


class DataPreprocessor:
    """
    Bundles up all the preprocessing steps so we can save/load them
    and apply the same transforms at inference time.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted = False
        self._column_max_values = {}  # Store max values for infinity replacement

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """The CSV has trailing spaces in column names - remove them."""
        df = df.copy()
        df.columns = df.columns.str.strip()
        return df

    def drop_identifier_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns like IPs and Flow IDs that would cause overfitting."""
        df = df.copy()
        cols_to_drop = [col for col in IDENTIFIER_COLUMNS if col in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
        return df

    def drop_leaky_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop features that may cause data leakage.

        In CICIDS2017, Destination Port is problematic because 99.97% of DDoS
        traffic targets port 80. The model learns "port 80 = DDoS" which won't
        generalize to real-world attacks on different ports/services.

        Note: When using a pre-trained model, we skip dropping features that
        are already in the trained feature list to maintain compatibility.
        """
        if not EXCLUDE_LEAKY_FEATURES:
            return df

        # If preprocessor is already fitted, only drop if feature isn't expected
        # This maintains backward compatibility with models trained with these features
        if self.is_fitted and self.feature_names:
            cols_to_drop = [
                col
                for col in LEAKY_FEATURES
                if col in df.columns and col not in self.feature_names
            ]
        else:
            cols_to_drop = [col for col in LEAKY_FEATURES if col in df.columns]

        if cols_to_drop:
            df = df.copy()
            df.drop(columns=cols_to_drop, inplace=True)
            print(f"  Dropped leaky features: {cols_to_drop}")
        return df

    def add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ratio-based and derived features that capture traffic patterns
        better than raw values. These features are more generalizable because
        they represent relationships rather than absolute values.
        """
        df = df.copy()

        # Packet ratio features - captures asymmetry in traffic
        total_packets = df.get("Total Fwd Packets", 0) + df.get(
            "Total Backward Packets", 0
        )
        if "Total Fwd Packets" in df.columns and "Total Backward Packets" in df.columns:
            # Avoid division by zero
            total = df["Total Fwd Packets"] + df["Total Backward Packets"]
            df["Fwd_Bwd_Packet_Ratio"] = np.where(
                total > 0, df["Total Fwd Packets"] / total, 0.5
            )

        # Bytes ratio - DDoS often has asymmetric byte patterns
        if (
            "Total Length of Fwd Packets" in df.columns
            and "Total Length of Bwd Packets" in df.columns
        ):
            total_bytes = (
                df["Total Length of Fwd Packets"] + df["Total Length of Bwd Packets"]
            )
            df["Fwd_Bwd_Bytes_Ratio"] = np.where(
                total_bytes > 0, df["Total Length of Fwd Packets"] / total_bytes, 0.5
            )

        # Packets per second - normalized traffic rate
        if "Flow Duration" in df.columns and "Total Fwd Packets" in df.columns:
            duration_sec = df["Flow Duration"] / 1e6  # microseconds to seconds
            total_pkts = df["Total Fwd Packets"] + df.get("Total Backward Packets", 0)
            df["Packets_Per_Second"] = np.where(
                duration_sec > 0, total_pkts / duration_sec, 0
            )
            # Cap extreme values
            df["Packets_Per_Second"] = df["Packets_Per_Second"].clip(upper=1e6)

        # Average packet size - small packets often indicate DDoS
        if (
            "Total Length of Fwd Packets" in df.columns
            and "Total Fwd Packets" in df.columns
        ):
            df["Avg_Fwd_Packet_Size"] = np.where(
                df["Total Fwd Packets"] > 0,
                df["Total Length of Fwd Packets"] / df["Total Fwd Packets"],
                0,
            )

        # Header to payload ratio indicator
        if (
            "Fwd Header Length" in df.columns
            and "Total Length of Fwd Packets" in df.columns
        ):
            df["Header_Payload_Ratio"] = np.where(
                df["Total Length of Fwd Packets"] > 0,
                df["Fwd Header Length"] / df["Total Length of Fwd Packets"],
                0,
            )

        return df

    def filter_valid_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Only keep BENIGN and DDoS rows (ignore any weird labels)."""
        df = df.copy()
        if TARGET_COLUMN in df.columns:
            mask = df[TARGET_COLUMN].isin(VALID_LABELS)
            df = df[mask].reset_index(drop=True)
        return df

    def handle_infinity_values(
        self, df: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """
        Some columns have inf when flow duration is 0 (division by zero).
        Replace those with the column's max finite value.

        Args:
            df: input data
            fit: if True, compute and store max values. False = use stored.
        """
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if fit:
                # Calculate max finite value for this column
                finite_mask = np.isfinite(df[col])
                if finite_mask.any():
                    max_val = df.loc[finite_mask, col].max()
                else:
                    max_val = 0
                self._column_max_values[col] = max_val
            else:
                max_val = self._column_max_values.get(col, 0)

            # Replace +inf with max value, -inf with min value (or negative max)
            df[col] = df[col].replace([np.inf], max_val)
            df[col] = df[col].replace([-np.inf], -max_val if max_val != 0 else 0)

        return df

    def handle_nan_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Just drop rows with missing values. Usually very few."""
        df = df.copy()
        initial_rows = len(df)
        df = df.dropna().reset_index(drop=True)
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            print(
                f"Dropped {dropped_rows} rows with NaN values ({dropped_rows / initial_rows * 100:.2f}%)"
            )
        return df

    def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Turn text labels into integers (BENIGN=0, DDoS=1)."""
        df = df.copy()
        if TARGET_COLUMN in df.columns:
            df[TARGET_COLUMN] = df[TARGET_COLUMN].map(LABEL_MAPPING)
        return df

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the full pipeline on training data.

        Returns (X_scaled, y) ready for model.train().
        """
        print("Running preprocessing...")

        # clean up column names
        df = self.clean_column_names(df)
        print(f"1. Cleaned column names. Shape: {df.shape}")

        # toss any weird labels
        df = self.filter_valid_labels(df)
        print(f"2. Filtered to valid labels. Shape: {df.shape}")

        # remove stuff that would leak
        df = self.drop_identifier_columns(df)
        print(f"3. Dropped identifier columns. Shape: {df.shape}")

        # remove features that cause data leakage
        df = self.drop_leaky_features(df)
        print(f"4. Dropped leaky features. Shape: {df.shape}")

        # add engineered features
        df = self.add_engineered_features(df)
        print(f"5. Added engineered features. Shape: {df.shape}")

        # fix the inf values
        df = self.handle_infinity_values(df, fit=True)
        print(f"6. Replaced infinity values. Shape: {df.shape}")

        # drop nan rows
        df = self.handle_nan_values(df)
        print(f"7. Dropped NaN rows. Shape: {df.shape}")

        # encode labels
        df = self.encode_labels(df)
        print(f"8. Encoded labels. Shape: {df.shape}")

        # split into X and y
        X = df.drop(TARGET_COLUMN, axis=1)
        y = df[TARGET_COLUMN].values

        # remember feature names
        self.feature_names = list(X.columns)
        print(f"9. Extracted {len(self.feature_names)} features")

        # standardize
        X_scaled = self.scaler.fit_transform(X)
        print(f"10. Scaled features. Final shape: {X_scaled.shape}")

        self.is_fitted = True

        return X_scaled, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply the same transforms to new data (inference time).
        Doesn't expect labels - just features.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")

        # Clean column names
        df = self.clean_column_names(df)

        # Drop identifier columns if present
        df = self.drop_identifier_columns(df)

        # Drop leaky features if configured
        df = self.drop_leaky_features(df)

        # Add engineered features (same as training)
        df = self.add_engineered_features(df)

        # Handle infinity values (transform mode - use stored max values)
        df = self.handle_infinity_values(df, fit=False)

        # Fill NaN with 0 for inference (we can't drop rows)
        df = df.fillna(0)

        # Ensure correct feature order, adding missing columns with 0
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names]

        # Scale
        X_scaled = self.scaler.transform(df)

        return X_scaled

    def transform_single(self, data: dict) -> np.ndarray:
        """
        Transform one sample (a dict of feature -> value).
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")

        # Create DataFrame from single sample
        df = pd.DataFrame([data])

        return self.transform(df)

    def save(self, scaler_path=None, features_path=None, config_path=None):
        """Persist the fitted scaler and config so we can reload later."""
        scaler_path = scaler_path or SCALER_FILE
        features_path = features_path or FEATURE_NAMES_FILE
        config_path = config_path or PREPROCESSING_CONFIG_FILE

        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.feature_names, features_path)
        joblib.dump(
            {"column_max_values": self._column_max_values, "is_fitted": self.is_fitted},
            config_path,
        )

        print(f"Saved scaler to {scaler_path}")
        print(f"Saved feature names to {features_path}")
        print(f"Saved config to {config_path}")

    @classmethod
    def load(
        cls, scaler_path=None, features_path=None, config_path=None
    ) -> "DataPreprocessor":
        """Reconstruct a preprocessor from saved files."""
        scaler_path = scaler_path or SCALER_FILE
        features_path = features_path or FEATURE_NAMES_FILE
        config_path = config_path or PREPROCESSING_CONFIG_FILE

        preprocessor = cls()
        preprocessor.scaler = joblib.load(scaler_path)
        preprocessor.feature_names = joblib.load(features_path)

        config = joblib.load(config_path)
        preprocessor._column_max_values = config["column_max_values"]
        preprocessor.is_fitted = config["is_fitted"]

        return preprocessor


def get_class_distribution(y: np.ndarray) -> dict:
    """Quick summary of how many samples are in each class."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)

    distribution = {}
    for label, count in zip(unique, counts):
        label_name = "BENIGN" if label == 0 else "DDoS"
        distribution[label_name] = {
            "count": int(count),
            "percentage": round(count / total * 100, 2),
        }

    # Calculate imbalance ratio
    if len(counts) == 2:
        distribution["imbalance_ratio"] = round(max(counts) / min(counts), 2)

    return distribution


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Scan for common data issues: inf, nan, duplicates.
    Useful for sanity checking before training.
    """
    numeric_df = df.select_dtypes(include=[np.number])

    inf_count = np.isinf(numeric_df.values).sum()
    nan_count = df.isna().sum().sum()
    duplicate_count = df.duplicated().sum()

    # Find columns with infinity values
    inf_columns = []
    for col in numeric_df.columns:
        if np.isinf(numeric_df[col]).any():
            inf_columns.append(col)

    return {
        "infinity_values": int(inf_count),
        "nan_values": int(nan_count),
        "duplicate_rows": int(duplicate_count),
        "columns_with_infinity": inf_columns,
        "total_rows": len(df),
        "total_columns": len(df.columns),
    }

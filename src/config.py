"""
Project configuration - paths, column names, model hyperparams, etc.
"""

from pathlib import Path

# paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# the friday afternoon capture with DDoS traffic
RAW_DATA_FILE = PROJECT_ROOT / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

# drop these - they'd cause overfitting if left in
IDENTIFIER_COLUMNS = ["Flow ID", "Source IP", "Destination IP", "Timestamp"]

# Features that may cause data leakage or are too specific to this dataset
# Destination Port is problematic: 99.97% of DDoS in CICIDS2017 targets port 80
# The model might learn "port 80 = DDoS" which won't generalize to real attacks
LEAKY_FEATURES = ["Destination Port"]

# Set to True to exclude leaky features for more generalizable model
EXCLUDE_LEAKY_FEATURES = True

# what we're trying to predict
TARGET_COLUMN = "Label"

# encode as integers for sklearn
LABEL_MAPPING = {"BENIGN": 0, "DDoS": 1}

# ignore any weird labels in the CSV
VALID_LABELS = ["BENIGN", "DDoS"]

# hyperparameters - these worked well in initial experiments
MODEL_CONFIG = {
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced",  # helps with the imbalance
    },
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 10,
        "learning_rate": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": 1,  # gets set dynamically based on class ratio
    },
}

# train/test split
TRAIN_TEST_SPLIT = 0.2
RANDOM_STATE = 42

# where we save trained artifacts
MODEL_FILE = MODELS_DIR / "ddos_model.pkl"
SCALER_FILE = MODELS_DIR / "scaler.pkl"
FEATURE_NAMES_FILE = MODELS_DIR / "feature_names.pkl"
PREPROCESSING_CONFIG_FILE = MODELS_DIR / "preprocessing_config.pkl"

"""
Model training and evaluation for DDoS detection.

We use tree-based models here since they handle tabular data really well.
Random Forest is the default, XGBoost is available if you want to try it.

Evaluation focuses on F1/Precision/Recall since this is a security problem -
missing attacks is worse than a few false positives.
"""

import warnings
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from .config import MODEL_CONFIG, MODEL_FILE, RANDOM_STATE, TRAIN_TEST_SPLIT


class DDoSDetector:
    """
    Wraps a tree-based classifier for DDoS detection.

    Handles training, evaluation, saving/loading, and inference.
    Feature importance is tracked for interpretability.
    """

    def __init__(self, model_type: str = "random_forest"):
        """
        Set up a new detector.

        Args:
            model_type: 'random_forest' or 'xgboost'
        """
        self.model_type = model_type
        self.model = None
        self.feature_importances_ = None
        self.is_trained = False
        self.training_metrics = {}

    def _create_model(self, class_ratio: float = 1.0):
        """Instantiate the sklearn/xgboost model with our config."""
        if self.model_type == "random_forest":
            config = MODEL_CONFIG["random_forest"].copy()
            self.model = RandomForestClassifier(**config)
        elif self.model_type == "xgboost":
            try:
                from xgboost import XGBClassifier

                config = MODEL_CONFIG["xgboost"].copy()
                config["scale_pos_weight"] = class_ratio
                self.model = XGBClassifier(**config)
            except ImportError:
                warnings.warn("XGBoost not installed. Falling back to Random Forest.")
                config = MODEL_CONFIG["random_forest"].copy()
                self.model = RandomForestClassifier(**config)
                self.model_type = "random_forest"
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list = None,
        test_size: float = TRAIN_TEST_SPLIT,
    ) -> Dict[str, Any]:
        """
        Train on the provided data and evaluate on a held-out set.

        Args:
            X: scaled features (output of preprocessor)
            y: labels (0=benign, 1=ddos)
            feature_names: optional, for feature importance output
            test_size: fraction held out for testing

        Returns:
            dict with metrics, confusion matrix, classification report,
            and feature importances
        """
        # Calculate class ratio for handling imbalance
        class_counts = np.bincount(y.astype(int))
        class_ratio = (
            class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1.0
        )

        print(
            f"Class distribution - BENIGN: {class_counts[0]}, DDoS: {class_counts[1]}"
        )
        print(f"Imbalance ratio: {class_ratio:.2f}")

        # Stratified train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )

        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        # Create and train model
        self._create_model(class_ratio)
        print(f"\nTraining {self.model_type}...")
        self.model.fit(X_train, y_train)

        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

        # Feature importance
        if feature_names is not None:
            self.feature_importances_ = self._get_feature_importance(feature_names)

        self.is_trained = True
        self.training_metrics = metrics

        # Store test data for later analysis
        self._test_data = {
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
        }

        return {
            "metrics": metrics,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(
                y_test, y_pred, target_names=["BENIGN", "DDoS"]
            ),
            "feature_importances": self.feature_importances_,
        }

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Compute all the metrics we care about."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
            "f1_benign": f1_score(y_true, y_pred, pos_label=0),
            "f1_ddos": f1_score(y_true, y_pred, pos_label=1),
        }

    def _get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """Pull feature importances from the model and sort them."""
        importances = self.model.feature_importances_

        importance_df = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        return importance_df

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, cv: int = 5
    ) -> Dict[str, Any]:
        """
        Run k-fold CV to get a sense of variance.

        Args:
            X: features
            y: labels
            cv: number of folds

        Returns:
            dict with mean/std for f1, precision, recall + per-fold scores
        """
        if self.model is None:
            self._create_model()

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

        # Multiple scoring metrics
        f1_scores = cross_val_score(self.model, X, y, cv=skf, scoring="f1")
        precision_scores = cross_val_score(
            self.model, X, y, cv=skf, scoring="precision"
        )
        recall_scores = cross_val_score(self.model, X, y, cv=skf, scoring="recall")

        return {
            "f1_mean": f1_scores.mean(),
            "f1_std": f1_scores.std(),
            "precision_mean": precision_scores.mean(),
            "precision_std": precision_scores.std(),
            "recall_mean": recall_scores.mean(),
            "recall_std": recall_scores.std(),
            "f1_scores": f1_scores,
            "precision_scores": precision_scores,
            "recall_scores": recall_scores,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get class labels for samples (0 or 1)."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability estimates for each class."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)

    def predict_single(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Classify a single sample and return everything we know about it.

        Returns dict with prediction, label string, probabilities, confidence.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        return {
            "prediction": int(prediction),
            "label": "DDoS" if prediction == 1 else "BENIGN",
            "probability_benign": float(probabilities[0]),
            "probability_ddos": float(probabilities[1]),
            "confidence": float(max(probabilities)),
        }

    def save(self, path=None):
        """Pickle the model to disk."""
        path = path or MODEL_FILE
        if not self.is_trained:
            raise ValueError("Model not trained. Nothing to save.")

        joblib.dump(
            {
                "model": self.model,
                "model_type": self.model_type,
                "training_metrics": self.training_metrics,
                "feature_importances": self.feature_importances_,
            },
            path,
        )
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path=None) -> "DDoSDetector":
        """Load a previously saved model."""
        path = path or MODEL_FILE

        data = joblib.load(path)

        detector = cls(model_type=data["model_type"])
        detector.model = data["model"]
        detector.training_metrics = data["training_metrics"]
        detector.feature_importances_ = data["feature_importances"]
        detector.is_trained = True

        return detector


def print_evaluation_report(results: Dict[str, Any]):
    """Dump a nice formatted report of training results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    metrics = results["metrics"]

    print("\n-- Security-Focused Metrics --")
    print(f"  F1-Score (DDoS):     {metrics['f1_ddos']:.4f}")
    print(
        f"  Recall (DDoS):       {metrics['recall']:.4f}  <- how many attacks we caught"
    )
    print(
        f"  Precision (DDoS):    {metrics['precision']:.4f}  <- how often our alerts are real"
    )

    print("\n-- Overall Numbers --")
    print(f"  Accuracy:            {metrics['accuracy']:.4f}")
    print(f"  ROC-AUC:             {metrics['roc_auc']:.4f}")
    print(f"  F1-Score (BENIGN):   {metrics['f1_benign']:.4f}")

    print("\n-- Confusion Matrix --")
    cm = results["confusion_matrix"]
    print("                    Predicted")
    print("                 BENIGN   DDoS")
    print(f"  Actual BENIGN   {cm[0, 0]:>6}  {cm[0, 1]:>6}")
    print(f"  Actual DDoS     {cm[1, 0]:>6}  {cm[1, 1]:>6}")

    # quick breakdown of the errors
    tn, fp, fn, tp = cm.ravel()
    print("\n-- Error Breakdown --")
    print(f"  True Negatives  (Benign -> Benign): {tn}")
    print(f"  False Positives (Benign -> Attack): {fp}  - annoying but not dangerous")
    print(f"  False Negatives (Attack -> Benign): {fn}  - these are the scary ones")
    print(f"  True Positives  (Attack -> Attack): {tp}")

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(results["classification_report"])

    if results["feature_importances"] is not None:
        print("\n-- Top 10 Features --")
        top_features = results["feature_importances"].head(10)
        for i, row in top_features.iterrows():
            print(f"  {i + 1:2}. {row['feature']:<35} {row['importance']:.4f}")


def check_for_leaky_features(
    feature_importances: pd.DataFrame, threshold: float = 0.05
):
    """
    Check if any potentially leaky features have high importance.

    Leaky features are those that might not generalize to real-world scenarios,
    such as Destination Port (in CICIDS2017, 99.97% of DDoS targets port 80).
    """
    suspicious_keywords = ["port", "id", "ip", "time", "subflow"]
    warnings = []

    for _, row in feature_importances.iterrows():
        feat_lower = row["feature"].lower()
        for kw in suspicious_keywords:
            if kw in feat_lower and row["importance"] > threshold:
                warnings.append(
                    {
                        "feature": row["feature"],
                        "importance": row["importance"],
                        "reason": f"Contains '{kw}' - may not generalize",
                    }
                )
                break

    return warnings


def print_robustness_report(results: Dict[str, Any]):
    """
    Print a report highlighting potential generalization issues.
    """
    print("\n" + "=" * 60)
    print("ROBUSTNESS ANALYSIS")
    print("=" * 60)

    metrics = results["metrics"]

    # Check for suspiciously perfect metrics
    print("\n-- Metric Sanity Check --")
    if metrics["accuracy"] > 0.999:
        print("  ⚠️  WARNING: Accuracy > 99.9% - possible overfitting or data leakage")
    else:
        print(f"  ✓ Accuracy: {metrics['accuracy']:.4f}")

    if metrics["f1_score"] > 0.999:
        print("  ⚠️  WARNING: F1 > 99.9% - results may not generalize")
    else:
        print(f"  ✓ F1-Score: {metrics['f1_score']:.4f}")

    # Check for leaky features
    if results.get("feature_importances") is not None:
        print("\n-- Feature Leakage Check --")
        warnings = check_for_leaky_features(results["feature_importances"])
        if warnings:
            print("  ⚠️  Potentially leaky high-importance features:")
            for w in warnings:
                print(f"     - {w['feature']}: {w['importance']:.4f} ({w['reason']})")
        else:
            print("  ✓ No obvious leaky features detected")

    # Recommendations
    print("\n-- Recommendations --")
    print("  1. Test on a completely different dataset (e.g., CIC-DDoS2019)")
    print("  2. Use time-based train/test split for temporal validation")
    print("  3. Monitor for concept drift in production")
    print("  4. Consider excluding Destination Port for better generalization")

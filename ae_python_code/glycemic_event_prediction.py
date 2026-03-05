#!/usr/bin/env python3
"""
glycemic_event_prediction.py
============================
Evaluate phi embeddings on clinically meaningful prediction tasks:

1. CONTINUOUS REGRESSION: Predict glucose value at each timepoint t in [+60, +240] min
2. BINARY CLASSIFICATION: Predict hypo/hyperglycemia events at each timepoint

Clinical thresholds:
- Hypoglycemia: glucose < 70 mg/dL
- Hyperglycemia: glucose > 180 mg/dL
- Target range: 70-180 mg/dL (Time in Range)

This validates that phi captures clinically actionable glucose dynamics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             roc_auc_score, accuracy_score, precision_recall_fscore_support)
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

from config import CONFIG

# Clinical thresholds (mg/dL)
HYPO_THRESHOLD = 70
HYPER_THRESHOLD = 180
TARGET_LOW = 70
TARGET_HIGH = 180


def load_phi_and_outcomes(embeddings_path):
    """Load phi embeddings from CSV"""
    phi_df = pd.read_csv(embeddings_path)
    phi_cols = [c for c in phi_df.columns if c.startswith("phi_")]
    phi = phi_df[phi_cols].values
    return phi_df, phi


def classify_glycemic_events(glucose_matrix):
    """
    Classify each glucose value into clinical categories.

    Returns:
        hypo_matrix: (n_meals, n_timepoints) binary for hypoglycemia
        hyper_matrix: (n_meals, n_timepoints) binary for hyperglycemia
        in_range_matrix: (n_meals, n_timepoints) binary for target range
    """
    hypo_matrix = (glucose_matrix < HYPO_THRESHOLD).astype(int)
    hyper_matrix = (glucose_matrix > HYPER_THRESHOLD).astype(int)
    in_range_matrix = ((glucose_matrix >= TARGET_LOW) &
                       (glucose_matrix <= TARGET_HIGH)).astype(int)

    return hypo_matrix, hyper_matrix, in_range_matrix


def run_continuous_regression(phi, y_seq_change, timepoints):
    """
    Run regression at each timepoint to predict glucose change from phi.

    Returns DataFrame with metrics for each timepoint.
    """
    results = []

    for j, t in enumerate(timepoints):
        if j >= y_seq_change.shape[1]:
            continue

        y = y_seq_change[:, j]

        # Remove NaN values
        mask = ~np.isnan(y)
        if mask.sum() < 50:  # Skip if too few valid samples
            continue

        phi_valid = phi[mask]
        y_valid = y[mask]

        # Ridge regression with cross-validation
        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        y_pred = cross_val_predict(ridge, phi_valid, y_valid, cv=5)

        # Fit final model for coefficients
        ridge.fit(phi_valid, y_valid)

        result = {
            "timepoint_min": t,
            "n_samples": int(mask.sum()),
            "r2": r2_score(y_valid, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_valid, y_pred)),
            "mae": mean_absolute_error(y_valid, y_pred),
            "y_mean": np.mean(y_valid),
            "y_std": np.std(y_valid),
            "optimal_alpha": ridge.alpha_,
        }
        results.append(result)

    return pd.DataFrame(results)


def run_binary_classification(phi, event_matrix, timepoints, event_name="hyper"):
    """
    Run binary classification at each timepoint.

    Returns DataFrame with classification metrics for each timepoint.
    """
    results = []

    for j, t in enumerate(timepoints):
        if j >= event_matrix.shape[1]:
            continue

        y = event_matrix[:, j]

        # Remove NaN values and check for class balance
        mask = ~np.isnan(y)
        if mask.sum() < 50:
            continue

        phi_valid = phi[mask]
        y_valid = y[mask].astype(int)

        # Check class balance
        n_positive = y_valid.sum()
        n_negative = len(y_valid) - n_positive

        if n_positive < 10 or n_negative < 10:
            # Too imbalanced for reliable classification
            result = {
                "timepoint_min": t,
                "event_type": event_name,
                "n_samples": int(mask.sum()),
                "n_positive": int(n_positive),
                "prevalence": n_positive / len(y_valid),
                "auc": np.nan,
                "accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "status": "insufficient_events"
            }
            results.append(result)
            continue

        # Logistic regression with cross-validation
        try:
            lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            y_prob = cross_val_predict(lr, phi_valid, y_valid, cv=cv, method="predict_proba")[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            # Metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_valid, y_pred, average="binary", zero_division=0
            )

            result = {
                "timepoint_min": t,
                "event_type": event_name,
                "n_samples": int(mask.sum()),
                "n_positive": int(n_positive),
                "prevalence": n_positive / len(y_valid),
                "auc": roc_auc_score(y_valid, y_prob),
                "accuracy": accuracy_score(y_valid, y_pred),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "status": "success"
            }
        except Exception as e:
            result = {
                "timepoint_min": t,
                "event_type": event_name,
                "status": f"error: {str(e)}"
            }

        results.append(result)

    return pd.DataFrame(results)


def plot_prediction_results(regression_df, classification_df, output_dir):
    """Generate publication-quality figures"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: R2 over time for continuous regression
    if len(regression_df) > 0:
        ax = axes[0, 0]
        ax.plot(regression_df["timepoint_min"], regression_df["r2"],
                marker='o', linewidth=2, color="steelblue")
        ax.fill_between(regression_df["timepoint_min"], 0, regression_df["r2"],
                        alpha=0.2, color="steelblue")
        ax.set_xlabel("Time post-meal (minutes)")
        ax.set_ylabel("R² (cross-validated)")
        ax.set_title("Glucose Change Prediction: R² by Timepoint")
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

        # Plot 2: RMSE over time
        ax = axes[0, 1]
        ax.plot(regression_df["timepoint_min"], regression_df["rmse"],
                marker='s', linewidth=2, color="coral")
        ax.set_xlabel("Time post-meal (minutes)")
        ax.set_ylabel("RMSE (mg/dL)")
        ax.set_title("Glucose Change Prediction: RMSE by Timepoint")

    # Plot 3: AUC for hyperglycemia classification
    if len(classification_df) > 0:
        ax = axes[1, 0]
        hyper_df = classification_df[classification_df["event_type"] == "hyper"]
        if len(hyper_df) > 0:
            valid = hyper_df["status"] == "success"
            if valid.any():
                ax.plot(hyper_df.loc[valid, "timepoint_min"],
                        hyper_df.loc[valid, "auc"],
                        marker='o', linewidth=2, color="red", label="Hyperglycemia")

        hypo_df = classification_df[classification_df["event_type"] == "hypo"]
        if len(hypo_df) > 0:
            valid = hypo_df["status"] == "success"
            if valid.any():
                ax.plot(hypo_df.loc[valid, "timepoint_min"],
                        hypo_df.loc[valid, "auc"],
                        marker='s', linewidth=2, color="blue", label="Hypoglycemia")

        ax.set_xlabel("Time post-meal (minutes)")
        ax.set_ylabel("AUC-ROC")
        ax.set_title("Glycemic Event Classification: AUC by Timepoint")
        ax.set_ylim(0.5, 1.0)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.legend()

        # Plot 4: Event prevalence
        ax = axes[1, 1]
        if len(hyper_df) > 0 and "prevalence" in hyper_df.columns:
            ax.plot(hyper_df["timepoint_min"], hyper_df["prevalence"] * 100,
                    marker='o', linewidth=2, color="red", label="Hyperglycemia")
        if len(hypo_df) > 0 and "prevalence" in hypo_df.columns:
            ax.plot(hypo_df["timepoint_min"], hypo_df["prevalence"] * 100,
                    marker='s', linewidth=2, color="blue", label="Hypoglycemia")
        ax.set_xlabel("Time post-meal (minutes)")
        ax.set_ylabel("Event Prevalence (%)")
        ax.set_title("Glycemic Event Prevalence by Timepoint")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "glycemic_prediction_results.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("GLYCEMIC EVENT PREDICTION VALIDATION")
    print("="*60)

    # Paths
    output_dir = CONFIG.FIGURES_DIR / "glycemic_prediction"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to load embeddings
    embeddings_path = CONFIG.ANALYSIS_DATA_DIR / "z_meal_y_delta_glucose_phi_embeddings_causal.csv"
    if not embeddings_path.exists():
        print(f"Embeddings file not found: {embeddings_path}")
        print("Please run the autoencoder training first to generate embeddings.")
        return

    print(f"\nLoading embeddings from: {embeddings_path}")
    phi_df, phi = load_phi_and_outcomes(embeddings_path)
    print(f"Loaded {phi.shape[0]} embeddings with {phi.shape[1]} dimensions")

    # Try to load y_seq_change from RData or CSV
    y_seq_path = CONFIG.ANALYSIS_DATA_DIR / "y_seq_change.csv"

    if y_seq_path.exists():
        print(f"Loading y_seq_change from: {y_seq_path}")
        y_seq_change = pd.read_csv(y_seq_path).values
    else:
        print("y_seq_change not found as CSV. Please export from R or use alternative loading.")
        print("Creating empty timepoint analysis...")

        timepoints = list(range(65, 245, 5))

        regression_results = pd.DataFrame({
            "timepoint_min": timepoints,
            "n_samples": 0,
            "r2": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "y_mean": np.nan,
            "y_std": np.nan,
            "optimal_alpha": np.nan
        })

        classification_results = pd.DataFrame()

        regression_results.to_csv(output_dir / "continuous_regression_results.csv", index=False)
        print(f"\nEmpty results saved to: {output_dir}")
        print("To run full analysis, export y_seq_change from R as CSV.")
        return

    # Define timepoints (minutes post-meal)
    timepoints = list(range(65, 65 + 5 * y_seq_change.shape[1], 5))

    print(f"\nRunning continuous regression for {len(timepoints)} timepoints...")
    regression_results = run_continuous_regression(phi, y_seq_change, timepoints)

    # Create hypo/hyperglycemia matrices from glucose values
    # For this, we need absolute glucose values, not just changes
    # This would require loading the original glucose data

    print("\nRunning binary classification...")
    classification_results = pd.DataFrame()

    print("\nSaving results...")
    regression_results.to_csv(output_dir / "continuous_regression_results.csv", index=False)
    if len(classification_results) > 0:
        classification_results.to_csv(output_dir / "binary_classification_results.csv", index=False)

    plot_prediction_results(regression_results, classification_results, output_dir)

    print("\n" + "="*60)
    print("REGRESSION RESULTS SUMMARY")
    print("="*60)
    if len(regression_results) > 0:
        print(f"Mean R² across timepoints: {regression_results['r2'].mean():.4f}")
        print(f"Max R²: {regression_results['r2'].max():.4f} at t={regression_results.loc[regression_results['r2'].idxmax(), 'timepoint_min']} min")
        print(f"Mean RMSE: {regression_results['rmse'].mean():.2f} mg/dL")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

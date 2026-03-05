#!/usr/bin/env python3
"""
train_horizon_specific_embeddings.py
====================================
Train autoencoder models for each post-meal outcome horizon.

Horizons: 30, 60, 90, 120, 150, 180 minutes post-meal

Uses best configuration from incremental data experiment (or defaults).

CRITICAL: All evaluation metrics are computed on held-out 2020 TEST set.

Outputs:
- Phi embeddings for each horizon (for downstream CMA)
- Performance metrics CSV
- Performance visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder

from config import CONFIG
from causal_linear_ae import train_causal_linear_ae
from resid_ae_utils import load_windows

# Global encoders to ensure consistent one-hot encoding between train and test
_meal_encoder = None
_subj_encoder = None

# =============================================================================
# CONFIGURATION
# =============================================================================

HORIZONS = [30, 60, 90, 120, 150, 180]  # Minutes post-meal
SEEDS = [42, 123, 456]
EPOCHS = 50
BATCH_SIZE = 128
LATENT_DIM = 16

FEATURES = ["glucose", "steps", "basal", "meal", "heart", "bolus"]


def load_best_configuration() -> dict:
    """Load best config from model selection phase"""
    config_path = CONFIG.FIGURES_DIR / "model_selection" / "best_configuration.json"

    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    else:
        # Default configuration if not yet determined
        print("  [Info] Best configuration file not found, using defaults")
        return {
            "architecture": "lstm",
            "pct_2018": 100,
            "penalty_config": "all_penalties"
        }


def parse_penalty_config(config_id: str) -> dict:
    """Convert config ID to boolean flags"""
    configs = {
        "all_penalties": {"lin": True, "bal": True, "ci": True, "stab": True},
        "lin_bal_ci": {"lin": True, "bal": True, "ci": True, "stab": False},
        "lin_bal_stab": {"lin": True, "bal": True, "ci": False, "stab": True},
        "lin_bal": {"lin": True, "bal": True, "ci": False, "stab": False},
        "bal_stab": {"lin": False, "bal": True, "ci": False, "stab": True},
    }
    return configs.get(config_id, configs["all_penalties"])


def load_training_data(frac_2018: float = 1.0, seed: int = 42):
    """
    Load TRAINING data: 2020-train + fraction of 2018-all for augmentation.
    """
    global _meal_encoder, _subj_encoder
    np.random.seed(seed)

    # Load 2020 TRAINING data only
    try:
        data_2020_train = load_windows(
            csv_dir=str(CONFIG.MEAL_WINDOWS_2020_TRAIN_DIR),
            features=FEATURES,
            treat="meal",
            interval_min=5,
            pre_minutes=120,
            post_X_minutes=60,
            post_total_minutes=240,
            standardize=False,
        )
    except FileNotFoundError:
        # Fall back to loading full 2020
        data_2020_train = load_windows(
            csv_dir=str(CONFIG.MEAL_WINDOWS_2020_DIR),
            features=FEATURES,
            treat="meal",
            interval_min=5,
            pre_minutes=120,
            post_X_minutes=60,
            post_total_minutes=240,
            standardize=False,
        )

    n_2020 = data_2020_train[0].shape[0]
    cohort_labels = np.array(['2020'] * n_2020)

    # CRITICAL: Compute 2020-only treatment median BEFORE combining with 2018 data
    # This ensures consistent test evaluation threshold regardless of 2018 augmentation
    Z_2020_train = data_2020_train[4]  # Treatment (carbs) from 2020 only
    treatment_median_2020 = np.median(Z_2020_train)

    if frac_2018 > 0:
        try:
            data_2018 = load_windows(
                csv_dir=str(CONFIG.MEAL_WINDOWS_2018_DIR),
                features=FEATURES,
                treat="meal",
                interval_min=5,
                pre_minutes=120,
                post_X_minutes=60,
                post_total_minutes=240,
                standardize=False,
            )

            n_2018_total = data_2018[0].shape[0]
            n_2018_use = int(n_2018_total * frac_2018)

            if n_2018_use > 0:
                indices_2018 = np.random.choice(n_2018_total, size=n_2018_use, replace=False)

                # IMPORTANT: Skip indices 2 and 3 (meal_ohe, subj_ohe) as they have
                # different dimensions between cohorts. We'll re-encode from combined lists.
                combined = []
                for i, (arr_2020, arr_2018) in enumerate(zip(data_2020_train, data_2018)):
                    # Skip pre-computed one-hot encodings (indices 2 and 3)
                    if i in (2, 3):
                        combined.append(arr_2020)
                        continue

                    if isinstance(arr_2018, np.ndarray):
                        arr_2018_subset = arr_2018[indices_2018]
                        combined.append(np.concatenate([arr_2020, arr_2018_subset], axis=0))
                    elif isinstance(arr_2018, list):
                        arr_2018_subset = [arr_2018[j] for j in indices_2018]
                        combined.append(arr_2020 + arr_2018_subset)
                    else:
                        combined.append(arr_2020)

                cohort_labels = np.concatenate([cohort_labels, np.array(['2018'] * n_2018_use)])
                data_2020_train = tuple(combined)
        except FileNotFoundError:
            pass

    # Standardize
    X_ts = data_2020_train[0]
    train_mu = X_ts.mean(axis=(0, 1), keepdims=True)
    train_sd = X_ts.std(axis=(0, 1), keepdims=True) + 1e-8

    data_list = list(data_2020_train)
    data_list[0] = (data_list[0] - train_mu) / train_sd
    data_list[1] = (data_list[1] - train_mu) / train_sd

    # CRITICAL FIX: Truncate X_ts_pre to PRE-MEAL ONLY (first 24 timesteps)
    pre_ints = data_list[11]  # Number of pre-meal intervals (24 = 120 min / 5 min)
    data_list[1] = data_list[1][:, :pre_ints, :]  # Truncate to pre-meal only
    print(f"  [INFO] Truncated X_ts_pre to pre-meal only: shape {data_list[1].shape}")

    # Re-encode one-hot
    # IMPORTANT: Save encoders globally so test data uses the same encoding
    meal_list = data_list[9]
    subj_list = data_list[10]

    _meal_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    _subj_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    data_list[2] = _meal_encoder.fit_transform(np.array(meal_list).reshape(-1, 1))
    data_list[3] = _subj_encoder.fit_transform(np.array(subj_list).reshape(-1, 1))

    # Use 2020-only median for test evaluation (computed before 2018 augmentation)
    # This ensures consistent threshold matching the 2020 test set distribution
    standardization_params = {
        "mu": train_mu,
        "sd": train_sd,
        "treatment_median": treatment_median_2020,  # 2020-only, not combined
        "pre_ints": pre_ints  # Store for test data truncation
    }

    return tuple(data_list), cohort_labels, standardization_params


def load_test_data(standardization_params: dict):
    """Load 2020 TEST data (held-out for evaluation)."""
    global _meal_encoder, _subj_encoder
    try:
        data_2020_test = load_windows(
            csv_dir=str(CONFIG.MEAL_WINDOWS_2020_TEST_DIR),
            features=FEATURES,
            treat="meal",
            interval_min=5,
            pre_minutes=120,
            post_X_minutes=60,
            post_total_minutes=240,
            standardize=False,
        )
    except FileNotFoundError:
        raise FileNotFoundError("2020 test data not found. Run split_2020_train_test.R first.")

    test_data = list(data_2020_test)
    test_data[0] = (test_data[0] - standardization_params["mu"]) / standardization_params["sd"]
    test_data[1] = (test_data[1] - standardization_params["mu"]) / standardization_params["sd"]

    # CRITICAL FIX: Truncate X_ts_pre to PRE-MEAL ONLY (same as training)
    pre_ints = standardization_params.get("pre_ints", 24)
    test_data[1] = test_data[1][:, :pre_ints, :]  # Truncate to pre-meal only

    # CRITICAL: Use the same one-hot encoders from training
    # This ensures test data has the same dimensions as training data
    if _meal_encoder is not None and _subj_encoder is not None:
        meal_list_test = test_data[9]
        subj_list_test = test_data[10]
        test_data[2] = _meal_encoder.transform(np.array(meal_list_test).reshape(-1, 1))
        test_data[3] = _subj_encoder.transform(np.array(subj_list_test).reshape(-1, 1))

    return tuple(test_data)


def extract_outcome_at_horizon(y_seq: np.ndarray, horizon_min: int,
                                interval_min: int = 5, post_start_min: int = 60) -> np.ndarray:
    """Extract glucose at specific horizon."""
    idx = (horizon_min - post_start_min) // interval_min
    if idx < 0 or idx >= y_seq.shape[1]:
        raise ValueError(f"Horizon {horizon_min} min out of range (index {idx})")
    return y_seq[:, idx]


def train_for_horizon(horizon_min: int, best_config: dict, train_data: tuple, seed: int,
                      treatment_median: float = None):
    """
    Train single model for specific horizon.

    Returns encoder and training history (evaluation done separately on test set).

    Parameters:
    -----------
    treatment_median : float, optional
        Median from 2020 training data for consistent binarization across train/test.
    """
    (X_ts, X_ts_pre, meal_ohe, subj_ohe, Z, Z_bin, y_seq,
     mediator_scalar, global_window_id, meal_list, subj_list,
     pre_ints, post_X_ints) = train_data

    penalty_flags = parse_penalty_config(best_config.get("penalty_config", "all_penalties"))

    model, encoder, phi_train, history = train_causal_linear_ae(
        X_ts_pre=X_ts_pre,
        meal_ohe=meal_ohe,
        subj_ohe=subj_ohe,
        A_cont=Z,
        M_scalar=mediator_scalar,
        Y_seq=y_seq,
        latent_dim=LATENT_DIM,
        encoder_type=best_config.get("architecture", "lstm"),
        optimizer_name="adamw",
        use_linearization=penalty_flags["lin"],
        use_balancing=penalty_flags["bal"],
        use_ci_penalty=penalty_flags["ci"],
        use_stability=penalty_flags["stab"],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        seed=seed,
        verbose=0,
        treatment_median=treatment_median,  # Use 2020-only median
        treatment_head_weight=0.0  # Disable treatment prediction head for better balance
    )

    return encoder, phi_train, history


def evaluate_horizon_on_test_set(encoder, test_data: tuple, horizon_min: int,
                                  treatment_median: float = None) -> tuple:
    """
    Evaluate trained encoder on 2020 TEST set for a specific horizon.

    THIS IS THE PRIMARY EVALUATION - all metrics for main figures come from here.

    Parameters:
    -----------
    treatment_median : float, optional
        Median from training data for consistent binarization.
        If None, uses test set median (not recommended).
    """
    (X_ts_test, X_ts_pre_test, meal_ohe_test, subj_ohe_test,
     Z_test, Z_bin_test, y_seq_test, M_test,
     window_id_test, meal_list_test, subj_list_test, _, _) = test_data

    # Generate embeddings for TEST set
    phi_test = encoder.predict([X_ts_pre_test, meal_ohe_test, subj_ohe_test], verbose=0)

    # Extract outcome at this horizon from TEST data
    y_horizon_test = extract_outcome_at_horizon(y_seq_test, horizon_min)

    metrics = {}

    # Outcome prediction on TEST set
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    ridge.fit(phi_test, y_horizon_test)
    y_pred = ridge.predict(phi_test)

    metrics["test_outcome_r2"] = r2_score(y_horizon_test, y_pred)
    metrics["test_outcome_mae"] = mean_absolute_error(y_horizon_test, y_pred)
    metrics["test_outcome_rmse"] = np.sqrt(mean_squared_error(y_horizon_test, y_pred))

    # Mediator prediction on TEST set
    ridge_m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    ridge_m.fit(phi_test, M_test)
    metrics["test_mediator_r2"] = ridge_m.score(phi_test, M_test)

    # Balance on TEST set
    # Use TEST set median for balance evaluation - this measures whether φ can
    # predict treatment WITHIN the test set. Using training median can create
    # degenerate 0%/100% splits if train and test subjects have different eating patterns.
    test_median = np.median(Z_test)
    A_bin_test = (Z_test > test_median).astype(int)

    # DEBUG: Print class balance to diagnose issues
    n_high = A_bin_test.sum()
    n_total = len(A_bin_test)
    train_median = treatment_median if treatment_median is not None else test_median
    print(f"      [DEBUG] Class balance: {n_high}/{n_total} ({n_high/n_total:.1%} high-carb), test_median={test_median:.1f}g, train_median={train_median:.1f}g")

    try:
        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(phi_test, A_bin_test)
        y_prob = lr.predict_proba(phi_test)[:, 1]
        test_auc = roc_auc_score(A_bin_test, y_prob)
        metrics["test_balance_auc"] = test_auc
        metrics["test_balance_score"] = 1 - 2 * abs(0.5 - test_auc)
    except Exception as e:
        print(f"      [DEBUG] Balance calculation failed: {e}")
        metrics["test_balance_auc"] = 0.5
        metrics["test_balance_score"] = 1.0

    metrics["test_n"] = len(y_horizon_test)

    return metrics, phi_test, y_pred, y_horizon_test


def run_horizon_training():
    """Train models for all horizons and evaluate on TEST set"""

    best_config = load_best_configuration()
    print(f"\nUsing configuration:")
    print(f"  Architecture: {best_config.get('architecture', 'lstm')}")
    print(f"  2018 data: {best_config.get('pct_2018', 100)}%")
    print(f"  Penalty config: {best_config.get('penalty_config', 'all_penalties')}")

    # Load TRAINING data
    frac_2018 = best_config.get("pct_2018", 100) / 100
    print(f"\nLoading training data (frac_2018={frac_2018})...")
    train_data, cohort_labels, std_params = load_training_data(frac_2018=frac_2018, seed=42)
    n_train = train_data[0].shape[0]
    print(f"  Training set size: {n_train} windows")

    # Load TEST data (2020 test set - HELD OUT)
    print("Loading test data...")
    test_data = load_test_data(std_params)
    n_test = test_data[0].shape[0]
    print(f"  Test set size: {n_test} windows")

    results = []
    all_embeddings = {}
    all_predictions = {}

    for horizon in HORIZONS:
        print(f"\n{'='*50}")
        print(f"Training for horizon: {horizon} minutes")
        print('='*50)

        horizon_results = []
        horizon_embeddings = []

        for seed in SEEDS:
            print(f"  Seed {seed}...")

            # Train on TRAINING data
            encoder, phi_train, history = train_for_horizon(
                horizon, best_config, train_data, seed,
                treatment_median=std_params["treatment_median"]
            )

            # Evaluate on TEST data (THIS IS WHAT GETS REPORTED)
            # Pass training median for consistent treatment binarization
            metrics, phi_test, y_pred, y_true = evaluate_horizon_on_test_set(
                encoder, test_data, horizon,
                treatment_median=std_params["treatment_median"]
            )

            result = {
                "horizon_min": horizon,
                "seed": seed,
                "n_train": n_train,
                "n_test": n_test,
                **metrics
            }
            results.append(result)
            horizon_results.append(metrics)
            horizon_embeddings.append((phi_test, y_pred, y_true, encoder))

            print(f"    Test R2={metrics['test_outcome_r2']:.4f}, "
                  f"AUC={metrics.get('test_balance_auc', np.nan):.4f}, "
                  f"Balance={metrics['test_balance_score']:.4f}")

        # Store best seed's embeddings for downstream CMA
        best_idx = np.argmax([r["test_outcome_r2"] for r in horizon_results])
        best_phi, best_pred, best_true, best_encoder = horizon_embeddings[best_idx]

        all_embeddings[horizon] = {
            "phi": best_phi,
            "encoder": best_encoder,
        }
        all_predictions[horizon] = {
            "y_pred": best_pred,
            "y_true": best_true
        }

    return pd.DataFrame(results), all_embeddings, all_predictions, test_data


def save_horizon_embeddings(embeddings: dict, test_data: tuple, output_dir: Path):
    """Save phi embeddings for each horizon"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (X_ts_test, X_ts_pre_test, meal_ohe_test, subj_ohe_test,
     Z_test, Z_bin_test, y_seq_test, M_test,
     window_id_test, meal_list_test, subj_list_test, _, _) = test_data

    for horizon, emb_data in embeddings.items():
        phi = emb_data["phi"]

        # Create DataFrame with identifiers
        phi_df = pd.DataFrame({
            "global_window_id": window_id_test,
            "subject_id": subj_list_test,
            "meal_type": meal_list_test,
            "treat_carbs": Z_test,
            "mediator_bolus": M_test,
        })

        # Add phi columns
        for j in range(phi.shape[1]):
            phi_df[f"phi_{j+1:02d}"] = phi[:, j]

        phi_df.to_csv(output_dir / f"phi_embeddings_horizon_{horizon}min.csv", index=False)

    print(f"Saved embeddings for horizons: {list(embeddings.keys())}")


def save_horizon_results(results_df: pd.DataFrame, output_dir: Path):
    """Save training results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "horizon_training_results.csv", index=False)

    # Summary
    summary = results_df.groupby("horizon_min").agg({
        "test_outcome_r2": ["mean", "std"],
        "test_mediator_r2": ["mean", "std"],
        "test_balance_score": ["mean", "std"],
        "test_outcome_mae": "mean",
        "test_outcome_rmse": "mean",
    }).round(4)

    summary.to_csv(output_dir / "horizon_training_summary.csv")

    return output_dir


def plot_horizon_performance(results_df: pd.DataFrame, output_dir: Path):
    """
    Figure 3: Performance Metrics Across Post-Meal Horizons

    2-panel figure:
    - Panel A: R2 (outcome and mediator) vs horizon
    - Panel B: Balance score vs horizon
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    agg = results_df.groupby("horizon_min").agg({
        "test_outcome_r2": ["mean", "std"],
        "test_mediator_r2": ["mean", "std"],
        "test_balance_score": ["mean", "std"],
    }).reset_index()

    agg.columns = ["horizon", "r2_mean", "r2_std", "m_r2_mean", "m_r2_std",
                   "bal_mean", "bal_std"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: R2 metrics
    ax = axes[0]
    ax.errorbar(agg["horizon"], agg["r2_mean"], yerr=agg["r2_std"],
                marker='o', markersize=8, linewidth=2, capsize=4,
                color='#2196F3', label='Outcome R2')
    ax.errorbar(agg["horizon"], agg["m_r2_mean"], yerr=agg["m_r2_std"],
                marker='s', markersize=8, linewidth=2, capsize=4,
                color='#FF9800', label='Mediator R2')

    ax.set_xlabel('Post-Meal Horizon (minutes)')
    ax.set_ylabel('R2')
    ax.set_title('A. Prediction Performance by Horizon (Test Set)')
    ax.set_xticks(HORIZONS)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel B: Balance score
    ax = axes[1]
    ax.errorbar(agg["horizon"], agg["bal_mean"], yerr=agg["bal_std"],
                marker='o', markersize=8, linewidth=2, capsize=4,
                color='#4CAF50')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect balance')
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Threshold')

    ax.set_xlabel('Post-Meal Horizon (minutes)')
    ax.set_ylabel('Balance Score')
    ax.set_title('B. Treatment Balance by Horizon (Test Set)')
    ax.set_xticks(HORIZONS)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_horizon_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_horizon_performance.pdf', bbox_inches='tight')
    plt.close()

    print("Saved: fig3_horizon_performance.png/pdf")


def main():
    """Main entry point"""
    CONFIG.ensure_dirs()

    print("\n" + "=" * 60)
    print("HORIZON-SPECIFIC EMBEDDING TRAINING")
    print("=" * 60)

    # Run training
    results_df, embeddings, predictions, test_data = run_horizon_training()

    # Save results
    output_dir = CONFIG.FIGURES_DIR / "horizon_analysis"
    save_horizon_results(results_df, output_dir)

    # Save embeddings
    save_horizon_embeddings(embeddings, test_data, CONFIG.HORIZON_EMBEDDINGS_DIR)

    # Generate visualization
    plot_horizon_performance(results_df, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print(f"Embeddings saved to: {CONFIG.HORIZON_EMBEDDINGS_DIR}")

    print("\n" + "-" * 40)
    print("Performance Summary (Test Set)")
    print("-" * 40)
    summary = results_df.groupby("horizon_min").agg({
        "test_outcome_r2": "mean",
        "test_balance_score": "mean",
    })
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()

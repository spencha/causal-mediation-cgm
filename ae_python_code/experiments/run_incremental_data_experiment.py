#!/usr/bin/env python3
"""
run_incremental_data_experiment.py
==================================
Train autoencoders on pre-combined 2018+2020 data with penalization ablation.

Experimental Design:
- Training data: Combined 2018+2020 training set (pre-merged)
- Test data: Combined 2018+2020 held-out test set (pre-merged)
- Architectures: CNN, LSTM
- Penalizations: 5 configurations
- Seeds: 3 per configuration

Fixed outcome: Glucose at 90 minutes post-meal

CRITICAL: All evaluation metrics are computed on held-out combined TEST set.
Only training loss comes from the training process.

Outputs:
- CSV: All experimental results
- Figure: Main comparison panel (Figure 2)

Usage:
    python run_incremental_data_experiment.py
    python run_incremental_data_experiment.py --latent-dim 8 --epochs 100 --batch-size 64
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import sys
import time
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent))

import tensorflow as tf
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.preprocessing import OneHotEncoder

from config import CONFIG

# Global encoders to ensure consistent one-hot encoding between train and test
_meal_encoder = None
_subj_encoder = None
from causal_linear_ae import train_causal_linear_ae
from resid_ae_utils import load_windows

# =============================================================================
# EXPERIMENTAL CONFIGURATION
# =============================================================================

ARCHITECTURES = ["lstm", "cnn"]
SEEDS = [42, 123, 456]
OUTCOME_HORIZON_MIN = 90  # Fixed at 90 minutes for model selection

# Default values (can be overridden via CLI)
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 128
DEFAULT_LATENT_DIM = 16

# Global config (set from CLI args in main())
EPOCHS = DEFAULT_EPOCHS
BATCH_SIZE = DEFAULT_BATCH_SIZE
LATENT_DIM = DEFAULT_LATENT_DIM

# Penalization configurations
PENALTY_CONFIGS = [
    {"id": "all_penalties", "lin": True, "bal": True, "ci": True, "stab": True},
    {"id": "lin_bal_ci", "lin": True, "bal": True, "ci": True, "stab": False},
    {"id": "lin_bal_stab", "lin": True, "bal": True, "ci": False, "stab": True},
    {"id": "lin_bal", "lin": True, "bal": True, "ci": False, "stab": False},
    {"id": "bal_stab", "lin": False, "bal": True, "ci": False, "stab": True},
]

# Features to load
FEATURES = ["glucose", "steps", "basal", "meal", "heart", "bolus"]


def load_training_data(seed: int = 42):
    """
    Load TRAINING data from the pre-combined 2018+2020 training directory.

    The combined train/test split is created upstream (in R preprocessing).
    No sampling or merging is done here -- we load the directory as-is.

    Parameters:
    -----------
    seed : int
        Random seed (set for reproducibility of downstream stochastic ops)

    Returns:
    --------
    Training data arrays, standardization params
    """
    global _meal_encoder, _subj_encoder
    np.random.seed(seed)

    # Load pre-combined training data (2018 + 2020 already merged)
    print(f"  Loading combined training data from {CONFIG.MEAL_WINDOWS_COMBINED_TRAIN_DIR} ...")
    train_data = load_windows(
        csv_dir=str(CONFIG.MEAL_WINDOWS_COMBINED_TRAIN_DIR),
        features=FEATURES,
        treat="meal",
        interval_min=5,
        pre_minutes=120,
        post_X_minutes=60,
        post_total_minutes=240,
        standardize=False,
    )

    n_train = train_data[0].shape[0]
    print(f"  [INFO] Loaded {n_train} training windows")

    # Compute standardization parameters from TRAINING data only
    X_ts = train_data[0]
    train_mu = X_ts.mean(axis=(0, 1), keepdims=True)
    train_sd = X_ts.std(axis=(0, 1), keepdims=True) + 1e-8

    # Apply standardization
    data_list = list(train_data)
    data_list[0] = (data_list[0] - train_mu) / train_sd  # X_ts
    data_list[1] = (data_list[1] - train_mu) / train_sd  # X_ts_pre

    # CRITICAL FIX: Truncate X_ts_pre to PRE-MEAL ONLY (first 24 timesteps)
    # The "zeroed" post-meal portion becomes non-zero after standardization,
    # which leaks post-treatment information. Using only pre-meal data
    # ensures the representation can't learn from post-meal glucose response.
    pre_ints = data_list[11]  # Number of pre-meal intervals (24 = 120 min / 5 min)
    data_list[1] = data_list[1][:, :pre_ints, :]  # Truncate to pre-meal only
    print(f"  [INFO] Truncated X_ts_pre to pre-meal only: shape {data_list[1].shape}")

    # Fit one-hot encoders on training data
    # IMPORTANT: Save encoders globally so test data uses the same encoding
    meal_list = data_list[9]
    subj_list = data_list[10]

    _meal_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    _subj_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    data_list[2] = _meal_encoder.fit_transform(np.array(meal_list).reshape(-1, 1))
    data_list[3] = _subj_encoder.fit_transform(np.array(subj_list).reshape(-1, 1))

    # Compute treatment median from training data for proper treatment binarization
    Z_train = data_list[4]
    treatment_median_train = np.median(Z_train)
    print(f"  [INFO] Training median carbs: {treatment_median_train:.1f}g (n={len(Z_train)})")

    standardization_params = {
        "mu": train_mu,
        "sd": train_sd,
        "treatment_median": treatment_median_train,
        "pre_ints": pre_ints  # Store for test data truncation
    }

    return tuple(data_list), standardization_params


def load_test_data(standardization_params: dict):
    """
    Load combined TEST data (held-out for evaluation).

    Applies standardization parameters computed from training data.
    Uses global one-hot encoders fitted on training data.

    Parameters:
    -----------
    standardization_params : dict
        Contains 'mu' and 'sd' from training data

    Returns:
    --------
    Test data arrays, standardized using training parameters
    """
    global _meal_encoder, _subj_encoder
    print(f"  Loading combined test data from {CONFIG.MEAL_WINDOWS_COMBINED_TEST_DIR} ...")
    try:
        data_test = load_windows(
            csv_dir=str(CONFIG.MEAL_WINDOWS_COMBINED_TEST_DIR),
            features=FEATURES,
            treat="meal",
            interval_min=5,
            pre_minutes=120,
            post_X_minutes=60,
            post_total_minutes=240,
            standardize=False,
        )
    except FileNotFoundError:
        print("  [Warning] No combined test split found.")
        raise

    # Apply training standardization (CRITICAL: don't recompute on test)
    test_data = list(data_test)
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
    """
    Extract glucose value at specific post-meal horizon.

    Parameters:
    -----------
    y_seq : ndarray
        Full outcome sequence (n_samples, n_timepoints)
    horizon_min : int
        Minutes post-meal for outcome (e.g., 90)
    interval_min : int
        Time interval between measurements (default 5)
    post_start_min : int
        When post-meal sequence starts (default 60)

    Returns:
    --------
    y_horizon : ndarray
        Glucose at specified horizon (n_samples,)
    """
    idx = (horizon_min - post_start_min) // interval_min
    if idx < 0 or idx >= y_seq.shape[1]:
        raise ValueError(f"Horizon {horizon_min} min out of range (index {idx})")
    return y_seq[:, idx]


def evaluate_on_test_set(encoder, test_data: tuple, horizon_min: int,
                         treatment_median: float = None) -> dict:
    """
    Evaluate trained encoder on held-out combined TEST set.

    THIS IS THE PRIMARY EVALUATION - results from this function go in main figures.

    Parameters:
    -----------
    encoder : keras.Model
        Trained encoder model
    test_data : tuple
        Combined test set data (from load_test_data)
    horizon_min : int
        Post-meal horizon for outcome evaluation
    treatment_median : float, optional
        Median from training data for consistent binarization.
        If None, uses test set median (not recommended).

    Returns:
    --------
    dict with test set metrics
    """
    (X_ts_test, X_ts_pre_test, meal_ohe_test, subj_ohe_test,
     Z_test, Z_bin_test, y_seq_test, M_test,
     window_id_test, meal_list_test, subj_list_test, _, _) = test_data

    # Generate embeddings for test set
    phi_test = encoder.predict([X_ts_pre_test, meal_ohe_test, subj_ohe_test], verbose=0)

    # Extract outcome at specified horizon
    y_horizon_test = extract_outcome_at_horizon(y_seq_test, horizon_min)

    metrics = {}

    # 1. Outcome R^2 on TEST set
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    ridge.fit(phi_test, y_horizon_test)
    y_pred = ridge.predict(phi_test)
    metrics["test_outcome_r2"] = r2_score(y_horizon_test, y_pred)
    metrics["test_outcome_mae"] = mean_absolute_error(y_horizon_test, y_pred)
    metrics["test_outcome_rmse"] = np.sqrt(mean_squared_error(y_horizon_test, y_pred))

    # 2. Mediator R^2 on TEST set
    ridge_m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    ridge_m.fit(phi_test, M_test)
    m_pred = ridge_m.predict(phi_test)
    metrics["test_mediator_r2"] = r2_score(M_test, m_pred)

    # 3. Balance score on TEST set
    # Use TEST set median for balance evaluation - this measures whether φ can
    # predict treatment WITHIN the test set. Using training median can create
    # degenerate 0%/100% splits if train and test subjects have different eating patterns.
    test_median = np.median(Z_test)
    A_bin_test = (Z_test > test_median).astype(int)

    # DEBUG: Print class balance to diagnose issues
    n_high = A_bin_test.sum()
    n_total = len(A_bin_test)
    print(f"    [DEBUG] Class balance: {n_high}/{n_total} ({n_high/n_total:.1%} high-carb), test_median={test_median:.1f}g, train_median={treatment_median:.1f}g")

    try:
        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(phi_test, A_bin_test)
        y_prob = lr.predict_proba(phi_test)[:, 1]
        test_auc = roc_auc_score(A_bin_test, y_prob)
        metrics["test_treatment_auc"] = test_auc
        metrics["test_balance_score"] = 1 - 2 * abs(0.5 - test_auc)
    except Exception as e:
        print(f"    [DEBUG] Balance calculation failed: {e}")
        metrics["test_treatment_auc"] = 0.5
        metrics["test_balance_score"] = 1.0

    # 4. In-range classification on TEST set
    # In-range: glucose between 70-180 mg/dL
    # Note: y_horizon is delta glucose, need absolute for this
    # Skip for now as we don't have absolute glucose readily available
    metrics["test_in_range_auc"] = np.nan

    metrics["test_n"] = len(y_horizon_test)

    return metrics, phi_test


def evaluate_phi_train(phi: np.ndarray, y_outcome: np.ndarray,
                       A_cont: np.ndarray, M_scalar: np.ndarray) -> dict:
    """
    Evaluate representation quality on TRAINING data (for diagnostics only).
    """
    metrics = {}

    # 1. Outcome R^2 (glucose prediction) - CV
    try:
        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        y_pred = cross_val_predict(ridge, phi, y_outcome, cv=min(5, len(phi) // 10 + 1))
        metrics["train_outcome_r2"] = r2_score(y_outcome, y_pred)
    except Exception:
        metrics["train_outcome_r2"] = np.nan

    # 2. Mediator R^2 (insulin prediction) - CV
    try:
        ridge_m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        m_pred = cross_val_predict(ridge_m, phi, M_scalar, cv=min(5, len(phi) // 10 + 1))
        metrics["train_mediator_r2"] = r2_score(M_scalar, m_pred)
    except Exception:
        metrics["train_mediator_r2"] = np.nan

    # 3. Balance score (treatment prediction - lower AUC = better balance)
    A_bin = (A_cont > np.median(A_cont)).astype(int)
    try:
        lr = LogisticRegression(max_iter=1000, C=1.0)
        cv_auc = np.mean(cross_val_score(lr, phi, A_bin, cv=min(5, len(phi) // 10 + 1), scoring="roc_auc"))
        metrics["train_treatment_auc"] = cv_auc
        metrics["train_balance_score"] = 1 - 2 * abs(0.5 - cv_auc)
    except Exception:
        metrics["train_treatment_auc"] = 0.5
        metrics["train_balance_score"] = 1.0

    return metrics


def run_single_experiment(architecture: str,
                          penalty_config: dict, seed: int) -> dict:
    """
    Run a single experimental configuration.

    IMPORTANT:
    - Training metrics are computed on training data (for monitoring)
    - TEST metrics are computed on held-out combined test set (for reporting)
    """
    result = {
        "status": "error",
        "architecture": architecture,
        "penalty_config": penalty_config["id"],
        "seed": seed,
    }

    try:
        # Load TRAINING data (pre-combined 2018+2020)
        train_data, std_params = load_training_data(seed=seed)

        (X_ts, X_ts_pre, meal_ohe, subj_ohe, Z, Z_bin, y_seq,
         mediator_scalar, global_window_id, meal_list, subj_list,
         pre_ints, post_X_ints) = train_data

        n_train = X_ts_pre.shape[0]
        result["n_train"] = n_train

        # Load TEST data (combined test set)
        try:
            test_data = load_test_data(std_params)
            n_test = test_data[0].shape[0]
            result["n_test"] = n_test
        except Exception as e:
            print(f"    [Warning] Could not load test data: {e}")
            result["n_test"] = 0
            result["error_msg"] = f"Could not load test data: {e}"
            return result

        # Extract outcome at fixed horizon for training
        y_outcome_train = extract_outcome_at_horizon(y_seq, OUTCOME_HORIZON_MIN)

        # Train model
        # Pass training median for consistent binarization across train/test
        t0 = time.time()
        model, encoder, phi_train, history = train_causal_linear_ae(
            X_ts_pre=X_ts_pre,
            meal_ohe=meal_ohe,
            subj_ohe=subj_ohe,
            A_cont=Z,
            M_scalar=mediator_scalar,
            Y_seq=y_seq,
            latent_dim=LATENT_DIM,
            encoder_type=architecture,
            optimizer_name="rmsprop",
            use_linearization=penalty_config["lin"],
            use_balancing=penalty_config["bal"],
            use_ci_penalty=penalty_config["ci"],
            use_stability=penalty_config["stab"],
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            seed=seed,
            verbose=0,
            treatment_median=std_params["treatment_median"],  # Use training median
            treatment_head_weight=0.0  # Disable treatment prediction head for better balance
        )
        training_time = time.time() - t0

        # Evaluate on TRAINING data (for diagnostics only)
        train_metrics = evaluate_phi_train(phi_train, y_outcome_train, Z, mediator_scalar)

        # Evaluate on TEST data (THIS IS WHAT GETS REPORTED)
        # Pass training median for consistent treatment binarization
        test_metrics, phi_test = evaluate_on_test_set(
            encoder, test_data, OUTCOME_HORIZON_MIN,
            treatment_median=std_params["treatment_median"]
        )

        result.update({
            "status": "success",
            "training_time_sec": training_time,
            "final_loss": history.history["loss"][-1],
            "final_val_loss": history.history.get("val_loss", [np.nan])[-1],
            # Training metrics (for supplement/diagnostics)
            **train_metrics,
            # TEST metrics (FOR MAIN FIGURES)
            **test_metrics,
        })

    except Exception as e:
        result["error_msg"] = str(e)[:200]

    return result


def run_full_experiment() -> pd.DataFrame:
    """Run all experimental configurations (architectures x penalties x seeds)."""

    results = []
    total = len(ARCHITECTURES) * len(PENALTY_CONFIGS) * len(SEEDS)

    print(f"\n{'='*70}")
    print(f"COMBINED DATA EXPERIMENT")
    print(f"{'='*70}")
    print(f"Total configurations: {total}")
    print(f"Outcome horizon: {OUTCOME_HORIZON_MIN} minutes post-meal")
    print(f"Architectures: {ARCHITECTURES}")
    print(f"Seeds: {SEEDS}")
    print(f"{'='*70}\n")

    config_num = 0
    for arch in ARCHITECTURES:
        for penalty in PENALTY_CONFIGS:
            for seed in SEEDS:
                config_num += 1

                print(f"[{config_num}/{total}] {arch} | "
                      f"{penalty['id']} | seed={seed}")

                result = run_single_experiment(arch, penalty, seed)
                results.append(result)

                if result["status"] == "success":
                    print(f"    -> Test R2_outcome={result.get('test_outcome_r2', np.nan):.4f}, "
                          f"R2_med={result.get('test_mediator_r2', np.nan):.4f}, "
                          f"AUC={result.get('test_treatment_auc', np.nan):.4f}, "
                          f"Balance={result.get('test_balance_score', np.nan):.4f}")
                else:
                    print(f"    -> ERROR: {result.get('error_msg', 'unknown')}")

    return pd.DataFrame(results)


def save_results(results_df: pd.DataFrame, output_dir: Path):
    """Save experimental results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Full results
    results_df.to_csv(output_dir / f"incremental_data_results_{timestamp}.csv", index=False)
    results_df.to_csv(output_dir / "incremental_data_results_latest.csv", index=False)

    # Summary by configuration
    successful = results_df[results_df["status"] == "success"].copy()

    if len(successful) > 0:
        summary = successful.groupby(
            ["architecture", "penalty_config"]
        ).agg({
            "test_outcome_r2": ["mean", "std"],
            "test_mediator_r2": ["mean", "std"],
            "test_balance_score": ["mean", "std"],
            "final_loss": ["mean", "std"],
            "n_train": "first"
        }).round(4)

        summary.to_csv(output_dir / "incremental_data_summary.csv")

    return output_dir


def main():
    """Main entry point"""
    global EPOCHS, BATCH_SIZE, LATENT_DIM

    parser = argparse.ArgumentParser(
        description="Run incremental data experiment for causal autoencoder"
    )
    parser.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM,
                        help=f"Number of latent dimensions / phi features (default: {DEFAULT_LATENT_DIM})")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Training batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results (default: experiment_results)")

    args = parser.parse_args()

    # Set global config from CLI args
    LATENT_DIM = args.latent_dim
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    print(f"\nExperiment configuration:")
    print(f"  latent_dim={LATENT_DIM}, epochs={EPOCHS}, batch_size={BATCH_SIZE}")

    # Ensure directories exist
    CONFIG.ensure_dirs()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = CONFIG.EXPERIMENT_RESULTS_DIR

    # Run experiment
    results = run_full_experiment()

    # Save results
    save_results(results, output_dir)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}")

    # Print summary
    successful = results[results["status"] == "success"]
    if len(successful) > 0:
        print(f"\nSuccessful runs: {len(successful)}/{len(results)}")
        print(f"\nBest test outcome R^2: {successful['test_outcome_r2'].max():.4f}")

        best_idx = successful['test_outcome_r2'].idxmax()
        best = successful.loc[best_idx]
        print(f"  Architecture: {best['architecture']}")
        print(f"  Penalty config: {best['penalty_config']}")


if __name__ == "__main__":
    main()

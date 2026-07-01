#!/usr/bin/env python3
"""
Train best model configuration and export embeddings for CMA.

This script:
1. Loads combined 2018+2020 training and test data
2. Trains a specific autoencoder configuration
3. Extracts phi embeddings for both train and test sets
4. Exports embeddings to CSV for npCBPS and mediation analysis

Usage:
    python train_and_export_embeddings.py --arch cnn --penalty lin_bal --seed 42

    # With custom latent dimension, epochs, and batch size:
    python train_and_export_embeddings.py --arch cnn --latent-dim 8 --epochs 50 --batch-size 64
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time
from sklearn.decomposition import PCA

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from resid_ae_utils import load_windows
from causal_linear_ae import train_causal_linear_ae
from diatrend_loader import load_diatrend_data
from sklearn.preprocessing import OneHotEncoder

# Import config
sys.path.insert(0, str(Path(__file__).parent.parent / "cma_cluster"))
try:
    from config import CONFIG
except ImportError:
    # Fallback paths
    class CONFIG:
        BASE_DIR = Path(__file__).parent.parent
        AE_CODE_DIR = Path(__file__).parent
        MEAL_WINDOWS_COMBINED_TRAIN_DIR = AE_CODE_DIR / "meal_windows_combined" / "train"
        MEAL_WINDOWS_COMBINED_TEST_DIR = AE_CODE_DIR / "meal_windows_combined" / "test"
        ANALYSIS_DATA_DIR = BASE_DIR / "analysis_data"


# ============================================================================
# CONFIGURATION
# ============================================================================

# NOTE: "bolus" is intentionally excluded from encoder features to prevent
# mediator leakage. ~50% of subjects pre-bolus before eating, so pre-meal
# bolus in the encoder input would leak the mediator (bolus_for_meal) into
# phi. The mediator is computed separately in load_windows() from the raw
# "bolus" column (pre + 60min post-meal sum) and is unaffected by this list.
# The glucose channel implicitly captures IOB effects (insulin on board from
# prior boluses manifests as a downward glucose trend).
FEATURES = ["glucose", "steps", "basal", "meal", "heart"]
LATENT_DIM = 16
EPOCHS = 100
BATCH_SIZE = 32
OUTCOME_HORIZON_MIN = 90  # 90 minutes post-meal

# Penalty configurations
# Based on ablation study results:
# - "balancing" achieves best treatment balance (AUC=0.315, closest to ideal 0.5)
# - Adding linearization slightly degrades balance (AUC=0.356 for lin_bal)
# - For causal mediation analysis, prefer configs with balancing enabled
PENALTY_CONFIGS = {
    "all_penalties": {"lin": True, "bal": True, "ci": True, "stab": True},
    "lin_bal_ci": {"lin": True, "bal": True, "ci": True, "stab": False},
    "lin_bal_stab": {"lin": True, "bal": True, "ci": False, "stab": True},
    "lin_bal": {"lin": True, "bal": True, "ci": False, "stab": False},
    "bal_stab": {"lin": False, "bal": True, "ci": False, "stab": True},
    "balancing": {"lin": False, "bal": True, "ci": False, "stab": False},  # Best balance (AUC=0.315)
    "none": {"lin": False, "bal": False, "ci": False, "stab": False},  # Baseline (best R², poor balance)
}


def load_train_data():
    """Load combined 2018+2020 training data.

    Loads pre-split training data from CONFIG.MEAL_WINDOWS_COMBINED_TRAIN_DIR,
    which contains meal windows from both the 2018 and 2020 cohorts.
    Cohort membership is inferred from subject_id prefixes (e.g., '2018_1' -> '2018').
    """

    print("  Loading combined training data...")
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
    subj_list = train_data[10]

    # Determine cohort from subject_id prefix (e.g., "2018_1" -> "2018", "2020_3" -> "2020")
    cohort_labels = np.array([s.split("_")[0] for s in subj_list])
    n_2018 = np.sum(cohort_labels == '2018')
    n_2020 = np.sum(cohort_labels == '2020')
    print(f"  [INFO] Combined train: {n_train} total ({n_2020} from 2020, {n_2018} from 2018)")

    Z_train = train_data[4]
    treatment_median = np.median(Z_train)
    print(f"  [INFO] Combined train median carbs: {treatment_median:.1f}g (n={len(Z_train)})")

    # Compute standardization parameters
    X_ts = train_data[0]
    train_mu = X_ts.mean(axis=(0, 1), keepdims=True)
    train_sd = X_ts.std(axis=(0, 1), keepdims=True) + 1e-8

    # IMPORTANT: Extract raw glucose at meal BEFORE standardization
    # Glucose is the first feature (index 0), meal time is at pre_ints index
    pre_ints_raw = train_data[11]  # pre_ints before converting to list
    glucose_idx = 0
    glucose_at_meal_raw = X_ts[:, pre_ints_raw, glucose_idx].copy()  # Raw mg/dL values
    print(f"  [INFO] Raw glucose at meal: mean={glucose_at_meal_raw.mean():.1f}, "
          f"range=[{glucose_at_meal_raw.min():.0f}, {glucose_at_meal_raw.max():.0f}] mg/dL")

    # Apply standardization
    data_list = list(train_data)
    data_list[0] = (data_list[0] - train_mu) / train_sd
    data_list[1] = (data_list[1] - train_mu) / train_sd

    # Truncate X_ts_pre to pre-meal only
    pre_ints = data_list[11]
    data_list[1] = data_list[1][:, :pre_ints, :]
    print(f"  [INFO] Truncated X_ts_pre to pre-meal only: shape {data_list[1].shape}")

    # Re-encode one-hot encodings
    meal_list = data_list[9]
    subj_list = data_list[10]

    meal_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    subj_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    data_list[2] = meal_encoder.fit_transform(np.array(meal_list).reshape(-1, 1))
    data_list[3] = subj_encoder.fit_transform(np.array(subj_list).reshape(-1, 1))

    # Training median for treatment binarization
    Z_train_combined = data_list[4]
    treatment_median_train = np.median(Z_train_combined)
    print(f"  [INFO] Final training median for binarization: {treatment_median_train:.1f}g")

    standardization_params = {
        "mu": train_mu,
        "sd": train_sd,
        "treatment_median": treatment_median_train,
        "pre_ints": pre_ints,
        "meal_encoder": meal_encoder,
        "subj_encoder": subj_encoder,
        "glucose_at_meal_raw": glucose_at_meal_raw,  # Raw glucose in mg/dL
    }

    return tuple(data_list), cohort_labels, standardization_params


def load_test_data(standardization_params: dict):
    """Load combined 2018+2020 test data with consistent standardization.

    Standardization parameters (mean, std) are computed from training data only
    and applied here to the test set. Cohort membership is inferred from
    subject_id prefixes (e.g., '2018_1' -> '2018').

    Returns:
        tuple: (test_data_tuple, cohort_labels, glucose_at_meal_raw)
            - test_data_tuple: standardized test data
            - cohort_labels: numpy array of '2018' or '2020' per sample
            - glucose_at_meal_raw: raw glucose values in mg/dL
    """

    print("  Loading combined test data...")
    test_data = load_windows(
        csv_dir=str(CONFIG.MEAL_WINDOWS_COMBINED_TEST_DIR),
        features=FEATURES,
        treat="meal",
        interval_min=5,
        pre_minutes=120,
        post_X_minutes=60,
        post_total_minutes=240,
        standardize=False,
    )

    # Determine cohort from subject_id prefix
    subj_list = test_data[10]
    cohort_labels = np.array([s.split("_")[0] for s in subj_list])
    n_test = test_data[0].shape[0]
    n_2018 = np.sum(cohort_labels == '2018')
    n_2020 = np.sum(cohort_labels == '2020')
    print(f"  [INFO] Combined test: {n_test} total ({n_2020} from 2020, {n_2018} from 2018)")

    # IMPORTANT: Extract raw glucose at meal BEFORE standardization
    X_ts_raw = test_data[0]
    pre_ints = standardization_params.get("pre_ints", 24)
    glucose_idx = 0
    glucose_at_meal_raw = X_ts_raw[:, pre_ints, glucose_idx].copy()  # Raw mg/dL values
    print(f"  [INFO] Test raw glucose at meal: mean={glucose_at_meal_raw.mean():.1f}, "
          f"range=[{glucose_at_meal_raw.min():.0f}, {glucose_at_meal_raw.max():.0f}] mg/dL")

    # Apply standardization from training
    test_data = list(test_data)
    test_data[0] = (test_data[0] - standardization_params["mu"]) / standardization_params["sd"]
    test_data[1] = (test_data[1] - standardization_params["mu"]) / standardization_params["sd"]

    # Truncate to pre-meal only
    test_data[1] = test_data[1][:, :pre_ints, :]

    # Use fitted encoders from training
    meal_encoder = standardization_params["meal_encoder"]
    subj_encoder = standardization_params["subj_encoder"]

    meal_list = test_data[9]
    subj_list = test_data[10]

    test_data[2] = meal_encoder.transform(np.array(meal_list).reshape(-1, 1))
    test_data[3] = subj_encoder.transform(np.array(subj_list).reshape(-1, 1))

    return tuple(test_data), cohort_labels, glucose_at_meal_raw


def train_and_export(arch: str, penalty_id: str, seed: int, output_dir: Path,
                     latent_dim: int = LATENT_DIM, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE):
    """Train model on combined 2018+2020 data and export embeddings.

    Parameters:
    -----------
    arch : str
        Encoder architecture ('cnn' or 'lstm')
    penalty_id : str
        Penalty configuration ID
    seed : int
        Random seed
    output_dir : Path
        Output directory for embeddings
    latent_dim : int
        Number of latent dimensions (phi features)
    epochs : int
        Number of training epochs
    batch_size : int
        Training batch size
    """

    np.random.seed(seed)

    penalty_config = PENALTY_CONFIGS[penalty_id]

    print(f"\n{'='*70}")
    print(f"TRAINING: {arch} | combined 2018+2020 | {penalty_id} | seed={seed}")
    print(f"  latent_dim={latent_dim}, epochs={epochs}, batch_size={batch_size}")
    print(f"{'='*70}")

    # Load combined training and test data
    train_data, cohort_labels, std_params = load_train_data()
    test_data, cohort_labels_test, glucose_test_raw = load_test_data(std_params)

    # Unpack training data (14 elements from load_windows)
    (X_ts, X_ts_pre, meal_ohe, subj_ohe, Z, Z_bin, y_seq,
     mediator_scalar, global_ids, meal_list, subj_list,
     pre_ints, post_X_ints, total_bolus_train) = train_data

    # Use raw glucose at meal from std_params (extracted BEFORE standardization)
    glucose_at_meal = std_params["glucose_at_meal_raw"]

    # Train model
    print(f"\n  Training {arch.upper()} encoder...")
    t0 = time.time()

    model, encoder, phi_train, history = train_causal_linear_ae(
        X_ts_pre=X_ts_pre,
        meal_ohe=meal_ohe,
        subj_ohe=subj_ohe,
        A_cont=Z,
        M_scalar=mediator_scalar,
        Y_seq=y_seq,
        latent_dim=latent_dim,
        encoder_type=arch,
        optimizer_name="adamw",
        use_linearization=penalty_config["lin"],
        use_balancing=penalty_config["bal"],
        use_ci_penalty=penalty_config["ci"],
        use_stability=penalty_config["stab"],
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        verbose=1,
        treatment_median=std_params["treatment_median"],
        treatment_head_weight=0.0
    )

    training_time = time.time() - t0
    print(f"  Training completed in {training_time:.1f}s")

    # Extract test embeddings (14 elements from load_windows)
    (X_ts_test, X_ts_pre_test, meal_ohe_test, subj_ohe_test, Z_test, Z_bin_test, y_seq_test,
     mediator_test, global_ids_test, meal_list_test, subj_list_test,
     pre_ints_test, post_X_ints_test, total_bolus_test) = test_data

    # Use raw glucose at meal (extracted BEFORE standardization in load_test_data)
    glucose_test = glucose_test_raw

    phi_test = encoder.predict([X_ts_pre_test, meal_ohe_test, subj_ohe_test], verbose=0)

    print(f"  Train embeddings shape: {phi_train.shape}")
    print(f"  Test embeddings shape: {phi_test.shape}")

    # ========================================================================
    # COMPUTE PCA ON PHI EMBEDDINGS
    # ========================================================================
    # Fit PCA on training data, transform both train and test
    # This provides orthogonal components for CMA (eliminates multicollinearity)

    n_pca_components = min(10, phi_train.shape[1])  # Up to 10 PCs
    pca = PCA(n_components=n_pca_components)
    pc_train = pca.fit_transform(phi_train)
    pc_test = pca.transform(phi_test)

    print(f"  PCA: {n_pca_components} components, variance explained: {pca.explained_variance_ratio_.sum():.1%}")
    print(f"    PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%}, PC3: {pca.explained_variance_ratio_[2]:.1%}")

    # ========================================================================
    # EXPORT TRAINING EMBEDDINGS
    # ========================================================================

    print(f"\n  Exporting training embeddings...")

    train_df = pd.DataFrame(
        phi_train,
        columns=[f"phi_{i+1}" for i in range(phi_train.shape[1])]
    )
    train_df["global_window_id"] = global_ids.astype(int)
    train_df["subject_id"] = subj_list
    train_df["meal_type"] = meal_list
    train_df["treat_meal_carbs"] = Z.astype(float)
    train_df["mediator_bolus_for_meal"] = mediator_scalar.astype(float)
    train_df["total_bolus"] = total_bolus_train.astype(float)
    train_df["glucose_at_meal"] = glucose_at_meal.astype(float)
    train_df["cohort"] = cohort_labels
    train_df["split"] = "train"

    # Add outcome at ALL time points (5-minute resolution for CMA analysis)
    # IMPORTANT: y_seq is already delta glucose (glucose_at_time - glucose_at_baseline)
    # computed in resid_ae_utils.py as y_seq_delta = (y_raw - base_glucose)
    # y_seq has time points covering 60-210+ minutes post-meal at 5-min resolution
    # Index mapping: y_seq[:, i] = delta glucose at (60 + i*5) minutes post-meal
    # Export all timepoints from 60 to 210 min (indices 0-30)
    for t_idx in range(min(31, y_seq.shape[1])):  # indices 0-30 for 60-210 min
        t_min = 60 + t_idx * 5
        train_df[f"Y_{t_min}min"] = y_seq[:, t_idx].astype(float)

    # Add PCA components (orthogonal features for CMA)
    for i in range(n_pca_components):
        train_df[f"PC_{i+1}"] = pc_train[:, i].astype(float)

    # ========================================================================
    # EXPORT TEST EMBEDDINGS
    # ========================================================================

    print(f"  Exporting test embeddings...")

    test_df = pd.DataFrame(
        phi_test,
        columns=[f"phi_{i+1}" for i in range(phi_test.shape[1])]
    )
    test_df["global_window_id"] = global_ids_test.astype(int)
    test_df["subject_id"] = subj_list_test
    test_df["meal_type"] = meal_list_test
    test_df["treat_meal_carbs"] = Z_test.astype(float)
    test_df["mediator_bolus_for_meal"] = mediator_test.astype(float)
    test_df["total_bolus"] = total_bolus_test.astype(float)
    test_df["glucose_at_meal"] = glucose_test.astype(float)
    test_df["cohort"] = cohort_labels_test
    test_df["split"] = "test"

    # Add outcome at ALL time points for test data (5-minute resolution)
    # Same mapping as train: y_seq is already delta glucose from resid_ae_utils.py
    # Export all timepoints from 60 to 210 min (indices 0-30)
    for t_idx in range(min(31, y_seq_test.shape[1])):  # indices 0-30 for 60-210 min
        t_min = 60 + t_idx * 5
        test_df[f"Y_{t_min}min"] = y_seq_test[:, t_idx].astype(float)

    # Add PCA components (orthogonal features for CMA)
    for i in range(n_pca_components):
        test_df[f"PC_{i+1}"] = pc_test[:, i].astype(float)

    # ========================================================================
    # SAVE TO CSV
    # ========================================================================

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_str = f"{arch}_combined_{penalty_id}_seed{seed}"

    # Save train embeddings
    train_file = output_dir / f"phi_embeddings_train_{config_str}.csv"
    train_df.to_csv(train_file, index=False)
    print(f"  Saved: {train_file}")

    # Save test embeddings
    test_file = output_dir / f"phi_embeddings_test_{config_str}.csv"
    test_df.to_csv(test_file, index=False)
    print(f"  Saved: {test_file}")

    # Save combined (for CMA that uses both)
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_file = output_dir / f"phi_embeddings_combined_{config_str}.csv"
    combined_df.to_csv(combined_file, index=False)
    print(f"  Saved: {combined_file}")

    # ========================================================================
    # PRINT SUMMARY
    # ========================================================================

    print(f"\n{'='*70}")
    print(f"EXPORT COMPLETE")
    print(f"{'='*70}")
    print(f"  Configuration: {config_str}")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"  Embedding dimensions: {phi_train.shape[1]}")
    print(f"\n  Output files:")
    print(f"    - {train_file}")
    print(f"    - {test_file}")
    print(f"    - {combined_file}")
    print(f"\n  Next steps:")
    print(f"    1. Run npCBPS: Rscript cma_cluster/ohiot1dm/npcbps_weights.R")
    print(f"    2. Run CMA: Rscript cma_cluster/ohiot1dm/run_mixed_effects_mediation.R \\")
    print(f"         --phi-file {test_file} --dataset 2020_TEST")

    return {
        "train_file": str(train_file),
        "test_file": str(test_file),
        "combined_file": str(combined_file),
        "n_train": len(train_df),
        "n_test": len(test_df),
    }


def train_and_export_diatrend(
    *,
    arch: str,
    penalty_id: str,
    seed: int,
    output_dir: Path,
    raw_dir: Path,
    features: tuple[str, ...],
    bob_dia_min: float | None,
    cohorts: set[int] | None,
    test_frac: float = 0.0,
    latent_dim: int = LATENT_DIM,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
):
    """Train the CLAE on DiaTrend episodes and export a single combined CSV.

    DiaTrend has no natural train/test split (54 subjects, no original
    protocol split). The CLAE is trained on whatever subset is admitted
    by ``cohorts`` and embeddings are exported for the same subset in
    one CSV. Downstream R-side mediation filters by cohort within that
    single CSV.
    """
    np.random.seed(seed)
    penalty_config = PENALTY_CONFIGS[penalty_id]

    print(f"\n{'='*70}")
    print(f"TRAINING (DiaTrend): {arch} | {penalty_id} | seed={seed}")
    print(f"  features={features} | bob_dia_min={bob_dia_min} | cohorts={cohorts}")
    print(f"  latent_dim={latent_dim}, epochs={epochs}, batch_size={batch_size}")
    print(f"{'='*70}")

    bob_params = None
    if bob_dia_min is not None:
        from data_processing.diatrend.bob_kernel import IOBKernelParams as _BobParams

        bob_params = _BobParams(dia_min=float(bob_dia_min))

    data, cohort_labels, std_params = load_diatrend_data(
        raw_dir,
        features=features,
        standardize=True,
        bob_params=bob_params,
        cohorts=cohorts,
        test_frac=test_frac,
    )

    (
        X_ts,
        X_ts_pre,
        meal_ohe,
        subj_ohe,
        Z,
        Z_bin,
        y_seq,
        mediator_scalar,
        global_ids,
        meal_list,
        subj_list,
        pre_ints,
        post_X_ints,
        total_bolus_arr,
    ) = data

    # Mirror the OhioT1DM driver's truncation of X_ts_pre to pre_ints.
    X_ts_pre = X_ts_pre[:, :pre_ints, :]
    glucose_at_meal = std_params["glucose_at_meal_raw"]
    iob_at_meal = std_params["iob_at_meal"]
    split_labels = std_params["split_labels"]

    # OhioT1DM-style train/test discipline: the encoder and PCA are fit on
    # the train rows only; embeddings are then produced for every episode
    # from that train-fit encoder. Downstream mediation runs on the held-out
    # test rows (filter on the exported `split` column). test_frac == 0
    # labels everything "all", so train_mask is all-True (single-split mode).
    train_mask = split_labels != "test"
    n_train, n_test = int(train_mask.sum()), int((~train_mask).sum())

    print(f"  Loaded {X_ts.shape[0]} DiaTrend episodes")
    print(
        f"  Within-subject temporal split (test_frac={test_frac}): "
        f"{n_train} train, {n_test} test"
    )
    if std_params.get("parse_errors"):
        n_errors = len(std_params["parse_errors"])
        print(f"  [WARN] {n_errors} workbook(s) failed to parse; see diagnostic report.")

    print(f"\n  Training {arch.upper()} encoder on the train split...")
    t0 = time.time()
    model, encoder, _phi_train, history = train_causal_linear_ae(
        X_ts_pre=X_ts_pre[train_mask],
        meal_ohe=meal_ohe[train_mask],
        subj_ohe=subj_ohe[train_mask],
        A_cont=Z[train_mask],
        M_scalar=mediator_scalar[train_mask],
        Y_seq=y_seq[train_mask],
        latent_dim=latent_dim,
        encoder_type=arch,
        optimizer_name="adamw",
        use_linearization=penalty_config["lin"],
        use_balancing=penalty_config["bal"],
        use_ci_penalty=penalty_config["ci"],
        use_stability=penalty_config["stab"],
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        verbose=1,
        treatment_median=std_params["treatment_median"],
        treatment_head_weight=0.0,
    )
    print(f"  Training completed in {time.time() - t0:.1f}s")

    # Embed ALL episodes with the train-fit encoder (test rows are out-of-sample).
    phi = encoder.predict([X_ts_pre, meal_ohe, subj_ohe], verbose=0)
    print(f"  Embeddings shape: {phi.shape}")

    # PCA fit on train embeddings, applied to all (mirrors OhioT1DM, which
    # fits PCA on the train split and transforms the test split).
    n_pca_components = min(10, phi.shape[1])
    pca = PCA(n_components=n_pca_components)
    pca.fit(phi[train_mask])
    pc = pca.transform(phi)
    print(
        f"  PCA: {n_pca_components} components, variance explained: "
        f"{pca.explained_variance_ratio_.sum():.1%}"
    )

    out_df = pd.DataFrame(
        phi, columns=[f"phi_{i+1}" for i in range(phi.shape[1])]
    )
    out_df["global_window_id"] = global_ids.astype(int)
    out_df["subject_id"] = subj_list
    out_df["meal_type"] = meal_list
    out_df["cohort"] = cohort_labels
    out_df["treat_meal_carbs"] = Z.astype(float)
    out_df["mediator_bolus_for_meal"] = mediator_scalar.astype(float)
    out_df["total_bolus"] = total_bolus_arr.astype(float)
    out_df["glucose_at_meal"] = glucose_at_meal.astype(float)
    out_df["iob_at_meal"] = iob_at_meal.astype(float)
    out_df["split"] = split_labels.astype(str)

    for t_idx in range(min(31, y_seq.shape[1])):
        out_df[f"Y_{60 + t_idx * 5}min"] = y_seq[:, t_idx].astype(float)
    for i in range(n_pca_components):
        out_df[f"PC_{i+1}"] = pc[:, i].astype(float)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feat_tag = "_".join(features) if isinstance(features, tuple) else features
    config_str = (
        f"{arch}_diatrend_{penalty_id}_seed{seed}_ld{latent_dim}_feats-{feat_tag}"
    )
    out_file = output_dir / f"phi_embeddings_diatrend_{config_str}.csv"
    out_df.to_csv(out_file, index=False)
    print(f"  Saved: {out_file}")

    print(f"\n{'='*70}")
    print("EXPORT COMPLETE (DiaTrend)")
    print(f"{'='*70}")
    print(f"  Output: {out_file}")
    print(f"  Episodes: {len(out_df)}")
    print(f"  Cohort 1: {(cohort_labels == '1').sum()}")
    print(f"  Cohort 2: {(cohort_labels == '2').sum()}")
    print(f"  Split: {(split_labels == 'train').sum()} train / "
          f"{(split_labels == 'test').sum()} test / "
          f"{(split_labels == 'all').sum()} unsplit")
    print(f"\n  Next: Rscript cma_cluster/diatrend/run_mixed_effects_mediation.R \\")
    print(f"           --phi-file {out_file}")

    return {"output_file": str(out_file), "n_episodes": len(out_df)}


def main():
    parser = argparse.ArgumentParser(description="Train CLAE and export embeddings for CMA")
    parser.add_argument("--dataset", type=str, default="ohiot1dm",
                        choices=["ohiot1dm", "diatrend"],
                        help="Which dataset to train on (default: ohiot1dm)")
    parser.add_argument("--arch", type=str, default="cnn", choices=["cnn", "lstm"],
                        help="Architecture (default: cnn)")
    parser.add_argument("--penalty", type=str, default="lin_bal",
                        choices=list(PENALTY_CONFIGS.keys()),
                        help="Penalty configuration (default: lin_bal)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default depends on --dataset)")
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM,
                        help=f"Number of latent dimensions / phi features (default: {LATENT_DIM})")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Number of training epochs (default: {EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Training batch size (default: {BATCH_SIZE})")

    # DiaTrend-specific arguments (ignored when --dataset ohiot1dm)
    parser.add_argument("--diatrend-raw-dir", type=str, default=None,
                        help="DiaTrend raw .xlsx directory (default: CONFIG.DIATREND_RAW_DIR)")
    parser.add_argument("--diatrend-features", type=str, default="glucose",
                        help=(
                            "Comma-separated CLAE input channels for DiaTrend. "
                            "Default 'glucose' is the Section 8.2 primary univariate spec; "
                            "use 'glucose,meal,bolus' for the Section 8.6 multivariate "
                            "sensitivity arm."
                        ))
    parser.add_argument("--diatrend-cohorts", type=str, default=None,
                        help=(
                            "Comma-separated cohort IDs to include for DiaTrend "
                            "(e.g. '2' for the primary 37-subject arm, '1,2' for the "
                            "full 54-subject robustness arm). Default: include all cohorts."
                        ))
    parser.add_argument("--diatrend-test-frac", type=float, default=0.2,
                        help=(
                            "Within-subject temporal test fraction for DiaTrend "
                            "(OhioT1DM-style split). The latest this fraction of each "
                            "subject's meals are held out as test; the encoder + PCA are "
                            "fit on train and mediation runs on test. Set 0.0 to disable "
                            "the split (single-sample mode). Default: 0.2."
                        ))
    parser.add_argument("--diatrend-bob-dia-min", type=float, default=None,
                        help=(
                            "If set, populate cohort-1 IOB via the Section 8.5 kernel-derived "
                            "BOB using this DIA (in minutes). Cohort 2 keeps pump IOB. "
                            "Default: leave cohort-1 IOB as NaN."
                        ))

    args = parser.parse_args()
    project_root = Path(__file__).parent.parent

    if args.dataset == "ohiot1dm":
        if args.output_dir is None:
            output_dir = CONFIG.ANALYSIS_DATA_DIR / "embeddings"
        else:
            output_dir = Path(args.output_dir)
            if not output_dir.is_absolute():
                output_dir = project_root / output_dir
        return train_and_export(
            arch=args.arch,
            penalty_id=args.penalty,
            seed=args.seed,
            output_dir=output_dir,
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

    # DiaTrend branch
    raw_dir = Path(args.diatrend_raw_dir) if args.diatrend_raw_dir else CONFIG.DIATREND_RAW_DIR
    if args.output_dir is None:
        output_dir = CONFIG.DIATREND_EMBEDDINGS_DIR
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir

    features = tuple(s.strip() for s in args.diatrend_features.split(",") if s.strip())
    cohorts: set[int] | None = None
    if args.diatrend_cohorts:
        cohorts = {int(c.strip()) for c in args.diatrend_cohorts.split(",") if c.strip()}

    return train_and_export_diatrend(
        arch=args.arch,
        penalty_id=args.penalty,
        seed=args.seed,
        output_dir=output_dir,
        raw_dir=raw_dir,
        features=features,
        bob_dia_min=args.diatrend_bob_dia_min,
        cohorts=cohorts,
        test_frac=args.diatrend_test_frac,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

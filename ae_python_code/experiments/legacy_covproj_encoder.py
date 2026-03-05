#!/usr/bin/env python3
"""
Run the covariate-projection encoder (default) and export phi embeddings for CMA.

- Loads meal windows
- Trains either:
    * Covariate-projection AE (DEFAULT): φ → linear head predicts pre-treatment covariates C
      (optionally with a tiny aux head for early ΔG steps)
    * Residual-targeted AE (legacy):     φ predicts residuals Rm, Ry (kept as fallback)
- Exports CSV with phi_* columns + identifiers/labels

Assumes resid_ae_utils.py is in the same directory.
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from resid_ae_utils import (
    load_windows,
    train_covproj_encoder,  # NEW
    train_residual_encoder, # fallback if needed
)

# --------------------------
# Config (edit as needed)
# --------------------------
CSV_DIR   = "meal_windows_2018"  # relative to this script; change if needed
FEATURES  = ["skintemp", "heart", "steps", "sleep_fraction", "glucose", "basal"]  # add "bolus","basal" if available

# Model/training params (shared-ish)
LATENT_DIM   = 32          # a bit larger for cov-proj
EPOCHS       = 40
BATCH_SIZE   = 128
LR           = 1e-3
L2_REG       = 1e-4
SEED         = 123
VERBOSE      = 2

# Residual AE knobs (for --mode residual)
K_FOLDS      = 5
USE_MMD_RES  = True
GAMMA_MAX    = 1e-3
M_LOSS       = "mse"
Y_LOSS       = "mse"

# Cov-proj knobs
ADD_FLAT_PCA       = True   # append PCA of flattened pre-window to C
FLAT_PCA_COMPONENTS= 32
ADD_AUX_Y2         = True   # tiny aux predicting first ΔG steps
Y2_IDX             = (0, 1) # (+60, +65 min when ΔG starts at +60)

def build_feat_index(features):
    """Map required names to column indices if present."""
    name_to_idx = {nm: i for i, nm in enumerate(features)}
    def idx_or_none(n): return name_to_idx[n] if n in name_to_idx else None
    if "glucose" not in name_to_idx:
        raise ValueError("FEATURES must include 'glucose' for covariate projection.")
    return {
        "glucose": name_to_idx["glucose"],
        "bolus":   idx_or_none("bolus"),
        "basal":   idx_or_none("basal"),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["covproj", "residual"], default="covproj",
                    help="Training mode: covproj (default) or residual (legacy).")
    args = ap.parse_args()

    t0 = time.time()

    # --------------------------
    # Load data
    # --------------------------
    (X_ts, X_ts_pre, meal_ohe, subj_ohe, Z, Z_bin, y_seq, mediator_scalar,
     global_window_id,
     meal_list, subj_list, pre_ints, post_X_ints) = load_windows(
        csv_dir=CSV_DIR,
        features=FEATURES,
        treat="meal",
        interval_min=5,
        pre_minutes=120,
        post_X_minutes=60,
        post_total_minutes=240,
        standardize=True,
        y_scalar_fn="mean"  # not used
    )

    n = X_ts_pre.shape[0]
    print(f"Loaded {n} windows from {Path(CSV_DIR).resolve()}")
    print("X_ts_pre:", X_ts_pre.shape,
          "  Δy_seq:", y_seq.shape,
          "  mediator_scalar:", mediator_scalar.shape)

    # --------------------------
    # Train encoder
    # --------------------------
    if args.mode == "covproj":
        print("[info] Training covariate-projection encoder (default).")
        feat_index = build_feat_index(FEATURES)

        model, enc, phi, meta, hist = train_covproj_encoder(
            X_ts_pre=X_ts_pre,
            meal_ohe=meal_ohe,
            subj_ohe=subj_ohe,
            outcome_seq=y_seq,          # only for optional Y2 aux head
            pre_len=pre_ints,
            feat_index=feat_index,
            latent_dim=LATENT_DIM,
            add_flatten_pca=ADD_FLAT_PCA,
            flatten_pca_components=FLAT_PCA_COMPONENTS,
            add_aux_y2=ADD_AUX_Y2,
            y2_indices=Y2_IDX,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            l2_reg=L2_REG,
            use_mmd=False,              # keep False for linear recoverability of C
            mmd_gamma_max=0.0,
            verbose=VERBOSE,
            seed=SEED
        )
        diags = {
            "mode": "covproj",
            "history": hist,
            "meta": {k: (np.array(v).tolist() if hasattr(v, "shape") else v)
                     for k, v in meta.items() if k != "pca"},
            "pca_components": (meta["pca"].components_.tolist() if meta.get("pca") is not None else None)
        }
    else:
        print("[info] Training residual-targeted encoder (legacy).")
        model, enc, phi, targets, diags = train_residual_encoder(
            X_ts_pre=X_ts_pre,
            meal_ohe=meal_ohe,
            subj_ohe=subj_ohe,
            Z_cont=Z,                        # treatment intensity (carbs)
            mediator_scalar=mediator_scalar, # bolus_for_meal over [-2h, +60m]
            outcome_seq=y_seq,               # Δ-glucose sequence (+60 to +240)
            latent_dim=LATENT_DIM,
            n_splits=K_FOLDS,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            l2_reg=L2_REG,
            use_mmd=USE_MMD_RES,
            gamma_max=GAMMA_MAX,
            m_loss=M_LOSS,
            y_loss=Y_LOSS,
            verbose=VERBOSE,
            seed=SEED
        )
        diags["mode"] = "residual"

    # --------------------------
    # Build embeddings DataFrame
    # --------------------------
    k = phi.shape[1]
    assert phi.shape[0] == n, "phi row count mismatch"

    df = pd.DataFrame({
        "row_index_0based": np.arange(n, dtype=int),
        "global_window_id": np.asarray(global_window_id, dtype=int),
        "subject_id": np.array(subj_list),
        "meal_type":  np.array(meal_list),
        "treat_meal_carbs": Z.astype(float),
        "mediator_bolus_for_meal": mediator_scalar.astype(float),
    })
    for j in range(k):
        df[f"phi_{j+1:02d}"] = phi[:, j].astype(np.float32)
    df["A_bin"] = (Z > np.median(Z)).astype(np.float32)

    # --------------------------
    # Save outputs (same locations)
    # --------------------------
    project_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    out_dir = os.path.join(project_root, "ae_cma", "cma_cluster", "analysis_data")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    out_csv  = os.path.join(out_dir, "z_meal_y_delta_glucose_phi_embeddings.csv")
    out_json = os.path.join(out_dir, "z_meal_y_delta_glucose_phi_diagnostics.json")

    df.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(diags, f, indent=2)

    print("✓ saved embeddings + labels to", out_csv)
    print("✓ saved training diagnostics to", out_json)
    print(f"Done in {time.time()-t0:.1f}s. Embedding dim = {k}, rows = {n}")

if __name__ == "__main__":
    main()

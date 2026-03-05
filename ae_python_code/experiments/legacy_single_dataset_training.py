#!/usr/bin/env python3
"""
run_causal_ae.py
================
Train causal linear-friendly autoencoders for mediation analysis.
TensorFlow 2.x compatible with portable path configuration.

Usage:
    python run_causal_ae.py                    # Run on 2018 data (default)
    python run_causal_ae.py --data 2020        # Run on 2020 data
    python run_causal_ae.py --data combined    # Run on combined 2018+2020 data
    python run_causal_ae.py --data_dir /path/to/custom/dir  # Custom path
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# =============================================================================
# CONFIGURATION - Import centralized paths
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import CONFIG
CONFIG.ensure_dirs()


def get_meal_windows_dir(args):
    """Get the meal windows directory based on command-line arguments."""
    if args.data_dir:
        return Path(args.data_dir)
    elif args.data == "2020":
        return CONFIG.MEAL_WINDOWS_2020_DIR
    elif args.data == "combined":
        return CONFIG.MEAL_WINDOWS_COMBINED_DIR
    else:  # default to 2018
        return CONFIG.MEAL_WINDOWS_DIR


# Set up paths from config
OUTPUT_DIR = CONFIG.ANALYSIS_DATA_DIR

# =============================================================================
# MODULE IMPORTS
# =============================================================================

import resid_ae_utils as RAE
import causal_linear_ae as CLAE
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import KFold


def verify_setup(meal_windows_dir):
    """Verify all directories and files are in place"""
    print("\n" + "="*70)
    print("VERIFYING SETUP")
    print("="*70)

    checks = {
        "Meal windows directory": meal_windows_dir,
        "Output directory": OUTPUT_DIR,
    }

    all_good = True
    for name, path in checks.items():
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {name}: {path}")
        if not exists and "output" in name.lower():
            path.mkdir(parents=True, exist_ok=True)
            print(f"  → Created directory")
        elif not exists:
            all_good = False

    # Check for required Python files
    print("\nChecking Python modules:")
    py_files = ["resid_ae_utils.py", "causal_linear_ae.py"]

    for name in py_files:
        path = SCRIPT_DIR / name
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {name}")
        if not exists:
            all_good = False

    # Check for data files
    print("\nChecking data files:")
    # Check for various naming patterns
    csv_patterns = ["meal_window_*.csv", "meal_2020_window_*.csv", "meal_combined_window_*.csv"]
    csv_files = []
    for pattern in csv_patterns:
        csv_files.extend(list(meal_windows_dir.glob(pattern)))

    if csv_files:
        print(f"✓ Found {len(csv_files)} meal window CSV files")
    else:
        print("✗ No meal window CSV files found")
        print(f"  Looked for patterns: {csv_patterns}")
        all_good = False

    return all_good


def validate_for_mediation(phi, Y, A, M, n_folds=5):
    """Validate features specifically for mediation analysis"""
    results = {}
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=123)
    
    # 1. Outcome prediction (key metric)
    y_scores = []
    for train_idx, test_idx in kf.split(phi):
        for t in [11, 17, 23]:  # 60, 90, 120 minutes
            phi_train, phi_test = phi[train_idx], phi[test_idx]
            y_train, y_test = Y[train_idx, t], Y[test_idx, t]
            
            ridge = RidgeCV(alphas=np.logspace(-4, 4, 20))
            ridge.fit(phi_train, y_train)
            score = ridge.score(phi_test, y_test)
            y_scores.append(score)
    
    results['outcome_R2'] = np.mean(y_scores)
    results['outcome_R2_std'] = np.std(y_scores)
    
    # 2. Mediator prediction
    m_scores = []
    for train_idx, test_idx in kf.split(phi):
        phi_train, phi_test = phi[train_idx], phi[test_idx]
        m_train, m_test = M[train_idx], M[test_idx]
        
        ridge = RidgeCV(alphas=np.logspace(-4, 4, 20))
        ridge.fit(phi_train, m_train)
        score = ridge.score(phi_test, m_test)
        m_scores.append(score)
    
    results['mediator_R2'] = np.mean(m_scores)
    
    # 3. Check sparsity
    lasso = LassoCV(cv=5, max_iter=2000)
    lasso.fit(phi, Y[:, 11])
    results['prop_nonzero'] = np.mean(lasso.coef_ != 0)
    
    return results


def run_analysis(meal_windows_dir, features=None, seeds=None, skip_residual=False):
    """Run the complete analysis"""

    if not verify_setup(meal_windows_dir):
        print("\n❌ Setup verification failed. Please check the paths above.")
        return None, None

    if features is None:
        features = ['glucose', 'bolus', 'basal', 'meal', 'steps', 'hr']
    if seeds is None:
        seeds = [88]

    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    print(f"Data directory: {meal_windows_dir}")

    try:
        (X_ts, X_ts_pre, meal_ohe, subj_ohe, Z, Z_bin, Y_seq, M,
         global_ids, meal_list, subj_list, pre_ints, postX_ints) = RAE.load_windows(
            csv_dir=str(meal_windows_dir),
            features=features,
            treat="meal",
            interval_min=5,
            pre_minutes=120,
            post_X_minutes=60,
            post_total_minutes=240,
            standardize=True
        )
        
        print(f"✓ Loaded {X_ts_pre.shape[0]} windows")
        print(f"  Shape: X_ts_pre={X_ts_pre.shape}, Y_seq={Y_seq.shape}")
        print(f"  Subjects: {len(np.unique(subj_list))}")
        print(f"  Meal types: {np.unique(meal_list)}")
        
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        return None, None
    
    results = []
    best_phi = None
    best_model_name = None
    best_score = -np.inf
    
    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*70}")
        print(f"SEED {seed_idx + 1}/{len(seeds)}: {seed}")
        print('='*70)
        
        # RESIDUAL AE (Optional)
        if not skip_residual:
            print("\n📊 Training Residual AE...")
            try:
                model_resid, enc_resid, phi_resid, targets, diag_resid = RAE.train_residual_encoder(
                    X_ts_pre=X_ts_pre,
                    meal_ohe=meal_ohe,
                    subj_ohe=subj_ohe,
                    Z_cont=Z,
                    mediator_scalar=M,
                    outcome_seq=Y_seq,
                    latent_dim=8,
                    n_splits=5,
                    epochs=20,
                    batch_size=256,
                    lr=1e-3,
                    seed=seed,
                    verbose=1
                )
                
                val_resid = CLAE.validate_linear_friendliness(phi_resid, Y_seq, Z, M)
                val_med_resid = validate_for_mediation(phi_resid, Y_seq, Z, M)
                
                results.append({
                    'seed': seed,
                    'model': 'Residual_AE',
                    'linear_R2': val_resid['linear_R2'],
                    'linearity_ratio': val_resid['linearity_ratio'],
                    'outcome_R2': val_med_resid['outcome_R2'],
                    'mediator_R2': val_med_resid['mediator_R2'],
                })
                
                score = val_resid['linearity_ratio']
                if score > best_score:
                    best_score = score
                    best_phi = phi_resid
                    best_model_name = 'residual'
                    
            except Exception as e:
                print(f"  ⚠️ Residual AE failed: {e}")
        
        # CAUSAL LINEAR AE
        print("\n🎯 Training Causal Linear AE...")
        try:
            model_causal, enc_causal, phi_causal, hist_causal = CLAE.train_causal_linear_ae(
                X_ts_pre=X_ts_pre,
                meal_ohe=meal_ohe,
                subj_ohe=subj_ohe,
                A_cont=Z,
                M_scalar=M,
                Y_seq=Y_seq,
                latent_dim=6,
                n_basis_functions=20,
                epochs=100,
                batch_size=256,
                lr=1e-3,
                seed=88,
                verbose=1
            )
            
            val_causal = CLAE.validate_linear_friendliness(phi_causal, Y_seq, Z, M)
            val_med_causal = validate_for_mediation(phi_causal, Y_seq, Z, M)
            
            results.append({
                'seed': seed,
                'model': 'Causal_Linear_AE',
                'linear_R2': val_causal['linear_R2'],
                'linearity_ratio': val_causal['linearity_ratio'],
                'outcome_R2': val_med_causal['outcome_R2'],
                'mediator_R2': val_med_causal['mediator_R2'],
            })
            
            score = val_causal['linearity_ratio']
            if score > best_score:
                best_score = score
                best_phi = phi_causal
                best_model_name = 'causal'
                
        except Exception as e:
            print(f"  ⚠️ Causal Linear AE failed: {e}")
    
    # SAVE RESULTS
    if best_phi is not None:
        print("\n" + "="*70)
        print("SAVING RESULTS")
        print("="*70)
        
        if results:
            results_df = pd.DataFrame(results)
            metrics_path = OUTPUT_DIR / "ae_comparison_metrics.csv"
            results_df.to_csv(metrics_path, index=False)
            print(f"✓ Saved metrics to: {metrics_path}")
        
        phi_path = OUTPUT_DIR / f"z_meal_y_delta_glucose_phi_embeddings_{best_model_name}.csv"
        
        phi_df = pd.DataFrame(best_phi, columns=[f'phi_{i+1}' for i in range(best_phi.shape[1])])
        phi_df['global_window_id'] = global_ids
        phi_df['meal_type'] = meal_list
        phi_df['subject_id'] = subj_list
        phi_df['treat_meal_carbs'] = Z
        phi_df['mediator_bolus_for_meal'] = M
        
        phi_df.to_csv(phi_path, index=False)
        print(f"✓ Saved phi to: {phi_path}")
        
        # Also save as main phi file
        main_phi_path = OUTPUT_DIR / "z_meal_y_delta_glucose_phi_embeddings.csv"
        phi_df.to_csv(main_phi_path, index=False)
        
        print("\n✅ ANALYSIS COMPLETE!")
        return results_df if results else None, phi_df
    
    else:
        print("\n❌ No models trained successfully")
        return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train causal linear-friendly autoencoders for mediation analysis."
    )
    parser.add_argument(
        "--data",
        choices=["2018", "2020", "combined"],
        default="2018",
        help="Dataset to use: 2018 (default), 2020, or combined"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Custom path to meal windows directory (overrides --data)"
    )
    parser.add_argument(
        "--skip_residual",
        action="store_true",
        help="Skip training the residual AE (faster)"
    )

    args = parser.parse_args()

    # Get the appropriate meal windows directory
    meal_windows_dir = get_meal_windows_dir(args)

    CONFIG.print_config()
    print(f"\n📂 Using meal windows from: {meal_windows_dir}")

    print("\n🚀 STARTING CAUSAL AE ANALYSIS")
    results, phi = run_analysis(
        meal_windows_dir=meal_windows_dir,
        features=['glucose', 'steps', 'basal', 'meal', 'heart', 'bolus'],
        seeds=[88],
        skip_residual=args.skip_residual
    )

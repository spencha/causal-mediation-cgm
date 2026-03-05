# resid_ae_utils.py
import os, glob, re
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

from tensorflow.keras.layers import (
    Layer, Input, Masking, LSTM, Dense, RepeatVector, TimeDistributed,
    Concatenate, GaussianNoise, BatchNormalization, Dropout, LayerNormalization
)
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import Callback

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA

# ===============================================================
# 1) Data loader
# ===============================================================

def _max_consecutive_na(series: pd.Series) -> int:
    """
    Compute the maximum number of consecutive NaN values in a series.

    Parameters:
    -----------
    series : pd.Series
        Input series (can contain NaN values)

    Returns:
    --------
    int : Maximum consecutive NaN count (0 if no NaNs)
    """
    is_na = series.isna()
    if not is_na.any():
        return 0

    # Group consecutive values and find max length of NA groups
    # Create group IDs by cumsum of changes
    groups = (~is_na).cumsum()
    # Count NAs in each group where is_na is True
    na_groups = is_na.groupby(groups).sum()
    return int(na_groups.max()) if len(na_groups) > 0 else 0


def load_windows(
    csv_dir,
    features,
    treat: str = "meal",
    interval_min=5,
    pre_minutes=120,
    post_X_minutes=60,
    post_total_minutes=240,
    standardize=True,
    y_scalar_fn="mean",  # kept for API compatibility; not used
    return_in_range=False,  # if True, also return in_range array
    max_consecutive_na=None,  # max consecutive missing glucose allowed (None=disabled, filtering done in R)
):
    global_id_list = []
    if treat not in ("meal", "bolus"):
        raise ValueError("`treat` must be either 'meal' or 'bolus'")

    # Support multiple naming patterns for different dataset years
    patterns = [
        f"{treat}_window_*.csv",           # 2018 format: meal_window_*.csv
        f"{treat}_2020_window_*.csv",      # 2020 format: meal_2020_window_*.csv
        f"{treat}_combined_window_*.csv",  # Combined format: meal_combined_window_*.csv
    ]
    base = Path(csv_dir)
    candidates = []
    if base.is_absolute():
        candidates.append(base)
    else:
        here = Path(__file__).resolve().parent
        candidates += [here / base, here.parent / base, here / "meal_windows_2018"]

    files = []
    matched_pattern = None
    for c in candidates:
        for pattern in patterns:
            files = sorted((c).glob(pattern))
            if files:
                csv_dir = str(c)
                matched_pattern = pattern
                break
        if files:
            break
    if not files:
        raise FileNotFoundError(
            f"No CSVs matched patterns {patterns}.\nTried: {', '.join(str(c) for c in candidates)}"
        )
    print(f"  Found {len(files)} files matching pattern: {matched_pattern}")

    pre_ints        = pre_minutes        // interval_min
    post_X_ints     = post_X_minutes     // interval_min
    post_total_ints = post_total_minutes // interval_min
    need_len        = pre_ints + post_total_ints

    X_list, X_preonly_list, y_list = [], [], []
    meal_list, subj_list, Z_list   = [], [], []
    mediator_scalar_list           = []
    total_bolus_list               = []
    in_range_list                  = []  # for in_range column if available
    n_skipped_na = 0  # Track windows skipped due to missing glucose

    for path in files:
        df = pd.read_csv(path)
        if len(df) < need_len:
            continue

        # Check for excessive consecutive missing glucose values
        # This filters out windows where glucose trajectory is mostly imputed
        if max_consecutive_na is not None and max_consecutive_na > 0:
            glucose_raw = pd.to_numeric(df["glucose"], errors="coerce")
            # Check the full window (pre + post) for consecutive NAs
            glucose_window = glucose_raw.iloc[:need_len]
            max_consec = _max_consecutive_na(glucose_window)
            if max_consec > max_consecutive_na:
                n_skipped_na += 1
                continue

        gid = None
        if "global_window_id" in df.columns:
            try: gid = int(df["global_window_id"].iloc[0])
            except Exception: gid = None
        if gid is None and "window_id" in df.columns:
            try: gid = int(df["window_id"].iloc[0])
            except Exception: gid = None
        if gid is None:
            m = re.search(r"(\d+)(?=\D*$)", str(path))
            if m:
                try: gid = int(m.group(1))
                except Exception: gid = None
        global_id_list.append(gid)

        Xf = (
            df[features]
            .interpolate(method="linear", limit_direction="both")
            .bfill().ffill()
            .iloc[: pre_ints + post_X_ints]
            .astype(float)
            .fillna(0.0)
        )
        Xf_pre = Xf.copy()
        if post_X_ints > 0:
            Xf_pre.iloc[pre_ints:] = 0.0

        X_list.append(Xf.values)
        X_preonly_list.append(Xf_pre.values)

        g = pd.to_numeric(df["glucose"], errors="coerce")
        upto = pre_ints + post_total_ints
        g.iloc[:upto] = g.iloc[:upto].interpolate(method="linear", limit_direction="both").bfill().ffill()
        # Use glucose at meal time as baseline (not 55 min post-meal)
        base_idx     = pre_ints  # meal time index
        base_glucose = float(g.iloc[base_idx])
        y_raw        = g.iloc[pre_ints + post_X_ints : pre_ints + post_total_ints].to_numpy()
        y_seq_delta  = (y_raw - base_glucose).astype(float)
        y_list.append(y_seq_delta)

        centre_row = pre_ints
        meal_list.append(df["meal_type"].iloc[centre_row])
        subj_list.append(df["subject_id"].iloc[centre_row])

        if treat == "meal":
            Z_list.append(df["meal_at_time_0"].iloc[centre_row])
        else:
            Z_list.append(df["bolus_taken"].iloc[centre_row])

        if "bolus" not in df.columns:
            raise KeyError("Expected 'bolus' column to compute bolus_for_meal.")
        X_len = pre_ints + post_X_ints
        bolus_for_meal = (
            pd.to_numeric(df["bolus"], errors="coerce").iloc[:X_len].fillna(0.0).sum()
        )
        mediator_scalar_list.append(float(bolus_for_meal))

        # Compute total bolus across the FULL observation window (-120 to +240 min)
        # This is broader than bolus_for_meal (-120 to +60 min) and captures late boluses.
        # Used to identify truly unbolused meals (where both mediator AND total are zero).
        total_bolus = (
            pd.to_numeric(df["bolus"], errors="coerce").fillna(0.0).sum()
        )
        total_bolus_list.append(float(total_bolus))

        # Extract in_range if available (for AUC calculations)
        if return_in_range:
            if "in_range" in df.columns:
                # Get in_range for the same range as y_seq
                ir = pd.to_numeric(df["in_range"], errors="coerce")
                ir_seq = ir.iloc[pre_ints + post_X_ints : pre_ints + post_total_ints].to_numpy()
                in_range_list.append(ir_seq.astype(float))
            else:
                # Compute from glucose: in_range = (glucose >= 70) & (glucose <= 140)
                g_abs = pd.to_numeric(df["glucose"], errors="coerce")
                g_abs = g_abs.interpolate(method="linear", limit_direction="both").bfill().ffill()
                g_seq = g_abs.iloc[pre_ints + post_X_ints : pre_ints + post_total_ints].to_numpy()
                ir_seq = ((g_seq >= 70) & (g_seq <= 140)).astype(float)
                in_range_list.append(ir_seq)

    if len(X_list) == 0:
        raise RuntimeError(f"Found files in {csv_dir} but none had the required length ({need_len} rows).")

    # Report filtering stats
    if n_skipped_na > 0:
        print(f"  Skipped {n_skipped_na} windows due to >{max_consecutive_na} consecutive missing glucose values")
    print(f"  Loaded {len(X_list)} windows after filtering")

    if not (len(X_preonly_list) == len(y_list) == len(meal_list) == len(subj_list) == len(Z_list) == len(global_id_list)):
        raise RuntimeError(
            f"List-length mismatch: X={len(X_list)}, X_pre={len(X_preonly_list)}, "
            f"y={len(y_list)}, meal={len(meal_list)}, subj={len(subj_list)}, "
            f"Z={len(Z_list)}, gid={len(global_id_list)}"
        )

    X_ts     = np.stack(X_list, axis=0)
    X_ts_pre = np.stack(X_preonly_list, axis=0)
    y_seq    = np.stack(y_list, axis=0)

    if standardize:
        mu = X_ts.mean(axis=(0,1), keepdims=True)
        sd = X_ts.std (axis=(0,1), keepdims=True) + 1e-8
        X_ts     = (X_ts - mu) / sd
        X_ts_pre = (X_ts_pre - mu) / sd

    meal_ohe = OneHotEncoder(sparse_output=False).fit_transform(np.array(meal_list).reshape(-1,1))
    subj_ohe = OneHotEncoder(sparse_output=False).fit_transform(np.array(subj_list).reshape(-1,1))

    Z     = np.array(Z_list, dtype=float)
    Z_bin = (Z > np.median(Z)).astype(np.float32)
    mediator_scalar = np.array(mediator_scalar_list, dtype=float)
    total_bolus_arr = np.array(total_bolus_list, dtype=float)

    global_window_id = np.array(global_id_list, dtype=float)
    if np.any(np.isnan(global_window_id)):
        n = len(global_window_id)
        print("[warn] Some global window ids missing; falling back to 1..n for those rows.")
        filler = np.arange(1, n+1, dtype=float)
        mask = np.isnan(global_window_id)
        global_window_id[mask] = filler[mask]
    global_window_id = global_window_id.astype(int)

    result = (X_ts, X_ts_pre, meal_ohe, subj_ohe, Z, Z_bin,
              y_seq, mediator_scalar, global_window_id,
              meal_list, subj_list, pre_ints, post_X_ints,
              total_bolus_arr)

    if return_in_range and in_range_list:
        in_range_seq = np.stack(in_range_list, axis=0)
        return result + (in_range_seq,)

    return result

# ===============================================================
# 2) Regularizers (unchanged)
# ===============================================================
class MMDPenalty(Layer):
    def __init__(self, gamma=1.0, **kwargs):
        super().__init__(**kwargs); self.gamma = gamma
    def call(self, inputs):
        latent, zbin = inputs
        z = tf.reshape(tf.cast(zbin, tf.float32), (-1,1))
        sum_t = tf.reduce_sum(latent * z, axis=0)
        sum_c = tf.reduce_sum(latent * (1-z), axis=0)
        n_t   = tf.reduce_sum(z)   + 1e-6
        n_c   = tf.reduce_sum(1-z) + 1e-6
        mu_t  = sum_t / n_t
        mu_c  = sum_c / n_c
        mmd   = tf.reduce_sum(tf.square(mu_t - mu_c))
        self.add_loss(self.gamma * mmd)
        return mmd

class GammaScheduler(tf.keras.callbacks.Callback):
    def __init__(self, gamma_max, ramp_epochs): super().__init__(); self.gamma_max=gamma_max; self.ramp_epochs=ramp_epochs
    def on_epoch_begin(self, epoch, logs=None):
        frac = min(epoch / self.ramp_epochs, 1.0)
        new_g = self.gamma_max * frac
        for layer in self.model.layers:
            if isinstance(layer, MMDPenalty):
                layer.gamma = new_g; break

class DecorrPenalty(Layer):
    def __init__(self, lam=1e-3, var_w=1.0, cov_w=1.0, mean_w=0.1, eps=1e-5, **kwargs):
        super().__init__(**kwargs); self.lam=lam; self.var_w=var_w; self.cov_w=cov_w; self.mean_w=mean_w; self.eps=eps
    def call(self, phi):
        phi = tf.cast(phi, tf.float32)
        mu  = tf.reduce_mean(phi, axis=0, keepdims=True)
        phi_c = phi - mu
        std = tf.math.reduce_std(phi_c, axis=0) + self.eps
        var_term = tf.reduce_mean(tf.square(std - 1.0))
        z = phi_c / std
        n = tf.cast(tf.shape(z)[0], tf.float32)
        c = tf.matmul(z, z, transpose_a=True) / (n - 1.0 + self.eps)
        diag = tf.linalg.diag_part(c)
        c_off = c - tf.linalg.diag(diag)
        cov_term = tf.reduce_mean(tf.square(c_off))
        mean_term = tf.reduce_mean(tf.square(mu))
        loss = self.lam * (self.var_w*var_term + self.cov_w*cov_term + self.mean_w*mean_term)
        self.add_loss(loss)
        return phi

# ===============================================================
# 3) Residual-targeted encoder (kept) + utilities
# ===============================================================
def pinball_loss(tau):
    def _loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(tau*e, (tau-1)*e))
    return _loss

def build_residual_encoder(
    T, p, n_meals, n_subs, latent_dim,
    l2_reg=1e-4, use_mmd=True, gamma=1e-3,
    rm_outdim=1, ry_outdim=36,
    use_decorr=True, decorr_lambda=1e-3
):
    inp_ts    = Input((T,p),      name="ts_input")
    inp_meal  = Input((n_meals,), name="meal_input")
    inp_subj  = Input((n_subs,),  name="subj_input")
    inp_zbin  = Input((),         name="zbin_input")

    meal_emb   = Dense(8, activation="relu")(inp_meal)
    subj_emb   = Dense(8, activation="relu")(inp_subj)
    meal_emb_t = RepeatVector(T)(meal_emb)
    subj_emb_t = RepeatVector(T)(subj_emb)

    x = Concatenate()([inp_ts, meal_emb_t, subj_emb_t])
    x = GaussianNoise(0.2)(x)
    x = Masking(mask_value=0.0)(x)
    x = LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_reg))(x)
    x = LSTM(32, kernel_regularizer=l2(l2_reg))(x)

    phi_core = Dense(latent_dim, kernel_regularizer=l2(l2_reg), name="phi_prebn")(x)
    phi_core = BatchNormalization(center=True, scale=False, name="phi_bn")(phi_core)
    if use_decorr:
        phi_core = DecorrPenalty(lam=decorr_lambda, name="phi_decorr")(phi_core)
    phi_core = Dropout(0.2, name="phi_do")(phi_core)
    phi_out = Dense(latent_dim, activation=None, name="phi",
                    kernel_regularizer=l2(l2_reg),
                    activity_regularizer=regularizers.l2(1e-6))(phi_core)
    phi_out = LayerNormalization(center=True, scale=True, name="phi_ln")(phi_out)

    if use_mmd:
        _ = MMDPenalty(gamma=gamma, name="mmd_pen")([phi_out, inp_zbin])

    Rm = Dense(32, activation="relu")(phi_out)
    Rm = Dense(rm_outdim, name="Rm")(Rm)

    Ry = Dense(64, activation="relu")(phi_out)
    Ry = Dense(ry_outdim, name="Ry")(Ry)

    Al = Dense(32, activation="relu")(phi_out)
    Al = Dense(1, activation=None, name="Alogit")(Al)

    model = Model([inp_ts, inp_meal, inp_subj, inp_zbin], [Rm, Ry, Al])
    return model

def get_encoder(model):
    return Model(
        inputs=[model.input[0], model.input[1], model.input[2]],
        outputs=model.get_layer("phi").output
    )

def _flatten_H(X_ts_pre, meal_ohe, subj_ohe):
    n, T, p = X_ts_pre.shape
    H = X_ts_pre.reshape(n, T*p)
    H = np.concatenate([H, meal_ohe, subj_ohe], axis=1)
    return H

def _fit_predict_oof_seq(H, A, M_scalar, Y_seq, n_splits=5, random_state=123):
    n, Hlen = Y_seq.shape
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    m_learner = lambda: Pipeline([
        ("sc", StandardScaler(with_mean=False)),
        ("hgb", HistGradientBoostingRegressor(loss="squared_error"))
    ])
    y_base = lambda: MultiOutputRegressor(HistGradientBoostingRegressor(loss="squared_error"))
    a_learner = lambda: LogisticRegression(max_iter=1000)

    m_hat = np.zeros(n, dtype=float)
    y_hat = np.zeros((n, Hlen), dtype=float)
    e_hat = np.zeros(n, dtype=float)
    A_bin = (A > np.median(A)).astype(int)

    for tr, ho in kf.split(H):
        H_tr, H_ho = H[tr], H[ho]
        A_tr, A_ho = A[tr], A[ho]
        M_tr, M_ho = M_scalar[tr], M_scalar[ho]
        Y_tr, Y_ho = Y_seq[tr], Y_seq[ho]
        Ab_tr, Ab_ho = A_bin[tr], A_bin[ho]

        Xm_tr = np.column_stack([A_tr, H_tr]); Xm_ho = np.column_stack([A_ho, H_ho])
        mdl_m = m_learner().fit(Xm_tr, M_tr); m_hat[ho] = mdl_m.predict(Xm_ho)

        Xy_tr = np.column_stack([A_tr, M_tr, H_tr]); Xy_ho = np.column_stack([A_ho, M_ho, H_ho])
        mdl_y = y_base().fit(Xy_tr, Y_tr); y_hat[ho, :] = mdl_y.predict(Xy_ho)

        mdl_a = a_learner().fit(H_tr, Ab_tr); e_hat[ho] = mdl_a.predict_proba(H_ho)[:, 1]

    Rm = (M_scalar - m_hat).astype(np.float32)
    Ry = (Y_seq - y_hat).astype(np.float32)
    return Rm, Ry, A_bin.astype(np.float32), e_hat

# ===============================================================
# 4) Residual AE training (kept)
# ===============================================================
def train_residual_encoder(
    X_ts_pre, meal_ohe, subj_ohe,
    Z_cont, mediator_scalar, outcome_seq,
    latent_dim=16, n_splits=5, epochs=30, batch_size=256, lr=1e-3,
    l2_reg=1e-4, use_mmd=True, gamma_max=1e-3,
    m_loss="mse", y_loss="mse", verbose=2, seed=123
):
    tf.keras.utils.set_random_seed(seed)
    n, T, p = X_ts_pre.shape
    n_meals = meal_ohe.shape[1]; n_subs = subj_ohe.shape[1]; Hlen = outcome_seq.shape[1]

    Hmat = _flatten_H(X_ts_pre, meal_ohe, subj_ohe)
    Rm, Ry, A_bin, e_hat = _fit_predict_oof_seq(Hmat, Z_cont, mediator_scalar, outcome_seq,
                                                n_splits=n_splits, random_state=seed)

    rm_mu, rm_sd = float(np.mean(Rm)), float(np.std(Rm) + 1e-8)
    Rm_s = ((Rm - rm_mu) / rm_sd).astype(np.float32).reshape(-1, 1)
    ry_mu = np.mean(Ry, axis=0, keepdims=True); ry_sd = np.std(Ry, axis=0, keepdims=True) + 1e-8
    Ry_s  = ((Ry - ry_mu) / ry_sd).astype(np.float32)

    model = build_residual_encoder(T, p, n_meals, n_subs, latent_dim,
                                   l2_reg=l2_reg, use_mmd=use_mmd, gamma=(0.0 if not use_mmd else 1e-10),
                                   rm_outdim=1, ry_outdim=Hlen)

    if m_loss == "mse": loss_m = "mse"
    elif isinstance(m_loss, tuple) and m_loss[0] == "pinball": loss_m = pinball_loss(float(m_loss[1]))
    else: loss_m = "mse"
    if y_loss == "mse": loss_y = "mse"
    elif isinstance(y_loss, tuple) and y_loss[0] == "pinball": loss_y = pinball_loss(float(y_loss[1]))
    else: loss_y = "mse"

    try:
        opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-5, clipnorm=1.0)
    except AttributeError:
        opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)

    model.compile(optimizer=opt,
                  loss={"Rm": loss_m, "Ry": loss_y, "Alogit": tf.keras.losses.BinaryCrossentropy(from_logits=True)},
                  loss_weights={"Rm": 1.0, "Ry": 1.0, "Alogit": 0.2})

    inputs  = [X_ts_pre.astype("float32"), meal_ohe.astype("float32"),
               subj_ohe.astype("float32"), A_bin.astype("float32")]
    targets = [Rm_s.astype("float32"), Ry_s.astype("float32"), A_bin.reshape(-1,1).astype("float32")]

    cbs = []
    if use_mmd:
        cbs.append(GammaScheduler(gamma_max=gamma_max, ramp_epochs=max(1, epochs//3)))

    hist = model.fit(inputs, targets, epochs=epochs, batch_size=batch_size,
                     validation_split=0.2, verbose=verbose, callbacks=cbs, shuffle=True)

    enc = get_encoder(model)
    phi = enc.predict([X_ts_pre, meal_ohe, subj_ohe], verbose=0)

    diagnostics = {
        "train_history": {k: [float(x) for x in v] for k, v in hist.history.items()},
        "target_scalers": {"Rm": [rm_mu, rm_sd],
                           "Ry_mu": ry_mu.flatten().astype(float).tolist(),
                           "Ry_sd": ry_sd.flatten().astype(float).tolist()}
    }
    return model, enc, phi, {"Rm": Rm, "Ry": Ry}, diagnostics

# ===============================================================
# 5) NEW — Build pre-treatment covariate targets C
# ===============================================================
def _mad(x, axis=None):
    med = np.nanmedian(x, axis=axis, keepdims=True)
    return np.nanmedian(np.abs(x - med), axis=axis)

def _acf1_2(x):
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    if np.allclose(x, 0): return 0.0, 0.0
    r = np.correlate(x, x, mode="full")
    mid = len(r)//2
    denom = r[mid] if r[mid] != 0 else 1.0
    r1 = r[mid+1] / denom if mid+1 < len(r) else 0.0
    r2 = r[mid+2] / denom if mid+2 < len(r) else 0.0
    return float(r1), float(r2)

def _slope_last_k(x, k):
    x = np.asarray(x, dtype=float)
    k = min(k, len(x))
    y = x[-k:]
    t = np.arange(k, dtype=float)
    t = t - t.mean()
    denom = np.sum(t**2) + 1e-8
    slope = np.dot(t, y - y.mean()) / denom
    return float(slope)

def build_pre_covariates(
    X_ts_pre,                # (n, T, p) standardized (ok)
    meal_ohe, subj_ohe,      # one-hots
    pre_len,                 # number of pre steps (e.g., 24 for 2h @5min)
    feat_index: dict,        # {"glucose": idx, "bolus": idx (optional), "basal": idx (optional)}
    k_for_slope=12,          # last 60 minutes if 5-min bins
    add_flatten_pca=False,
    flatten_pca_components=32,
    pca_random_state=123
):
    """
    Returns:
      C: (n, C_dim)
      meta: dict with scaler (mean, std), and optional PCA object
    """
    n, T, p = X_ts_pre.shape
    assert pre_len <= T, "pre_len must be <= T"

    gi = feat_index.get("glucose", None)
    if gi is None:
        raise ValueError("feat_index must include 'glucose' column index")

    # PRE slices only
    Gpre = X_ts_pre[:, :pre_len, gi]  # (n, pre_len)

    # Basic stats
    level   = np.nanmean(Gpre, axis=1)
    slope   = np.array([_slope_last_k(row, k_for_slope) for row in Gpre])
    sd_pre  = np.nanstd(Gpre, axis=1)
    mad_pre = np.array([_mad(row) for row in Gpre])
    acf1    = np.zeros(n); acf2 = np.zeros(n)
    for i in range(n):
        acf1[i], acf2[i] = _acf1_2(Gpre[i])

    # Bolus/Basal sums (optional)
    bolus_sum = None; basal_sum = None
    if "bolus" in feat_index and feat_index["bolus"] is not None:
        bi = feat_index["bolus"]; bolus_sum = np.nansum(X_ts_pre[:, :pre_len, bi], axis=1)
    if "basal" in feat_index and feat_index["basal"] is not None:
        bsi = feat_index["basal"]; basal_sum = np.nansum(X_ts_pre[:, :pre_len, bsi], axis=1)

    cols = [
        level, slope, sd_pre, mad_pre, acf1, acf2,
    ]
    colnames = ["g_level", "g_slope60", "g_sd", "g_mad", "g_acf1", "g_acf2"]

    if bolus_sum is not None:
        cols.append(bolus_sum); colnames.append("bolus_sum_pre")
    if basal_sum is not None:
        cols.append(basal_sum); colnames.append("basal_sum_pre")

    # One-hots (include to force linear recoverability)
    C_core = np.column_stack(cols).astype(np.float32)
    C_ohe  = np.concatenate([meal_ohe, subj_ohe], axis=1).astype(np.float32)
    C = np.concatenate([C_core, C_ohe], axis=1)

    # Optional flattened-pre PCA
    pca_obj = None
    if add_flatten_pca:
        Xflat_pre = X_ts_pre[:, :pre_len, :].reshape(n, pre_len*p)
        pca_obj = PCA(n_components=flatten_pca_components, random_state=pca_random_state)
        Xflat_pca = pca_obj.fit_transform(Xflat_pre).astype(np.float32)
        C = np.concatenate([C, Xflat_pca], axis=1)

    # Standardize C columns
    C_mu = C.mean(axis=0, keepdims=True)
    C_sd = C.std(axis=0, keepdims=True) + 1e-8
    C_s  = (C - C_mu) / C_sd

    meta = {"C_mu": C_mu.astype(np.float32), "C_sd": C_sd.astype(np.float32), "pca": pca_obj,
            "colnames_core": colnames, "meal_ohe_dim": meal_ohe.shape[1], "subj_ohe_dim": subj_ohe.shape[1]}
    return C_s.astype(np.float32), meta

# ===============================================================
# 6) NEW — Covariate-projection encoder (linear head)
# ===============================================================
def build_covproj_encoder(
    T, p, n_meals, n_subs, latent_dim, C_dim,
    l2_reg=1e-4, use_mmd=False, mmd_gamma=0.0,
    use_decorr=True, decorr_lambda=5e-4,
    add_aux_y2=True, y2_dim=2
):
    inp_ts    = Input((T,p),      name="ts_input")
    inp_meal  = Input((n_meals,), name="meal_input")
    inp_subj  = Input((n_subs,),  name="subj_input")
    inp_zbin  = Input((),         name="zbin_input")  # still available for MMD if desired

    meal_emb   = Dense(8, activation="relu")(inp_meal)
    subj_emb   = Dense(8, activation="relu")(inp_subj)
    meal_emb_t = RepeatVector(T)(meal_emb)
    subj_emb_t = RepeatVector(T)(subj_emb)

    x = Concatenate()([inp_ts, meal_emb_t, subj_emb_t])
    x = GaussianNoise(0.1)(x)
    x = Masking(mask_value=0.0)(x)
    x = LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_reg))(x)
    x = LSTM(32, kernel_regularizer=l2(l2_reg))(x)

    phi = Dense(latent_dim, kernel_regularizer=l2(l2_reg), name="phi_prebn")(x)
    phi = BatchNormalization(center=True, scale=False, name="phi_bn")(phi)
    if use_decorr:
        phi = DecorrPenalty(lam=decorr_lambda, name="phi_decorr")(phi)
    phi = Dropout(0.2, name="phi_do")(phi)
    phi = Dense(latent_dim, activation=None, name="phi",
                kernel_regularizer=l2(l2_reg),
                activity_regularizer=regularizers.l2(1e-6))(phi)
    phi = LayerNormalization(center=True, scale=True, name="phi_ln")(phi)

    if use_mmd and mmd_gamma > 0:
        _ = MMDPenalty(gamma=mmd_gamma, name="mmd_pen")([phi, inp_zbin])

    # --- Linear projection to C (no nonlinearity)
    C_lin = Dense(C_dim, activation=None, name="C_lin")(phi)

    outputs = [C_lin]
    if add_aux_y2:
        Y2_aux = Dense(y2_dim, activation=None, name="Y2_aux")(phi)
        outputs.append(Y2_aux)

    model = Model([inp_ts, inp_meal, inp_subj, inp_zbin], outputs)
    return model

# ===============================================================
# 7) NEW — Train covariate-projection AE
# ===============================================================
def train_covproj_encoder(
    X_ts_pre, meal_ohe, subj_ohe,
    outcome_seq,            # for optional Y2 aux (+60, +65)
    pre_len,                # number of pre steps
    feat_index,             # {"glucose": idx, "bolus": idx?, "basal": idx?}
    latent_dim=32,
    add_flatten_pca=False,
    flatten_pca_components=32,
    add_aux_y2=True,
    y2_indices=(0,1),       # which ΔG steps to predict as tiny aux (k=0,1 => +60,+65)
    epochs=40, batch_size=256, lr=1e-3,
    l2_reg=1e-4, use_mmd=False, mmd_gamma_max=0.0,
    verbose=2, seed=123
):
    """
    Returns:
      model, encoder, phi, {"C_mu","C_sd","pca","colnames_core", ...}, history
    """
    tf.keras.utils.set_random_seed(seed)
    n, T, p = X_ts_pre.shape
    n_meals = meal_ohe.shape[1]; n_subs = subj_ohe.shape[1]

    # --- Build C targets (standardized)
    C_s, meta = build_pre_covariates(
        X_ts_pre=X_ts_pre, meal_ohe=meal_ohe, subj_ohe=subj_ohe,
        pre_len=pre_len, feat_index=feat_index,
        add_flatten_pca=add_flatten_pca, flatten_pca_components=flatten_pca_components
    )
    C_dim = C_s.shape[1]

    # --- Optional tiny aux: first two ΔG steps (standardize per-col)
    targets = {"C_lin": C_s.astype("float32")}
    loss_dict = {"C_lin": "mse"}
    loss_wts  = {"C_lin": 1.0}

    if add_aux_y2:
        y2 = outcome_seq[:, list(y2_indices)]
        y2_mu = np.mean(y2, axis=0, keepdims=True); y2_sd = np.std(y2, axis=0, keepdims=True) + 1e-8
        y2_s  = (y2 - y2_mu) / y2_sd
        targets["Y2_aux"] = y2_s.astype("float32")
        loss_dict["Y2_aux"] = "mse"
        loss_wts["Y2_aux"]  = 0.1
        meta["Y2_mu"] = y2_mu.astype(np.float32); meta["Y2_sd"] = y2_sd.astype(np.float32)

    # --- Build model
    model = build_covproj_encoder(
        T=T, p=p, n_meals=n_meals, n_subs=n_subs, latent_dim=latent_dim, C_dim=C_dim,
        l2_reg=l2_reg, use_mmd=use_mmd, mmd_gamma=1e-10 if use_mmd else 0.0,
        use_decorr=True, decorr_lambda=5e-4,
        add_aux_y2=add_aux_y2, y2_dim=(targets["Y2_aux"].shape[1] if add_aux_y2 else 0)
    )

    try:
        opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-5, clipnorm=1.0)
    except AttributeError:
        opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)

    model.compile(optimizer=opt, loss=loss_dict, loss_weights=loss_wts)

    inputs = [X_ts_pre.astype("float32"),
              meal_ohe.astype("float32"),
              subj_ohe.astype("float32"),
              np.zeros((n,), dtype="float32")]  # zbin placeholder (unused unless MMD)

    cbs = []
    if use_mmd and mmd_gamma_max > 0:
        cbs.append(GammaScheduler(gamma_max=mmd_gamma_max, ramp_epochs=max(1, epochs//3)))

    hist = model.fit(inputs, targets, epochs=epochs, batch_size=batch_size,
                     validation_split=0.2, verbose=verbose, callbacks=cbs, shuffle=True)

    # --- Encoder & embeddings
    enc = get_encoder(model)
    phi = enc.predict([X_ts_pre, meal_ohe, subj_ohe], verbose=0)

    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    return model, enc, phi, meta, history

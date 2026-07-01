"""DiaTrend loader: episodes -> tensors in the load_windows() shape.

This is the DiaTrend analog of resid_ae_utils.load_windows(). It walks
a directory of subject ``.xlsx`` workbooks, parses each, builds
meal-centered episodes via the shared episode builder, and stacks the
result into the same 14-tuple that the OhioT1DM training driver
already consumes. The training driver dispatches on ``--dataset`` and
calls one or the other; no architecture-level code changes.

Default ``features=("glucose",)`` matches the Section 8.2 univariate
primary CLAE input. Pass ``features=("glucose", "meal", "bolus")`` to
build the (T, 3) input tensor for the Section 8.6 multivariate
sensitivity arm; the carb and bolus channels are event-driven spikes
on a zero baseline, placed in the 5-min bin containing each event.

The returned ``std_params`` dict carries the same keys as the OhioT1DM
driver's (mu, sd, treatment_median, pre_ints, meal_encoder,
subj_encoder, glucose_at_meal_raw) plus a DiaTrend-specific
``iob_at_meal`` array. IOB comes from pump ``insulinOnBoard`` for
cohort 2; for cohort 1 it is NaN unless ``bob_params`` is supplied,
in which case the kernel-derived BOB (Section 8.5) fills cohort 1.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_processing.diatrend.bob_kernel import (  # noqa: E402
    IOBKernelParams,
    bob_at_meal_times,
)
from data_processing.diatrend.episode_builder import (  # noqa: E402
    Episode,
    EpisodeConfig,
    build_episodes,
)
from data_processing.diatrend.parser import (  # noqa: E402
    DiaTrendParseError,
    parse_subject,
)


DEFAULT_FEATURES: tuple[str, ...] = ("glucose",)


def _episode_to_tensor(
    episode: Episode,
    features: tuple[str, ...],
    *,
    pre_ints: int,
    post_X_ints: int,
    bolus_history: pd.DataFrame | None,
) -> np.ndarray:
    """Stack episode signals into a (pre_ints + post_X_ints, n_features) tensor.

    The post-X portion preserves the CGM signal (matches load_windows'
    X_ts semantics); the loader produces X_ts_pre by zeroing that
    portion separately.
    """
    n_bins = pre_ints + post_X_ints
    grid = episode.cgm_grid[:n_bins]
    rel = episode.cgm_times_relative_min[:n_bins]
    if rel.size < 2:
        raise ValueError("episode has fewer than 2 CGM bins; cannot infer interval")
    interval_min = float(rel[1] - rel[0])
    out = np.zeros((n_bins, len(features)), dtype=float)
    for i, name in enumerate(features):
        if name == "glucose":
            # The episode builder preserves NaN bins (its job is to report
            # true missingness for diagnostics); the CLAE training expects
            # numeric input, so interpolate here to match the OhioT1DM
            # loader's behaviour (resid_ae_utils.load_windows lines
            # ~144-151): linear both-direction interpolation, then bfill
            # + ffill, then zero-fill as a last resort.
            series = (
                pd.Series(grid)
                .interpolate(method="linear", limit_direction="both")
                .bfill()
                .ffill()
                .fillna(0.0)
            )
            out[:, i] = series.to_numpy(dtype=float)
        elif name == "meal":
            zero_idx = pre_ints
            if 0 <= zero_idx < n_bins:
                out[zero_idx, i] = float(episode.treatment_carbs)
        elif name == "bolus":
            if bolus_history is None or bolus_history.empty:
                continue
            meal_t = episode.meal_time
            for j in range(pre_ints):
                t_rel = float(rel[j])
                bin_start = meal_t + pd.Timedelta(minutes=t_rel)
                bin_end = bin_start + pd.Timedelta(minutes=interval_min)
                in_bin = bolus_history.loc[
                    (bolus_history["date"] >= bin_start)
                    & (bolus_history["date"] < bin_end),
                    "normal",
                ]
                if not in_bin.empty:
                    out[j, i] = float(in_bin.fillna(0.0).sum())
        else:
            raise ValueError(f"unknown DiaTrend CLAE feature: {name!r}")
    return out


def _within_subject_temporal_split(
    subj_list: list[str], meal_times: list, test_frac: float
) -> np.ndarray:
    """Label each episode ``"train"`` or ``"test"`` via a per-subject
    temporal cutoff: the latest ``test_frac`` of a subject's meals (by
    absolute ``meal_time``) become ``"test"``, the earlier remainder
    ``"train"``.

    This reproduces OhioT1DM's design, where each subject contributes an
    earlier training period and a later testing period. Sorting on the
    absolute timestamp (not file order) keeps the split correct even for
    the reverse-chronological cohort-1 workbooks. A subject with a single
    retained episode is kept entirely in train, so every subject that can
    appear in test also appears in train and its random intercept stays
    estimable.
    """
    n_total = len(subj_list)
    split = np.full(n_total, "train", dtype=object)
    subj_arr = np.asarray(subj_list)
    # Chronological order across all episodes; stable so ties keep input order.
    order = np.argsort([t.value for t in meal_times], kind="stable")
    for subj in np.unique(subj_arr):
        subj_idx = order[subj_arr[order] == subj]  # this subject, oldest -> newest
        n = subj_idx.size
        if n < 2:
            continue
        n_test = min(max(int(round(test_frac * n)), 0), n - 1)
        if n_test > 0:
            split[subj_idx[-n_test:]] = "test"
    return split


def load_diatrend_data(
    raw_dir: str | Path,
    *,
    features: tuple[str, ...] | list[str] = DEFAULT_FEATURES,
    interval_min: int = 5,
    pre_minutes: int = 120,
    post_X_minutes: int = 60,
    post_total_minutes: int = 240,
    standardize: bool = True,
    episode_config: EpisodeConfig | None = None,
    bob_params: IOBKernelParams | None = None,
    cohorts: Iterable[int] | None = None,
    test_frac: float = 0.0,
) -> tuple[tuple, np.ndarray, dict]:
    """Build DiaTrend episode tensors. Returns (data_tuple, cohort_labels, std_params).

    ``data_tuple`` mirrors the 14-element output of
    ``resid_ae_utils.load_windows()`` so the OhioT1DM training driver
    can consume it without modification.

    ``test_frac`` enables a within-subject temporal train/test split that
    mirrors OhioT1DM's study-defined split (each subject's later
    monitoring period is held out as test). When ``test_frac > 0`` the
    latest ``test_frac`` of each subject's meals (by absolute meal time)
    are labelled ``"test"`` in ``std_params["split_labels"]`` and the
    standardization statistics are computed on the train rows only — so
    the encoder never sees test-period data. ``test_frac == 0`` (the
    default) labels every episode ``"all"`` and standardizes over the
    full sample, preserving the original single-split behaviour.
    """
    raw_dir = Path(raw_dir)
    workbooks = sorted(raw_dir.glob("*.xlsx"))
    if not workbooks:
        raise FileNotFoundError(f"No .xlsx files in {raw_dir}")

    pre_ints = pre_minutes // interval_min
    post_X_ints = post_X_minutes // interval_min
    post_total_ints = post_total_minutes // interval_min
    cfg = episode_config or EpisodeConfig(
        pre_min=pre_minutes,
        post_total_min=post_total_minutes,
        post_grace_min=post_X_minutes,
        interval_min=interval_min,
    )
    allowed_cohorts = set(cohorts) if cohorts is not None else {1, 2}

    X_list: list[np.ndarray] = []
    X_pre_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    meal_list: list[str] = []
    subj_list: list[str] = []
    cohort_list: list[str] = []
    Z_list: list[float] = []
    mediator_list: list[float] = []
    total_bolus_list: list[float] = []
    iob_list: list[float] = []
    glucose_at_meal_list: list[float] = []
    meal_time_list: list[pd.Timestamp] = []
    global_ids: list[int] = []

    next_global_id = 1
    parse_errors: list[str] = []

    for path in workbooks:
        try:
            subject = parse_subject(path)
        except DiaTrendParseError as exc:
            parse_errors.append(f"{path.stem}: {exc}")
            continue
        if subject.cohort not in allowed_cohorts:
            continue

        result = build_episodes(subject, cfg)
        if not result.episodes:
            continue

        bob_lookup: pd.DataFrame | None = None
        if bob_params is not None and subject.cohort == 1:
            bob_lookup = bob_at_meal_times(
                subject.bolus, bob_params, require_meal=False
            ).set_index("date")

        for episode in result.episodes:
            tensor = _episode_to_tensor(
                episode,
                tuple(features),
                pre_ints=pre_ints,
                post_X_ints=post_X_ints,
                bolus_history=subject.bolus if "bolus" in features else None,
            )
            X_list.append(tensor)
            tensor_pre = tensor.copy()
            tensor_pre[pre_ints:] = 0.0
            X_pre_list.append(tensor_pre)

            # Match the OhioT1DM loader (resid_ae_utils.load_windows lines
            # ~159-167): the outcome AND its baseline are taken from the same
            # linearly-interpolated glucose series used for the input window,
            # not from the raw grid. Reading raw values let NaN bins reach Y
            # — a NaN meal-onset baseline (the old nanmean fallback returned
            # NaN when both straddling bins were missing) or NaN gaps inside
            # the outcome window — which poisons the CLAE y_pred head, giving
            # a NaN training loss and all-NaN embeddings. The builder
            # guarantees a non-NaN last bin and >=80% coverage, so
            # bfill/ffill always resolves; no zero-fill is needed (and
            # OhioT1DM applies none to its outcome).
            grid = episode.cgm_grid
            g_filled = (
                pd.Series(grid[: pre_ints + post_total_ints])
                .interpolate(method="linear", limit_direction="both")
                .bfill()
                .ffill()
                .to_numpy(dtype=float)
            )
            base = g_filled[pre_ints]
            outcome_slice = g_filled[
                pre_ints + post_X_ints : pre_ints + post_total_ints
            ]
            y_list.append((outcome_slice - base).astype(float))

            meal_list.append(episode.meal_type)
            subj_list.append(episode.subject_id)
            meal_time_list.append(episode.meal_time)
            cohort_list.append(str(episode.cohort))
            Z_list.append(float(episode.treatment_carbs))
            mediator_list.append(float(episode.mediator_bolus))
            glucose_at_meal_list.append(float(base))
            global_ids.append(next_global_id)
            next_global_id += 1

            lo = episode.meal_time - pd.Timedelta(minutes=pre_minutes)
            hi = episode.meal_time + pd.Timedelta(minutes=post_total_minutes)
            in_full = subject.bolus.loc[
                (subject.bolus["date"] >= lo) & (subject.bolus["date"] <= hi),
                "normal",
            ]
            total_bolus_list.append(float(in_full.fillna(0.0).sum()))

            if subject.cohort == 2:
                iob_value = episode.iob_at_meal
                iob_list.append(
                    float(iob_value) if iob_value is not None else float("nan")
                )
            elif bob_lookup is not None and episode.meal_time in bob_lookup.index:
                iob_list.append(float(bob_lookup.loc[episode.meal_time, "kernel_bob"]))
            else:
                iob_list.append(float("nan"))

    if not X_list:
        raise RuntimeError(
            f"No retained episodes from {len(workbooks)} workbooks in {raw_dir}. "
            f"Parse errors (first 5): {parse_errors[:5]}"
        )

    X_ts = np.stack(X_list, axis=0).astype(float)
    X_ts_pre = np.stack(X_pre_list, axis=0).astype(float)
    y_seq = np.stack(y_list, axis=0).astype(float)

    # Within-subject temporal split (OhioT1DM-style). test_frac == 0 keeps
    # the original single-split behaviour: every episode is labelled "all"
    # and standardization spans the full sample.
    if test_frac > 0:
        split_labels = _within_subject_temporal_split(
            subj_list, meal_time_list, test_frac
        )
    else:
        split_labels = np.full(len(subj_list), "all", dtype=object)

    # Standardize on TRAIN rows only so the encoder never sees test-period
    # statistics. With test_frac == 0 the mask is all-True (full sample).
    train_mask = split_labels != "test"
    if standardize:
        mu = X_ts[train_mask].mean(axis=(0, 1), keepdims=True)
        sd = X_ts[train_mask].std(axis=(0, 1), keepdims=True) + 1e-8
        X_ts_std = (X_ts - mu) / sd
        X_ts_pre_std = (X_ts_pre - mu) / sd
    else:
        mu = np.zeros((1, 1, X_ts.shape[-1]))
        sd = np.ones((1, 1, X_ts.shape[-1]))
        X_ts_std = X_ts
        X_ts_pre_std = X_ts_pre

    meal_arr = np.array(meal_list).reshape(-1, 1)
    subj_arr = np.array(subj_list).reshape(-1, 1)
    meal_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(meal_arr)
    subj_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(subj_arr)
    meal_ohe = meal_encoder.transform(meal_arr)
    subj_ohe = subj_encoder.transform(subj_arr)

    Z = np.array(Z_list, dtype=float)
    # Binarization threshold from TRAIN rows only (matches the train-only
    # standardization); with test_frac == 0 train_mask spans every row.
    Z_train = Z[train_mask]
    treatment_median = float(np.median(Z_train)) if Z_train.size > 0 else 0.0
    Z_bin = (Z > treatment_median).astype(np.float32)
    mediator_scalar = np.array(mediator_list, dtype=float)
    total_bolus_arr = np.array(total_bolus_list, dtype=float)
    iob_arr = np.array(iob_list, dtype=float)
    glucose_at_meal_raw = np.array(glucose_at_meal_list, dtype=float)
    global_window_id = np.array(global_ids, dtype=int)
    cohort_labels = np.array(cohort_list)

    data_tuple = (
        X_ts_std,
        X_ts_pre_std,
        meal_ohe,
        subj_ohe,
        Z,
        Z_bin,
        y_seq,
        mediator_scalar,
        global_window_id,
        meal_list,
        subj_list,
        pre_ints,
        post_X_ints,
        total_bolus_arr,
    )

    std_params = {
        "mu": mu,
        "sd": sd,
        "treatment_median": treatment_median,
        "pre_ints": pre_ints,
        "meal_encoder": meal_encoder,
        "subj_encoder": subj_encoder,
        "glucose_at_meal_raw": glucose_at_meal_raw,
        "iob_at_meal": iob_arr,
        "split_labels": split_labels,
        "parse_errors": parse_errors,
    }

    return data_tuple, cohort_labels, std_params

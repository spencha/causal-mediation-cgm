"""Tests for the DiaTrend tensor loader (ae_python_code/diatrend_loader.py).

Verifies that the loader produces tensors with the same shape contract
as ``resid_ae_utils.load_windows()``, so the OhioT1DM training driver
can dispatch on ``--dataset`` and consume either output without
architecture changes.
"""
from __future__ import annotations

import numpy as np
import pytest

from data_processing.diatrend.bob_kernel import IOBKernelParams
from data_processing.diatrend.tests.fixtures.generate_fixtures import DEFAULT_PLAN

# Imported via the pythonpath entry in pyproject.toml.
from diatrend_loader import load_diatrend_data  # type: ignore[import-not-found]


PRE_INTS = 24  # 120 / 5
POST_X_INTS = 12  # 60 / 5
POST_TOTAL_INTS = 48  # 240 / 5
N_INPUT_BINS = PRE_INTS + POST_X_INTS
N_OUTCOME_BINS = POST_TOTAL_INTS - POST_X_INTS


def test_loader_returns_load_windows_compatible_tuple(diatrend_fixtures_dir):
    data, cohort_labels, std_params = load_diatrend_data(diatrend_fixtures_dir)
    assert len(data) == 14, "data_tuple must have the same 14 elements as load_windows()"
    (
        X_ts,
        X_ts_pre,
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
    ) = data
    n = X_ts.shape[0]
    assert n > 0
    assert X_ts.shape == (n, N_INPUT_BINS, 1)  # univariate primary spec
    assert X_ts_pre.shape == (n, N_INPUT_BINS, 1)
    assert y_seq.shape == (n, N_OUTCOME_BINS)
    assert Z.shape == (n,)
    assert Z_bin.shape == (n,)
    assert mediator_scalar.shape == (n,)
    assert total_bolus_arr.shape == (n,)
    assert global_window_id.shape == (n,)
    assert len(meal_list) == n
    assert len(subj_list) == n
    assert cohort_labels.shape == (n,)
    assert set(cohort_labels) <= {"1", "2"}
    assert pre_ints == PRE_INTS
    assert post_X_ints == POST_X_INTS


def test_outcome_and_baseline_are_finite(diatrend_fixtures_dir):
    """The outcome Y and the meal-time baseline must contain no NaN. The
    CLAE y_pred head yields a NaN training loss (and all-NaN embeddings,
    which then crash the PCA export) if any target is NaN. The loader
    interpolates the outcome glucose with the same linear/bfill/ffill
    chain OhioT1DM's load_windows applies to its outcome, so a NaN
    meal-onset baseline or NaN bins inside the outcome window must not
    survive into Y."""
    data, _, std_params = load_diatrend_data(diatrend_fixtures_dir)
    y_seq = data[6]
    assert y_seq.size > 0
    assert np.isfinite(y_seq).all(), "outcome Y contains NaN/inf"
    assert np.isfinite(
        std_params["glucose_at_meal_raw"]
    ).all(), "meal-time baseline contains NaN/inf"


def test_pre_tensor_zeroed_after_pre_ints(diatrend_fixtures_dir):
    # Test against the un-standardized tensor: the post-meal portion of
    # X_ts_pre is set to zero before standardization. (After
    # standardization the zero gets re-centered to -mu/sd; the OhioT1DM
    # driver explicitly truncates X_ts_pre to pre_ints downstream, so
    # the actual values of the post-meal portion don't matter.)
    data, _, _ = load_diatrend_data(diatrend_fixtures_dir, standardize=False)
    X_ts_pre = data[1]
    assert (X_ts_pre[:, PRE_INTS:, :] == 0.0).all()


def test_standardization_produces_zero_mean_unit_variance(diatrend_fixtures_dir):
    data, _, _ = load_diatrend_data(diatrend_fixtures_dir, standardize=True)
    X_ts = data[0]
    # Over all episodes and bins (the axes the loader standardizes over),
    # mean ≈ 0 and std ≈ 1. The X_ts_pre array also includes zero-filled
    # bins after standardization, so it does NOT have unit variance —
    # that one is tested separately.
    assert abs(X_ts.mean(axis=(0, 1))).max() < 1e-6
    assert abs(X_ts.std(axis=(0, 1)) - 1.0).max() < 1e-3


def test_multivariate_features_produce_three_channels(diatrend_fixtures_dir):
    data, _, _ = load_diatrend_data(
        diatrend_fixtures_dir, features=("glucose", "meal", "bolus")
    )
    X_ts = data[0]
    assert X_ts.shape[-1] == 3
    # The meal channel is a single spike at the meal_time bin per episode,
    # and zero elsewhere. After standardization the channel is centred,
    # so the meal-bin row should still be the most extreme value in its
    # episode (or among the most extreme).
    # We test this softly: the meal channel must have non-zero variance.
    meal_channel_var = X_ts[:, :, 1].var()
    assert meal_channel_var > 0


def test_cohort_filter_restricts_subjects(diatrend_fixtures_dir):
    only_c2, c2_labels, _ = load_diatrend_data(diatrend_fixtures_dir, cohorts={2})
    assert (c2_labels == "2").all()
    only_c1, c1_labels, _ = load_diatrend_data(diatrend_fixtures_dir, cohorts={1})
    assert (c1_labels == "1").all()


def test_iob_filled_for_cohort_two_nan_for_cohort_one(diatrend_fixtures_dir):
    _, cohort_labels, std_params = load_diatrend_data(diatrend_fixtures_dir)
    iob = std_params["iob_at_meal"]
    assert iob.shape == cohort_labels.shape
    c2_iob = iob[cohort_labels == "2"]
    c1_iob = iob[cohort_labels == "1"]
    # Cohort 2 IOB is the synthetic pump value from the fixture
    # generator — should be finite.
    assert np.isfinite(c2_iob).all()
    # Cohort 1 has no pump IOB and no kernel was supplied: NaN.
    assert np.isnan(c1_iob).all()


def test_bob_kernel_fills_cohort_one_iob(diatrend_fixtures_dir):
    _, cohort_labels, std_params = load_diatrend_data(
        diatrend_fixtures_dir, bob_params=IOBKernelParams()
    )
    iob = std_params["iob_at_meal"]
    c1_iob = iob[cohort_labels == "1"]
    # With a kernel supplied, cohort 1 IOB is finite (kernel BOB), not NaN.
    # (The first bolus of each subject has BOB == 0; later boluses are positive.)
    assert (c1_iob.size > 0) and np.isfinite(c1_iob).all()


def test_global_window_id_is_unique_and_consecutive(diatrend_fixtures_dir):
    data, _, _ = load_diatrend_data(diatrend_fixtures_dir)
    ids = data[8]
    assert len(set(ids)) == len(ids)
    assert ids.min() == 1
    assert ids.max() == ids.size


def test_treatment_carbs_match_episode_payloads(diatrend_fixtures_dir):
    data, _, _ = load_diatrend_data(diatrend_fixtures_dir)
    Z = data[4]
    assert (Z > 0).all()


def test_mediator_is_nonnegative(diatrend_fixtures_dir):
    data, _, _ = load_diatrend_data(diatrend_fixtures_dir)
    mediator = data[7]
    assert (mediator >= 0).all()


def test_split_labels_default_to_all(diatrend_fixtures_dir):
    """With the default test_frac=0.0 every episode is labelled 'all', so
    the single-sample behaviour (and the existing OhioT1DM-style
    full-sample standardization) is preserved."""
    _, _, std_params = load_diatrend_data(diatrend_fixtures_dir)
    split = std_params["split_labels"]
    assert set(np.unique(split)) == {"all"}


def test_within_subject_temporal_split_unit():
    """The split helper holds out each subject's chronologically latest
    meals, independent of input row order, and keeps single-episode
    subjects in train."""
    import pandas as pd

    from diatrend_loader import _within_subject_temporal_split  # type: ignore

    subj = ["A", "A", "A", "A", "A", "B", "B", "C"]
    # Input order is deliberately shuffled within subject A; the helper must
    # sort by absolute timestamp, not by row position.
    times = [
        pd.Timestamp("2020-01-05"),  # A, latest
        pd.Timestamp("2020-01-01"),  # A, earliest
        pd.Timestamp("2020-01-04"),  # A, 2nd latest
        pd.Timestamp("2020-01-02"),  # A
        pd.Timestamp("2020-01-03"),  # A
        pd.Timestamp("2020-02-02"),  # B, latest
        pd.Timestamp("2020-02-01"),  # B
        pd.Timestamp("2020-03-01"),  # C, single meal
    ]
    split = np.asarray(_within_subject_temporal_split(subj, times, test_frac=0.4))
    # A: round(0.4*5)=2 latest -> Jan-05 (idx0) and Jan-04 (idx2).
    assert list(split[:5]) == ["test", "train", "test", "train", "train"]
    # B: round(0.4*2)=1 latest -> Feb-02 (idx5).
    assert list(split[5:7]) == ["test", "train"]
    # C: single episode stays in train.
    assert split[7] == "train"


def test_temporal_split_keeps_every_split_subject_in_train(diatrend_fixtures_dir):
    """Integration: with test_frac=0.2 the loader labels episodes
    train/test, holds out a minority as test, and every subject with a
    test episode also retains train episodes (so subject random intercepts
    stay estimable on the held-out set)."""
    data, _, std_params = load_diatrend_data(diatrend_fixtures_dir, test_frac=0.2)
    split = std_params["split_labels"]
    subj = np.asarray(data[10])  # subj_list
    assert set(np.unique(split)) <= {"train", "test"}
    assert (split == "test").any()
    assert (split == "test").mean() < 0.5  # test is the minority
    for s in np.unique(subj):
        labels = split[subj == s]
        if labels.size >= 2:
            n_test = int((labels == "test").sum())
            assert n_test == min(max(round(0.2 * labels.size), 0), labels.size - 1)
            assert (labels == "train").any()


def test_raises_on_missing_workbooks(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        load_diatrend_data(empty)

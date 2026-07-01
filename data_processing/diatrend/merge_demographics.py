"""Merge DiaTrend subject demographics into a phi-embeddings CSV.

Adds subject-level covariates (age, sex, HbA1c) keyed by subject so the
R mediation pipeline can include them as confounders in the propensity,
mediator, and outcome models. Demographics are subject-constant, so each
subject's values are broadcast to every one of that subject's episodes.

Run once per embeddings CSV (no CLAE retraining needed); both
npcbps_weights.R and run_mixed_effects_mediation.R then read the
augmented CSV and add the demo_* columns to their formulas when
--demographics is set.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Source column in SubjectDemographics_Feb2025.xlsx -> standardized output
# column. Race / CGM model / insulin-delivery columns are intentionally not
# carried (not requested as model covariates). Keep the standardized names
# space-free so they are safe as R formula terms.
DEMOGRAPHIC_COLUMNS = {
    "Age": "demo_age",
    "Gender": "demo_sex",
    "Hemoglobin A1C": "demo_hba1c",
}


def merge_demographics(
    embeddings: pd.DataFrame,
    demographics: pd.DataFrame,
    *,
    subject_col: str = "Subject",
) -> pd.DataFrame:
    """Left-join demographic covariates onto an embeddings frame.

    ``embeddings`` must have a ``subject_id`` column formatted like
    ``Subject<N>`` (the parser's ``path.stem``). ``demographics`` must have
    an integer ``Subject`` column (1..54) plus the keys in
    DEMOGRAPHIC_COLUMNS. Every episode of a subject receives that subject's
    constant demographic values; subjects absent from the demographics
    table get NaN (and will drop from any model that uses them).
    """
    needed = [subject_col, *DEMOGRAPHIC_COLUMNS]
    missing = [c for c in needed if c not in demographics.columns]
    if missing:
        raise ValueError(f"demographics missing columns: {missing}")
    if "subject_id" not in embeddings.columns:
        raise ValueError("embeddings missing subject_id column")

    demo = demographics[needed].rename(columns=DEMOGRAPHIC_COLUMNS).copy()
    demo["subject_id"] = "Subject" + demo[subject_col].astype(int).astype(str)
    demo = demo.drop(columns=[subject_col])

    return embeddings.merge(demo, on="subject_id", how="left")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Augment a DiaTrend embeddings CSV with subject demographics."
    )
    ap.add_argument("--embeddings-csv", required=True, type=Path)
    ap.add_argument("--demographics-xlsx", required=True, type=Path)
    ap.add_argument("--output-csv", required=True, type=Path)
    args = ap.parse_args()

    emb = pd.read_csv(args.embeddings_csv)
    demo = pd.read_excel(args.demographics_xlsx)
    merged = merge_demographics(emb, demo)

    out_cols = list(DEMOGRAPHIC_COLUMNS.values())
    bad = merged[out_cols].isna().any(axis=1)
    if int(bad.sum()):
        n_subj = merged.loc[bad, "subject_id"].nunique()
        print(
            f"  [warn] {int(bad.sum())} episodes across {n_subj} subject(s) have "
            f"missing demographics (NaN); they drop from models that use them."
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_csv, index=False)
    print(f"  Wrote {len(merged)} rows, {merged.shape[1]} columns -> {args.output_csv}")
    print(f"  Added columns: {out_cols}")


if __name__ == "__main__":
    main()

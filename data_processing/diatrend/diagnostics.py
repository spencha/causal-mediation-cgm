"""DiaTrend diagnostic report writer.

Aggregated-only window into the real DiaTrend data, written from
HPC3 against the actual workbooks. The coding agent cannot see any
patient-level value directly; this report is the bridge that lets
parser / episode-builder defects surface without exposing
individual records.

The report has two output files, both written to
``analysis_data/diatrend/diagnostics/``:

- ``<run_id>.md`` — human-readable Markdown
- ``<run_id>.json`` — same data in machine-readable form for diffing
  across runs

Section 13 of the handoff governs what the report may and may not
contain. Permitted: aggregated counts, durations, pooled summary
statistics, subject identifiers. Forbidden: individual timestamps,
individual glucose readings, individual bolus doses, single-subject
time series. The diagnostic writer enforces this by only ever
emitting aggregates and durations; no raw observation flows through
to either output file.

Initial checks called out in Section 13 are all present here:
reverse-chronology detection (list of affected subjects), CGM
gap distribution, and bolus coverage window per subject.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from data_processing.diatrend.episode_builder import (
    EpisodeBuildResult,
    build_episodes,
)
from data_processing.diatrend.parser import (
    DiaTrendParseError,
    SubjectData,
    parse_subject,
)


@dataclass(frozen=True)
class SubjectDiagnostic:
    subject_id: str
    cohort: int | None
    bolus_schema: str | None
    was_reversed: bool
    cgm_rows: int
    bolus_rows: int
    basal_rows: int
    cgm_duration_days: float
    bolus_duration_days: float
    bolus_coverage_frac: float
    cgm_gap_median_min: float
    cgm_gap_p95_min: float
    cgm_gap_5min_match_rate: float
    cgm_gaps_gt_15min: int
    cgm_gaps_gt_60min: int
    bolus_meal_count: int
    bolus_correction_count: int
    bolus_other_count: int
    candidate_meals: int
    retained_episodes: int
    rejected_episodes: int
    parse_error: str | None = None


@dataclass(frozen=True)
class FeatureSummary:
    name: str
    n: int
    n_missing: int
    pmin: float
    p25: float
    median: float
    p75: float
    pmax: float


@dataclass(frozen=True, eq=False)
class DiagnosticReport:
    run_id: str
    n_subjects: int
    n_cohort_1: int
    n_cohort_2: int
    n_failed_parse: int
    subjects: list[SubjectDiagnostic]
    reversed_subject_ids: list[str]
    pooled_cgm_gap_median_min: float
    pooled_cgm_gap_p95_min: float
    pooled_cgm_5min_match_rate: float
    pooled_cgm_gaps_gt_15min: int
    pooled_cgm_gaps_gt_60min: int
    rejection_reason_counts: dict[str, int]
    meal_type_counts_by_cohort: dict[int, dict[str, int]]
    feature_summaries: dict[str, FeatureSummary]
    retained_feature_summaries: dict[str, FeatureSummary]


def _gap_minutes(cgm: pd.DataFrame) -> np.ndarray:
    if len(cgm) < 2:
        return np.array([], dtype=float)
    return (
        cgm["date"].diff().dt.total_seconds().iloc[1:].to_numpy() / 60.0
    )


def _gap_summary(diffs_min: np.ndarray) -> dict:
    if diffs_min.size == 0:
        return {
            "median": float("nan"),
            "p95": float("nan"),
            "match_5min": 0.0,
            "gaps_gt_15min": 0,
            "gaps_gt_60min": 0,
        }
    return {
        "median": float(np.median(diffs_min)),
        "p95": float(np.percentile(diffs_min, 95)),
        "match_5min": float(((diffs_min >= 4.5) & (diffs_min <= 5.5)).mean()),
        "gaps_gt_15min": int((diffs_min > 15).sum()),
        "gaps_gt_60min": int((diffs_min > 60).sum()),
    }


def _bolus_counts(bolus: pd.DataFrame) -> tuple[int, int, int]:
    if bolus.empty:
        return 0, 0, 0
    carbs = bolus["carbInput"].fillna(0.0).to_numpy()
    normal = bolus["normal"].fillna(0.0).to_numpy()
    meal = int((carbs > 0).sum())
    correction = int(((carbs == 0) & (normal > 0)).sum())
    other = int(((carbs == 0) & (normal == 0)).sum())
    return meal, correction, other


def _span_days(frame: pd.DataFrame) -> float:
    if len(frame) < 2:
        return 0.0
    return float((frame["date"].iloc[-1] - frame["date"].iloc[0]).total_seconds() / 86400.0)


def _coverage_fraction(cgm: pd.DataFrame, bolus: pd.DataFrame) -> float:
    cgm_days = _span_days(cgm)
    bolus_days = _span_days(bolus)
    if cgm_days <= 0:
        return 0.0
    return float(bolus_days / cgm_days)


def _five_number(name: str, values: np.ndarray) -> FeatureSummary:
    n_total = int(values.size)
    finite = values[np.isfinite(values)]
    n_missing = n_total - int(finite.size)
    if finite.size == 0:
        nan = float("nan")
        return FeatureSummary(name, n_total, n_missing, nan, nan, nan, nan, nan)
    return FeatureSummary(
        name=name,
        n=n_total,
        n_missing=n_missing,
        pmin=float(finite.min()),
        p25=float(np.percentile(finite, 25)),
        median=float(np.median(finite)),
        p75=float(np.percentile(finite, 75)),
        pmax=float(finite.max()),
    )


def _subject_diagnostic(
    subject: SubjectData, result: EpisodeBuildResult
) -> SubjectDiagnostic:
    diffs = _gap_minutes(subject.cgm)
    gap = _gap_summary(diffs)
    meal_c, corr_c, other_c = _bolus_counts(subject.bolus)
    return SubjectDiagnostic(
        subject_id=subject.subject_id,
        cohort=subject.cohort,
        bolus_schema=subject.bolus_schema,
        was_reversed=subject.was_reversed,
        cgm_rows=int(len(subject.cgm)),
        bolus_rows=int(len(subject.bolus)),
        basal_rows=int(len(subject.basal)) if subject.basal is not None else 0,
        cgm_duration_days=_span_days(subject.cgm),
        bolus_duration_days=_span_days(subject.bolus),
        bolus_coverage_frac=_coverage_fraction(subject.cgm, subject.bolus),
        cgm_gap_median_min=gap["median"],
        cgm_gap_p95_min=gap["p95"],
        cgm_gap_5min_match_rate=gap["match_5min"],
        cgm_gaps_gt_15min=gap["gaps_gt_15min"],
        cgm_gaps_gt_60min=gap["gaps_gt_60min"],
        bolus_meal_count=meal_c,
        bolus_correction_count=corr_c,
        bolus_other_count=other_c,
        candidate_meals=result.n_candidate_meals,
        retained_episodes=len(result.episodes),
        rejected_episodes=len(result.rejected),
    )


def _failed_subject_diagnostic(subject_id: str, error: str) -> SubjectDiagnostic:
    nan = float("nan")
    return SubjectDiagnostic(
        subject_id=subject_id,
        cohort=None,
        bolus_schema=None,
        was_reversed=False,
        cgm_rows=0,
        bolus_rows=0,
        basal_rows=0,
        cgm_duration_days=0.0,
        bolus_duration_days=0.0,
        bolus_coverage_frac=0.0,
        cgm_gap_median_min=nan,
        cgm_gap_p95_min=nan,
        cgm_gap_5min_match_rate=0.0,
        cgm_gaps_gt_15min=0,
        cgm_gaps_gt_60min=0,
        bolus_meal_count=0,
        bolus_correction_count=0,
        bolus_other_count=0,
        candidate_meals=0,
        retained_episodes=0,
        rejected_episodes=0,
        parse_error=error,
    )


def _numeric_id_key(subject_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in subject_id if ch.isdigit())
    if digits:
        try:
            return (int(digits), subject_id)
        except ValueError:
            pass
    return (10**9, subject_id)


def compute_diagnostics(
    parsed: Iterable[tuple[SubjectData | None, EpisodeBuildResult | None, str | None]],
    *,
    run_id: str,
) -> DiagnosticReport:
    """Aggregate parsed subjects and built results into a report.

    Each input tuple is ``(subject_or_None, result_or_None, error_or_None)``.
    Exactly one of (subject, error) is non-None per tuple.
    """
    subjects: list[SubjectDiagnostic] = []
    pooled_gaps: list[np.ndarray] = []
    rejection_counts: dict[str, int] = {}
    meal_type_by_cohort: dict[int, dict[str, int]] = {1: {}, 2: {}}
    carb_values: list[float] = []
    normal_values: list[float] = []
    iob_values: list[float] = []
    bg_values: list[float] = []
    # Same four features, but measured over RETAINED episodes only (post all
    # episode-builder filters) rather than raw bolus rows. Lets us see the
    # actual treatment / mediator distributions that reach the model — e.g.
    # the carb cap means retained treatment_carbs never exceeds 200 g.
    retained_carb_values: list[float] = []
    retained_normal_values: list[float] = []
    retained_iob_values: list[float] = []
    retained_bg_values: list[float] = []

    n_cohort_1 = 0
    n_cohort_2 = 0
    n_failed = 0

    for subject, result, error in parsed:
        if error is not None:
            subjects.append(_failed_subject_diagnostic(error.split(": ", 1)[0], error))
            n_failed += 1
            continue
        assert subject is not None
        assert result is not None
        subjects.append(_subject_diagnostic(subject, result))
        if subject.cohort == 1:
            n_cohort_1 += 1
        else:
            n_cohort_2 += 1

        pooled_gaps.append(_gap_minutes(subject.cgm))

        for rejected in result.rejected:
            key = _rejection_category(rejected.reason)
            rejection_counts[key] = rejection_counts.get(key, 0) + 1

        for ep in result.episodes:
            meal_type_by_cohort[subject.cohort][ep.meal_type] = (
                meal_type_by_cohort[subject.cohort].get(ep.meal_type, 0) + 1
            )
            retained_carb_values.append(ep.treatment_carbs)
            retained_normal_values.append(ep.mediator_bolus)
            if ep.bg_input_at_meal is not None:
                retained_bg_values.append(ep.bg_input_at_meal)
            if ep.iob_at_meal is not None:
                retained_iob_values.append(ep.iob_at_meal)

        carb_values.extend(
            subject.bolus.loc[subject.bolus["carbInput"] > 0, "carbInput"].dropna().tolist()
        )
        normal_values.extend(subject.bolus["normal"].dropna().tolist())
        bg_values.extend(subject.bolus["bgInput"].dropna().tolist())
        if subject.cohort == 2 and "insulinOnBoard" in subject.bolus.columns:
            iob_values.extend(subject.bolus["insulinOnBoard"].dropna().tolist())

    subjects.sort(key=lambda s: _numeric_id_key(s.subject_id))

    pooled = (
        np.concatenate(pooled_gaps)
        if pooled_gaps
        else np.array([], dtype=float)
    )
    pooled_summary = _gap_summary(pooled)

    feature_summaries = {
        "carbInput": _five_number("carbInput", np.array(carb_values, dtype=float)),
        "normal": _five_number("normal", np.array(normal_values, dtype=float)),
        "bgInput": _five_number("bgInput", np.array(bg_values, dtype=float)),
        "insulinOnBoard": _five_number(
            "insulinOnBoard", np.array(iob_values, dtype=float)
        ),
    }

    retained_feature_summaries = {
        "treatment_carbs": _five_number(
            "treatment_carbs", np.array(retained_carb_values, dtype=float)
        ),
        "mediator_bolus": _five_number(
            "mediator_bolus", np.array(retained_normal_values, dtype=float)
        ),
        "bg_input_at_meal": _five_number(
            "bg_input_at_meal", np.array(retained_bg_values, dtype=float)
        ),
        "iob_at_meal": _five_number(
            "iob_at_meal", np.array(retained_iob_values, dtype=float)
        ),
    }

    reversed_ids = sorted(
        (s.subject_id for s in subjects if s.was_reversed),
        key=_numeric_id_key,
    )

    return DiagnosticReport(
        run_id=run_id,
        n_subjects=len(subjects),
        n_cohort_1=n_cohort_1,
        n_cohort_2=n_cohort_2,
        n_failed_parse=n_failed,
        subjects=subjects,
        reversed_subject_ids=reversed_ids,
        pooled_cgm_gap_median_min=pooled_summary["median"],
        pooled_cgm_gap_p95_min=pooled_summary["p95"],
        pooled_cgm_5min_match_rate=pooled_summary["match_5min"],
        pooled_cgm_gaps_gt_15min=pooled_summary["gaps_gt_15min"],
        pooled_cgm_gaps_gt_60min=pooled_summary["gaps_gt_60min"],
        rejection_reason_counts=dict(sorted(rejection_counts.items())),
        meal_type_counts_by_cohort=meal_type_by_cohort,
        feature_summaries=feature_summaries,
        retained_feature_summaries=retained_feature_summaries,
    )


_REJECTION_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("implausible bgInput", "implausible bgInput"),
    ("implausible carb", "implausible carb input"),
    ("contamination", "postprandial contamination"),
    ("full", "CGM record does not span the full window"),
    ("last CGM bin", "last CGM bin NaN"),
    ("consecutive NaN", "max consecutive NaN exceeded"),
    ("postprandial coverage", "postprandial coverage below threshold"),
    ("no CGM", "no CGM data"),
)


def _rejection_category(reason: str) -> str:
    for keyword, category in _REJECTION_KEYWORDS:
        if keyword in reason:
            return category
    return "other"


def _fmt(value, fmt: str = "{:.2f}") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if np.isnan(value):
            return "—"
        return fmt.format(value)
    return str(value)


def _markdown(report: DiagnosticReport) -> str:
    lines: list[str] = []
    lines.append(f"# DiaTrend diagnostic report — {report.run_id}")
    lines.append("")
    lines.append(
        f"Aggregated-only summary. No individual timestamps, glucose readings, "
        f"bolus doses, or single-subject time series appear in this file."
    )
    lines.append("")

    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Subjects scanned: **{report.n_subjects}**")
    lines.append(f"- Cohort 1 (base bolus schema, has Basal): **{report.n_cohort_1}**")
    lines.append(f"- Cohort 2 (extended bolus schema, no Basal): **{report.n_cohort_2}**")
    lines.append(f"- Failed to parse: **{report.n_failed_parse}**")
    if report.reversed_subject_ids:
        lines.append(
            f"- Reverse-chronology detected and corrected on **{len(report.reversed_subject_ids)}** subjects: "
            f"{', '.join(report.reversed_subject_ids)}"
        )
    else:
        lines.append("- No reverse-chronology subjects detected.")
    lines.append("")

    lines.append("## Per-subject parse status")
    lines.append("")
    lines.append(
        "| Subject | Cohort | Schema | Reversed | CGM rows | Bolus rows | Basal rows | "
        "Record span (days) | Bolus coverage frac | CGM gap median (min) | "
        "Gaps > 15 min | Meal boluses | Corrections | Candidate meals | Retained | Rejected | Parse error |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for s in report.subjects:
        if s.parse_error is not None:
            lines.append(
                f"| {s.subject_id} | — | — | — | 0 | 0 | 0 | 0.00 | 0.00 | — | 0 | 0 | 0 | 0 | 0 | 0 | {s.parse_error} |"
            )
            continue
        lines.append(
            f"| {s.subject_id} | {s.cohort} | {s.bolus_schema} | "
            f"{'yes' if s.was_reversed else 'no'} | "
            f"{s.cgm_rows} | {s.bolus_rows} | {s.basal_rows} | "
            f"{_fmt(s.cgm_duration_days)} | {_fmt(s.bolus_coverage_frac)} | "
            f"{_fmt(s.cgm_gap_median_min)} | {s.cgm_gaps_gt_15min} | "
            f"{s.bolus_meal_count} | {s.bolus_correction_count} | "
            f"{s.candidate_meals} | {s.retained_episodes} | {s.rejected_episodes} | |"
        )
    lines.append("")

    lines.append("## CGM temporal sanity (pooled across subjects)")
    lines.append("")
    lines.append(f"- Median inter-sample gap: **{_fmt(report.pooled_cgm_gap_median_min)} min**")
    lines.append(f"- 95th-percentile inter-sample gap: **{_fmt(report.pooled_cgm_gap_p95_min)} min**")
    lines.append(
        f"- Fraction of samples on the 5-min grid (±30 s): "
        f"**{_fmt(report.pooled_cgm_5min_match_rate, '{:.3f}')}**"
    )
    lines.append(f"- Pooled gaps > 15 min: **{report.pooled_cgm_gaps_gt_15min}**")
    lines.append(f"- Pooled gaps > 60 min: **{report.pooled_cgm_gaps_gt_60min}**")
    lines.append("")

    lines.append("## Episode-construction rejections")
    lines.append("")
    if not report.rejection_reason_counts:
        lines.append("No rejected meals.")
    else:
        lines.append("| Reason | Count |")
        lines.append("|---|---|")
        for reason, count in sorted(
            report.rejection_reason_counts.items(), key=lambda kv: -kv[1]
        ):
            lines.append(f"| {reason} | {count} |")
    lines.append("")

    lines.append("## Retained episodes by meal type and cohort")
    lines.append("")
    meal_types = sorted(
        {
            mt
            for cohort_dict in report.meal_type_counts_by_cohort.values()
            for mt in cohort_dict
        }
    )
    if meal_types:
        lines.append("| Meal type | Cohort 1 | Cohort 2 |")
        lines.append("|---|---|---|")
        for mt in meal_types:
            c1 = report.meal_type_counts_by_cohort.get(1, {}).get(mt, 0)
            c2 = report.meal_type_counts_by_cohort.get(2, {}).get(mt, 0)
            lines.append(f"| {mt} | {c1} | {c2} |")
    else:
        lines.append("No retained episodes.")
    lines.append("")

    lines.append(
        "## Feature distributions — raw bolus rows (pooled across all subjects)"
    )
    lines.append("")
    lines.append(
        "Computed over every bolus row (carbInput > 0 for `carbInput`), "
        "BEFORE episode-builder filtering. Episode-level caps (carb 200 g, "
        "bgInput 600 mg/dL) do NOT bound these values."
    )
    lines.append("")
    lines.append("| Feature | n | n missing | min | p25 | median | p75 | max |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name in ("carbInput", "normal", "bgInput", "insulinOnBoard"):
        fs = report.feature_summaries[name]
        lines.append(
            f"| {fs.name} | {fs.n} | {fs.n_missing} | "
            f"{_fmt(fs.pmin)} | {_fmt(fs.p25)} | {_fmt(fs.median)} | "
            f"{_fmt(fs.p75)} | {_fmt(fs.pmax)} |"
        )
    lines.append("")

    lines.append(
        "## Feature distributions — retained episodes (post-filter, pooled)"
    )
    lines.append("")
    lines.append(
        "Computed over RETAINED episodes only, so these reflect the "
        "treatment / mediator / context values that reach the model. The "
        "carb cap bounds `treatment_carbs` at 200 g and the bgInput cap "
        "bounds `bg_input_at_meal` at 600 mg/dL. `iob_at_meal` is cohort-2 "
        "only."
    )
    lines.append("")
    lines.append("| Feature | n | n missing | min | p25 | median | p75 | max |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name in (
        "treatment_carbs",
        "mediator_bolus",
        "bg_input_at_meal",
        "iob_at_meal",
    ):
        fs = report.retained_feature_summaries[name]
        lines.append(
            f"| {fs.name} | {fs.n} | {fs.n_missing} | "
            f"{_fmt(fs.pmin)} | {_fmt(fs.p25)} | {_fmt(fs.median)} | "
            f"{_fmt(fs.p75)} | {_fmt(fs.pmax)} |"
        )
    lines.append("")

    return "\n".join(lines) + "\n"


def write_markdown(report: DiagnosticReport, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(_markdown(report))


def _report_to_json(report: DiagnosticReport) -> dict:
    payload = asdict(report)
    # JSON cannot use int keys for objects; remap meal_type_counts_by_cohort.
    payload["meal_type_counts_by_cohort"] = {
        str(k): v for k, v in payload["meal_type_counts_by_cohort"].items()
    }
    return payload


def write_json(report: DiagnosticReport, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_report_to_json(report), indent=2, default=str))


def _default_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d_%H%M%S")


def run(
    input_dir: Path | str,
    output_dir: Path | str,
    run_id: str | None = None,
) -> tuple[Path, Path]:
    """Parse every workbook in ``input_dir``, build episodes, write the report.

    Returns ``(md_path, json_path)``.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    rid = run_id or _default_run_id()

    workbooks = sorted(input_dir.glob("*.xlsx"))
    if not workbooks:
        raise FileNotFoundError(f"No .xlsx files in {input_dir}")

    triples: list[tuple[SubjectData | None, EpisodeBuildResult | None, str | None]] = []
    for path in workbooks:
        try:
            subject = parse_subject(path)
            result = build_episodes(subject)
            triples.append((subject, result, None))
        except DiaTrendParseError as exc:
            triples.append((None, None, f"{path.stem}: {exc}"))

    report = compute_diagnostics(triples, run_id=rid)
    md_path = output_dir / f"{rid}.md"
    json_path = output_dir / f"{rid}.json"
    write_markdown(report, md_path)
    write_json(report, json_path)
    return md_path, json_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="diatrend-diagnostics",
        description=(
            "Run the DiaTrend diagnostic pass over a directory of "
            "subject workbooks and emit a Markdown + JSON report."
        ),
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing per-subject .xlsx workbooks.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Directory to write the report into. Created if missing. "
            "On HPC3 this should be "
            "`analysis_data/diatrend/diagnostics/`."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Identifier used as the report filename stem. "
            "Defaults to current UTC time as `YYYY-MM-DD_HHMMSS`."
        ),
    )
    args = parser.parse_args(argv)
    try:
        md_path, json_path = run(args.input_dir, args.output_dir, args.run_id)
    except FileNotFoundError as exc:
        print(f"diatrend-diagnostics: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

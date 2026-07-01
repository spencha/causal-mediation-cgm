"""Raw-workbook parser for the DiaTrend dataset.

Loads one subject's Excel workbook (CGM + Bolus sheets, plus Basal for
cohort 1) and returns cleaned DataFrames with cohort and reverse-chronology
metadata. Strictly a parser: no episode construction, no feature
engineering, no aggregation. Those live in their own modules so this
file remains small and testable against synthetic fixtures.

DiaTrend schema reference: Prioleau et al., *Scientific Data* (2023);
also section 6 of the project handoff. The two cohorts are complementary:
cohort 1 has the Basal sheet and the 7-column base bolus schema; cohort 2
has no Basal sheet and the 11-column extended bolus schema (with
``insulinOnBoard``). Disagreement between these two signals is treated
as a hard error since the real data is documented to never produce one.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

CGM_REQUIRED_COLS: frozenset[str] = frozenset({"date", "mg/dl"})
BOLUS_BASE_COLS: frozenset[str] = frozenset(
    {
        "date",
        "normal",
        "carbInput",
        "insulinCarbRatio",
        "bgInput",
        "recommended.carb",
        "recommended.net",
    }
)
BOLUS_EXTENDED_EXTRA_COLS: frozenset[str] = frozenset(
    {
        "recommended.correction",
        "insulinSensitivityFactor",
        "targetBloodGlucose",
        "insulinOnBoard",
    }
)
BOLUS_EXTENDED_COLS: frozenset[str] = BOLUS_BASE_COLS | BOLUS_EXTENDED_EXTRA_COLS
BASAL_REQUIRED_COLS: frozenset[str] = frozenset({"date", "duration", "rate"})

CGM_SHEET = "CGM"
BOLUS_SHEET = "Bolus"
BASAL_SHEET = "Basal"


class DiaTrendParseError(ValueError):
    """Raised when a workbook violates the documented DiaTrend schema."""


@dataclass(frozen=True)
class SubjectData:
    subject_id: str
    cohort: Literal[1, 2]
    bolus_schema: Literal["base", "extended"]
    cgm: pd.DataFrame
    bolus: pd.DataFrame
    basal: pd.DataFrame | None
    was_reversed: bool


def _validate_columns(
    actual: pd.Index, required: frozenset[str], sheet: str, subject_id: str
) -> None:
    actual_set = set(actual)
    missing = required - actual_set
    if missing:
        raise DiaTrendParseError(
            f"{subject_id}: sheet '{sheet}' missing required columns: {sorted(missing)}"
        )


def _classify_bolus_schema(
    columns: pd.Index, subject_id: str
) -> Literal["base", "extended"]:
    cols = set(columns)
    if BOLUS_EXTENDED_COLS <= cols:
        return "extended"
    if BOLUS_BASE_COLS <= cols:
        extra = cols - BOLUS_BASE_COLS
        if extra & BOLUS_EXTENDED_EXTRA_COLS:
            raise DiaTrendParseError(
                f"{subject_id}: Bolus sheet has partial extended schema; "
                f"missing extended columns: "
                f"{sorted(BOLUS_EXTENDED_EXTRA_COLS - cols)}"
            )
        return "base"
    raise DiaTrendParseError(
        f"{subject_id}: Bolus sheet does not match base or extended schema; "
        f"missing base columns: {sorted(BOLUS_BASE_COLS - cols)}"
    )


def _ensure_datetime(frame: pd.DataFrame, sheet: str, subject_id: str) -> pd.DataFrame:
    if pd.api.types.is_datetime64_any_dtype(frame["date"]):
        return frame
    try:
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"])
    except (ValueError, TypeError) as exc:
        raise DiaTrendParseError(
            f"{subject_id}: sheet '{sheet}' has unparseable 'date' column"
        ) from exc
    return frame


def _sort_ascending(frame: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    if frame.empty or frame["date"].is_monotonic_increasing:
        return frame.reset_index(drop=True), False
    return (
        frame.sort_values("date", kind="mergesort").reset_index(drop=True),
        True,
    )


def parse_subject(path: str | Path, subject_id: str | None = None) -> SubjectData:
    """Parse one DiaTrend subject workbook.

    Parameters
    ----------
    path
        Path to the subject's ``.xlsx`` file.
    subject_id
        Identifier reported in error messages and stored on the returned
        ``SubjectData``. Defaults to the file stem (e.g. ``S001`` from
        ``S001.xlsx``).
    """
    path = Path(path)
    if subject_id is None:
        subject_id = path.stem

    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")

    if CGM_SHEET not in sheets:
        raise DiaTrendParseError(f"{subject_id}: missing required sheet '{CGM_SHEET}'")
    if BOLUS_SHEET not in sheets:
        raise DiaTrendParseError(
            f"{subject_id}: missing required sheet '{BOLUS_SHEET}'"
        )

    cgm = sheets[CGM_SHEET]
    bolus = sheets[BOLUS_SHEET]
    raw_basal = sheets.get(BASAL_SHEET)

    _validate_columns(cgm.columns, CGM_REQUIRED_COLS, CGM_SHEET, subject_id)
    schema = _classify_bolus_schema(bolus.columns, subject_id)

    cgm = _ensure_datetime(cgm, CGM_SHEET, subject_id)
    bolus = _ensure_datetime(bolus, BOLUS_SHEET, subject_id)

    cgm, cgm_reversed = _sort_ascending(cgm)
    bolus, _ = _sort_ascending(bolus)

    basal: pd.DataFrame | None = None
    if raw_basal is not None and not raw_basal.empty:
        _validate_columns(raw_basal.columns, BASAL_REQUIRED_COLS, BASAL_SHEET, subject_id)
        raw_basal = _ensure_datetime(raw_basal, BASAL_SHEET, subject_id)
        basal, _ = _sort_ascending(raw_basal)
    elif raw_basal is not None:
        basal = raw_basal.copy()

    has_basal_data = basal is not None and not basal.empty
    if schema == "extended" and has_basal_data:
        # Bolus says cohort 2 (no basal) but a basal sheet has data.
        # Per Section 6.1 the two signals always agree, so this is a hard
        # parser-side or data-corruption signal worth surfacing loudly.
        raise DiaTrendParseError(
            f"{subject_id}: extended bolus schema (cohort 2) coexists with "
            f"non-empty Basal sheet. The documented partition forbids this."
        )
    if schema == "base" and not has_basal_data:
        raise DiaTrendParseError(
            f"{subject_id}: base bolus schema (cohort 1) but no Basal sheet "
            f"data. The documented partition forbids this."
        )

    cohort: Literal[1, 2] = 2 if schema == "extended" else 1

    return SubjectData(
        subject_id=subject_id,
        cohort=cohort,
        bolus_schema=schema,
        cgm=cgm,
        bolus=bolus,
        basal=basal,
        was_reversed=cgm_reversed,
    )

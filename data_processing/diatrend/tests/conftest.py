"""Session-scoped fixtures for the DiaTrend test suite."""
from __future__ import annotations

from pathlib import Path

import pytest

from data_processing.diatrend.tests.fixtures.generate_fixtures import (
    DEFAULT_PLAN,
    FixturePlan,
    generate_all,
    subject_id,
)


@pytest.fixture(scope="session")
def diatrend_fixtures_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("diatrend_fixtures")
    generate_all(out, seed=42)
    return out


@pytest.fixture(scope="session")
def fixture_plans() -> tuple[FixturePlan, ...]:
    return DEFAULT_PLAN


@pytest.fixture(scope="session")
def fixture_subject_ids() -> list[str]:
    return [subject_id(plan) for plan in DEFAULT_PLAN]

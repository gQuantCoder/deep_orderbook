from pathlib import Path

from deep_orderbook.research_core import (
    ACTIVE_CORE_SHORT_ROUTES,
    ACTIVE_SHORT_ROUTE_NAMES,
    RouteSpec,
    completed_holdout_tags_from_summary_jsons,
    next_chronological_holdouts,
    select_files_for_date_range,
)


def test_next_chronological_holdouts_skips_completed_tags_and_returns_ordered_pair() -> None:
    available = [Path(f"/tmp/2026-04-23T{hour:02d}-00-00.parquet") for hour in [11, 12, 13, 14, 15, 16]]
    completed = {"20260423T110000", "20260423T120000", "20260423T130000", "20260423T140000"}

    picked = next_chronological_holdouts(available, completed_tags=completed, count=2)

    assert [p.name for p in picked] == [
        "2026-04-23T15-00-00.parquet",
        "2026-04-23T16-00-00.parquet",
    ]


def test_next_chronological_holdouts_returns_empty_when_nothing_new_exists() -> None:
    available = [Path("/tmp/2026-04-23T15-00-00.parquet")]
    completed = {"20260423T150000"}

    assert next_chronological_holdouts(available, completed_tags=completed, count=2) == []


def test_completed_holdout_tags_from_summary_jsons_reads_validation_rows(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text('{"validation": [{"holdout_tag": "20260423T150010"}, {"holdout_tag": "20260423T160011"}]}')

    assert completed_holdout_tags_from_summary_jsons([path]) == {"20260423T150010", "20260423T160011"}


def test_route_spec_strategy_kwargs_returns_exactly_the_six_evaluator_kwargs() -> None:
    route = RouteSpec(
        name="q90_p2_hold96",
        entry_threshold=0.42,
        exit_threshold=0.31,
        side_margin=0.07,
        persistence=2,
        cooldown=6,
        max_hold=96,
    )
    assert route.strategy_kwargs() == {
        "entry_threshold": 0.42,
        "exit_threshold": 0.31,
        "side_margin": 0.07,
        "persistence": 2,
        "cooldown": 6,
        "max_hold": 96,
    }


def test_route_spec_but_preserves_display_metadata_when_calibration_overrides_thresholds() -> None:
    catalog = RouteSpec(name="q95_p1_hold48", display_label="q95 fast", color="#4C78A8")
    calibrated = catalog.but(entry_threshold=0.9, exit_threshold=0.8, side_margin=0.05, persistence=1, cooldown=8, max_hold=48)
    assert calibrated.name == "q95_p1_hold48"
    assert calibrated.display_label == "q95 fast"
    assert calibrated.color == "#4C78A8"
    assert calibrated.entry_threshold == 0.9
    assert calibrated.persistence == 1


def test_route_spec_defaults_match_the_short_protocol() -> None:
    route = RouteSpec(name="anything")
    assert route.direction == "short"
    assert route.entry_threshold == 0.0
    assert route.persistence == 1
    assert route.display_label == ""
    assert route.color == "#888888"


def test_active_core_short_routes_catalog_is_consistent_with_legacy_name_tuple() -> None:
    assert tuple(r.name for r in ACTIVE_CORE_SHORT_ROUTES) == ACTIVE_SHORT_ROUTE_NAMES
    # Catalog entries must carry display metadata (one source of truth for dashboards)
    assert all(r.display_label for r in ACTIVE_CORE_SHORT_ROUTES)
    assert all(r.color.startswith("#") for r in ACTIVE_CORE_SHORT_ROUTES)
    # All catalog entries are short by convention
    assert all(r.direction == "short" for r in ACTIVE_CORE_SHORT_ROUTES)


def test_select_files_for_date_range_returns_only_inclusive_day_block() -> None:
    available = [
        Path("/tmp/2026-04-13T23-00-00.parquet"),
        Path("/tmp/2026-04-14T00-00-00.parquet"),
        Path("/tmp/2026-04-15T12-00-00.parquet"),
        Path("/tmp/2026-04-16T18-00-00.parquet"),
        Path("/tmp/2026-04-17T00-00-00.parquet"),
    ]

    selected = select_files_for_date_range(available, start_date="2026-04-14", end_date="2026-04-16")

    assert [p.name for p in selected] == [
        "2026-04-14T00-00-00.parquet",
        "2026-04-15T12-00-00.parquet",
        "2026-04-16T18-00-00.parquet",
    ]

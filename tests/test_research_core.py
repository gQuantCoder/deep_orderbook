from pathlib import Path

from deep_orderbook.research_core import (
    completed_holdout_tags_from_summary_jsons,
    next_chronological_holdouts,
    select_files_for_date_range,
)


def test_next_chronological_holdouts_skips_completed_tags_and_returns_ordered_pair() -> None:
    available = [
        Path(f"/tmp/2026-04-23T{hour:02d}-00-00.parquet")
        for hour in [11, 12, 13, 14, 15, 16]
    ]
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

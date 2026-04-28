from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

ACTIVE_SHORT_ROUTE_NAMES: tuple[str, ...] = (
    "q95_p1_hold48",
    "q90_p1_hold48",
    "q90_p2_hold96",
    "q90_persistence2_fast_exit",
)


def holdout_tag_from_path(path: Path | str) -> str:
    text = Path(path).stem if not isinstance(path, Path) else path.stem
    return text.replace("-", "").replace(":", "")


def next_chronological_holdouts(
    available_paths: Iterable[Path],
    *,
    completed_tags: set[str],
    count: int = 2,
) -> list[Path]:
    ordered = sorted(Path(p) for p in available_paths)
    fresh = [path for path in ordered if holdout_tag_from_path(path) not in completed_tags]
    return fresh[:count]


def select_files_for_date_range(
    available_paths: Iterable[Path],
    *,
    start_date: str,
    end_date: str,
) -> list[Path]:
    ordered = sorted(Path(p) for p in available_paths)
    return [path for path in ordered if start_date <= path.name[:10] <= end_date]


def completed_holdout_tags_from_summary_jsons(summary_paths: Iterable[Path]) -> set[str]:
    tags: set[str] = set()
    for path in summary_paths:
        payload = json.loads(Path(path).read_text())
        for row in payload.get("validation", []):
            tag = row.get("holdout_tag")
            if tag:
                tags.add(str(tag))
    return tags

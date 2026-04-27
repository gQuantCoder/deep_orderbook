"""exp63 — active-core short continuation on the next fresh chronological pair.

This is the cleaned-up canonical continuation path for the current deep_orderbook
research branch.

Principles
----------
- keep the mapper fixed at `regonly_l1_2d_h8`
- keep the recent 24-file 2026 train pool fixed
- keep short-only direction fixed
- cut active route surface area to the 4 serious contenders
- choose the next two fresh chronological holdouts automatically from completed summaries
- keep corrected friction accounting as final_pnl - trade_count * cost

Usage
-----
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. ./.venv/bin/python scripts/exp63_active_core_recent24_short_continuation.py
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from pathlib import Path

import scripts.exp54_recent24_stress_short_next_date_friction_retest as base
from deep_orderbook.research_core import (
    ACTIVE_SHORT_ROUTE_NAMES,
    completed_holdout_tags_from_summary_jsons,
    next_chronological_holdouts,
)

DATA_DIR = Path("/mnt/data/repos/gaelreinaudi/crypto")
FRICTION_COSTS = (1.0, 2.0, 5.0, 10.0)
LABEL_PREFIX = "exp63_active_core_recent24_short_continuation"
ROUTES = list(ACTIVE_SHORT_ROUTE_NAMES)
ROUTE_LABELS = {
    "q95_p1_hold48": "q95 fast",
    "q90_p1_hold48": "q90 fast",
    "q90_p2_hold96": "q90 persistence2 hold96",
    "q90_persistence2_fast_exit": "q90 persistence-fast-exit",
}
ROUTE_COLORS = {
    "q95_p1_hold48": "#4C78A8",
    "q90_p1_hold48": "#F58518",
    "q90_p2_hold96": "#72B7B2",
    "q90_persistence2_fast_exit": "#54A24B",
}
SUMMARY_PREFIXES = (
    "exp57_recent24_stress_short_0600_0700_route_falsification_corrected_friction",
    "exp58_recent24_stress_short_0800_route_continuation_corrected_friction",
    "exp59_recent24_stress_short_0910_route_continuation_corrected_friction",
    "exp60_recent24_stress_short_1112_route_continuation_corrected_friction",
    "exp62_recent24_stress_short_1314_route_continuation_corrected_friction",
    LABEL_PREFIX,
)


def _corrected_adjusted_pnl(final_pnl: float, trade_count: int) -> dict[float, float]:
    return {cost: float(final_pnl - (trade_count * cost)) for cost in FRICTION_COSTS}


def _summary_json_paths() -> list[Path]:
    paths: list[Path] = []
    for prefix in SUMMARY_PREFIXES:
        for path in sorted(base.RESULTS_DIR.glob(f"{prefix}_*.json")):
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if "validation" in payload and "aggregate_results" in payload:
                paths.append(path)
    deduped = {path.resolve(): path for path in paths}
    return sorted(deduped.values())


def _candidate_holdouts() -> list[Path]:
    return sorted(list(DATA_DIR.glob("2026-04-23T1[1-9]-*.parquet")) + list(DATA_DIR.glob("2026-04-23T2[0-3]-*.parquet")))


async def main() -> None:
    overall_start = time.perf_counter()
    timestamp_utc, stamp = base._utc_stamp()
    git_commit = base._git_commit()

    completed_tags = completed_holdout_tags_from_summary_jsons(_summary_json_paths())
    validation_holdouts = next_chronological_holdouts(_candidate_holdouts(), completed_tags=completed_tags, count=2)
    if len(validation_holdouts) < 2:
        raise RuntimeError(
            f"Need 2 fresh chronological holdouts after completed_tags={sorted(completed_tags)}; found {validation_holdouts}"
        )

    base.ROUTES = ROUTES
    base.ROUTE_LABELS = ROUTE_LABELS
    base.ROUTE_COLORS = ROUTE_COLORS
    base.LABEL_PREFIX = LABEL_PREFIX

    validation_rows = []
    holdout_runtimes: dict[str, float] = {}
    for holdout in validation_holdouts:
        start = time.perf_counter()
        row = await base._run_one_holdout(holdout)
        for route in ROUTES:
            rr = row["route_results"][route]
            rr["adjusted_pnl_corrected"] = _corrected_adjusted_pnl(rr["final_pnl"], rr["trade_count"])
        validation_rows.append(row)
        holdout_runtimes[holdout.name] = round(time.perf_counter() - start, 3)

    route_totals = {route: float(sum(row["route_results"][route]["final_pnl"] for row in validation_rows)) for route in ROUTES}
    route_trade_totals = {route: int(sum(row["route_results"][route]["trade_count"] for row in validation_rows)) for route in ROUTES}
    route_win_counts = {route: sum(1 for row in validation_rows if row["best_route"] == route) for route in ROUTES}
    route_mean_precision = {route: float(statistics.mean(row["route_results"][route]["precision"] for row in validation_rows)) for route in ROUTES}
    route_mean_recall = {route: float(statistics.mean(row["route_results"][route]["recall"] for row in validation_rows)) for route in ROUTES}
    route_mean_f1 = {route: float(statistics.mean(row["route_results"][route]["f1"] for row in validation_rows)) for route in ROUTES}
    route_mean_rmse_ratio = {route: float(statistics.mean(row["route_results"][route]["rmse_ratio"] for row in validation_rows)) for route in ROUTES}
    route_mean_rmse = {route: float(statistics.mean(row["route_results"][route]["rmse"] for row in validation_rows)) for route in ROUTES}
    route_friction_totals = {
        route: {
            cost: float(sum(row["route_results"][route]["adjusted_pnl_corrected"][cost] for row in validation_rows))
            for cost in FRICTION_COSTS
        }
        for route in ROUTES
    }

    champion_route = max(
        ROUTES,
        key=lambda route: (
            route_friction_totals[route][5.0],
            route_totals[route],
            -route_trade_totals[route],
            route_mean_precision[route],
            -abs(route_mean_rmse_ratio[route] - 1.0),
        ),
    )
    champion_payload = {
        "route": champion_route,
        "total_pnl": route_totals[champion_route],
        "total_adjusted_pnl": route_friction_totals[champion_route],
        "win_count": route_win_counts[champion_route],
        "trade_count": route_trade_totals[champion_route],
        "mean_precision": route_mean_precision[champion_route],
        "mean_recall": route_mean_recall[champion_route],
        "mean_f1": route_mean_f1[champion_route],
        "mean_rmse": route_mean_rmse[champion_route],
        "mean_rmse_ratio": route_mean_rmse_ratio[champion_route],
    }
    decision = "not promising yet"

    summary_png = base._summary_picture(
        validation_rows,
        route_totals,
        route_friction_totals,
        base.PICTURES_DIR / f"{LABEL_PREFIX}_route_and_friction_summary_{stamp}.png",
    )

    summary = {
        "experiment": LABEL_PREFIX,
        "timestamp_utc": timestamp_utc,
        "git_commit": git_commit,
        "device": "cpu",
        "hypothesis": "If the active-core route set captures the real execution contenders, continuing only these four families on the next fresh chronological pair should preserve truth while reducing route-noise and experiment sprawl.",
        "expected_failure_mode": "If previous reactivation was mostly threshold noise, even the reduced active-core route set should fail under cost-5 on the next pair or leadership should still thrash without any survivor.",
        "mutation": "Keep mapper/train pool/stress filter frozen; cut active route surface to q95_p1_hold48, q90_p1_hold48, q90_p2_hold96, q90_persistence2_fast_exit; auto-select the next two fresh chronological holdouts and score with corrected friction.",
        "cleanup": {
            "active_routes": ROUTES,
            "route_count_reduced_from": 7,
            "route_count_reduced_to": len(ROUTES),
            "why": "Cut obvious churn traps and route bloat; keep one strict fast, one medium fast, one persistence, one persistence-fast-exit foil.",
        },
        "validation": validation_rows,
        "data_scale": {
            "train_file_count": len(base.DEFAULT_TRAIN_FILES),
            "train_files": [str(path) for path in base.DEFAULT_TRAIN_FILES],
            "validation_holdout_count": len(validation_holdouts),
            "validation_holdouts": [str(path) for path in validation_holdouts],
            "completed_holdout_tags_seen": sorted(completed_tags),
            "total_unique_2026_files_touched": len(base.DEFAULT_TRAIN_FILES) + len(validation_holdouts),
            "capped_vs_uncapped": "uncapped (max_windows_per_file=null)",
            "serious_vs_screening": "serious fresh multi-holdout continuation on cleaned active route surface",
        },
        "aggregate_results": {
            "route_totals": route_totals,
            "route_trade_totals": route_trade_totals,
            "route_win_counts": route_win_counts,
            "route_mean_precision": route_mean_precision,
            "route_mean_recall": route_mean_recall,
            "route_mean_f1": route_mean_f1,
            "route_mean_rmse": route_mean_rmse,
            "route_mean_rmse_ratio": route_mean_rmse_ratio,
            "route_friction_totals": route_friction_totals,
            "champion": champion_payload,
            "decision": decision,
            "friction_accounting_note": "Corrected locally from final_pnl - trade_count*cost.",
        },
        "artifacts": {
            "route_and_friction_summary_png": str(summary_png),
            "best_holdout_pngs": [row["best_picture"] for row in validation_rows],
        },
        "runtime_seconds": round(time.perf_counter() - overall_start, 3),
        "holdout_runtimes_seconds": holdout_runtimes,
    }

    result_path = base.RESULTS_DIR / f"{LABEL_PREFIX}_{stamp}.json"
    result_path.write_text(json.dumps(summary, indent=2))
    note_path = base.NOTES_DIR / f"{LABEL_PREFIX}_{stamp}.md"
    note_path.write_text(
        "\n".join(
            [
                f"# {LABEL_PREFIX}",
                "",
                f"- timestamp: {timestamp_utc}",
                f"- active_routes: {json.dumps(ROUTES)}",
                f"- validation_holdouts: {json.dumps([str(path) for path in validation_holdouts])}",
                f"- champion_route: {champion_route}",
                f"- champion_payload: {json.dumps(champion_payload, sort_keys=True)}",
                f"- route_totals: {json.dumps(route_totals, sort_keys=True)}",
                f"- route_friction_totals: {json.dumps(route_friction_totals, sort_keys=True)}",
                f"- decision: `{decision}`",
                f"- summary_png: {summary_png}",
                f"- result_json: {result_path}",
            ]
        )
        + "\n"
    )

    row_id = base.register_experiment_run(
        experiment=LABEL_PREFIX,
        variant=f"regonly_l1_2d_h8__short__{champion_route}",
        timestamp_utc=timestamp_utc,
        git_commit=git_commit,
        symbol="BTC-USD",
        cadence="1000ms",
        look_ahead=64,
        model="regonly_l1_2d_h8",
        result_json_path=result_path,
        picture_path=summary_png,
        metrics={
            "precision": champion_payload["mean_precision"],
            "recall": champion_payload["mean_recall"],
            "f1": champion_payload["mean_f1"],
            "rmse": champion_payload["mean_rmse"],
            "rmse_ratio": champion_payload["mean_rmse_ratio"],
            "final_pnl": champion_payload["total_pnl"],
            "trade_count": champion_payload["trade_count"],
            "adjusted_pnl_5": champion_payload["total_adjusted_pnl"][5.0],
            **route_totals,
        },
        notes=f"notes_file={note_path}",
    )
    summary["experiment_db_row_id"] = row_id
    result_path.write_text(json.dumps(summary, indent=2))

    journal_text = f"""## {timestamp_utc}
- **Continuity**: active-core cleanup cut route surface from 7 to {len(ROUTES)} and switched to auto-selected next holdouts after completed tags `{sorted(completed_tags)}`.
- **Hypothesis**: if route bloat was obscuring the real execution picture, the reduced active-core set should preserve signal truth on the next pair while making route leadership easier to interpret.
- **What changed**: cleaned the active research path only; kept mapper/train pool/stress filter fixed and validated only `{validation_holdouts[0].name}` and `{validation_holdouts[1].name}`.
- **Main result**: corrected champion `{champion_route}` raw pnl `{champion_payload['total_pnl']:.2f}`, cost-5 `{champion_payload['total_adjusted_pnl'][5.0]:.2f}`, trades `{champion_payload['trade_count']}`, precision `{champion_payload['mean_precision']:.4f}`, f1 `{champion_payload['mean_f1']:.4f}`, rmse_ratio `{champion_payload['mean_rmse_ratio']:.4f}`.
- **Decision**: `{decision}`.
- **Exact next mutation**: if more fresh `2026-04-23` files exist, continue the same active-core fixed-route falsification forward chronologically again before any selector or mapper change.
"""
    base._append_text(base.HOURLY_JOURNAL, journal_text)
    base._append_text(
        base.LAB_LOG,
        f"## {timestamp_utc} exp63 active-core continuation\n\nStatus: DONE\n\n- Active routes: `{ROUTES}`\n- Validation holdouts: `{[str(path) for path in validation_holdouts]}`\n- Champion: `{champion_route}` cost5 `{champion_payload['total_adjusted_pnl'][5.0]:.5f}`\n- Decision: `{decision}`\n",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

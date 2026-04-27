"""exp60 — freshest 11:00/12:00 continuation with corrected-friction fixed-route falsification.

Hypothesis
----------
If the recent stress-conditioned short mapper is still alive after the raw-positive but cost-5-dead 09:00/10:00 checkpoint,
then extending the same corrected-friction fixed-route comparison to the next two chronological fresh holdouts (11:00 and 12:00)
should show whether the branch reactivates on a broader fresh intraday slice or whether exp59 was just another local wobble.

Single research mutation vs exp59:
- keep mapper family fixed at `regonly_l1_2d_h8`
- keep cadence / geometry fixed at 1000ms, rolling=1024, look_ahead=64
- keep observable present-time stress filtering fixed
- keep the recent 24-file 2026 train pool fixed (2026-04-16..2026-04-21 at 00/06/12/18)
- keep the broadened fixed execution family fixed
- advance validation to the next two fresh 2026 holdouts: 2026-04-23T11-00-01 and 2026-04-23T12-00-10
- keep corrected friction accounting as final_pnl - trade_count * cost

Practical note
--------------
Two new post-exp59 2026 parquets were available at run time, so this mutation is a fresher multi-holdout falsification step
rather than a single-file checkpoint.

Usage
-----
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. ./.venv/bin/python scripts/exp60_recent24_stress_short_1112_route_continuation_corrected_friction.py
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from pathlib import Path

import scripts.exp54_recent24_stress_short_next_date_friction_retest as base

DATA_DIR = Path("/mnt/data/repos/gaelreinaudi/crypto")
ROUTES = [
    "q80_p1_hold48",
    "q95_p1_hold48",
    "q90_p1_hold48",
    "q90_persistence2_fast_exit",
    "q95_p2_hold96",
    "q90_p2_hold96",
    "q80_persistence2_wide_margin",
]
ROUTE_LABELS = {
    "q80_p1_hold48": "q80 fast",
    "q95_p1_hold48": "q95 fast",
    "q90_p1_hold48": "q90 fast",
    "q90_persistence2_fast_exit": "q90 persistence-fast-exit",
    "q95_p2_hold96": "q95 persistence2 hold96",
    "q90_p2_hold96": "q90 persistence2 hold96",
    "q80_persistence2_wide_margin": "q80 persistence-wide-margin",
}
ROUTE_COLORS = {
    "q80_p1_hold48": "#B279A2",
    "q95_p1_hold48": "#4C78A8",
    "q90_p1_hold48": "#F58518",
    "q90_persistence2_fast_exit": "#54A24B",
    "q95_p2_hold96": "#E45756",
    "q90_p2_hold96": "#72B7B2",
    "q80_persistence2_wide_margin": "#FF9DA6",
}
FRICTION_COSTS = (1.0, 2.0, 5.0, 10.0)
LABEL_PREFIX = "exp60_recent24_stress_short_1112_route_continuation_corrected_friction"
VALIDATION_HOLDOUTS = [
    DATA_DIR / "2026-04-23T11-00-01.parquet",
    DATA_DIR / "2026-04-23T12-00-10.parquet",
]


def _corrected_adjusted_pnl(final_pnl: float, trade_count: int) -> dict[float, float]:
    return {cost: float(final_pnl - (trade_count * cost)) for cost in FRICTION_COSTS}


async def main() -> None:
    overall_start = time.perf_counter()
    timestamp_utc, stamp = base._utc_stamp()
    git_commit = base._git_commit()
    experiment = LABEL_PREFIX

    base.ROUTES = ROUTES
    base.ROUTE_LABELS = ROUTE_LABELS
    base.ROUTE_COLORS = ROUTE_COLORS
    base.LABEL_PREFIX = LABEL_PREFIX

    validation_rows = []
    holdout_runtimes: dict[str, float] = {}
    for holdout in VALIDATION_HOLDOUTS:
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
    positive_raw_routes = [route for route in ROUTES if route_totals[route] > 0]
    positive_friction_routes = [route for route in ROUTES if route_friction_totals[route][5.0] > 0]
    overall_decision = "not promising yet"

    summary_png = base._summary_picture(
        validation_rows,
        route_totals,
        route_friction_totals,
        base.PICTURES_DIR / f"{experiment}_route_and_friction_summary_{stamp}.png",
    )

    summary = {
        "experiment": experiment,
        "timestamp_utc": timestamp_utc,
        "git_commit": git_commit,
        "device": "cpu",
        "hypothesis": "If the recent stress-conditioned short mapper is still alive after the raw-positive but cost-5-dead 09:00/10:00 checkpoint, extending the same corrected-friction fixed-route comparison to the next two chronological fresh holdouts (11:00 and 12:00) should show whether the branch reactivates on a broader fresh intraday slice or whether exp59 was just another local wobble.",
        "expected_failure_mode": "If the branch is stale, the same fixed-route family should stay friction-dead or leadership should flip randomly across 11:00/12:00 without any route surviving cost-5 consistently.",
        "mutation": "Keep mapper/train pool/stress filter and broadened route set frozen; advance validation to 2026-04-23T11-00-01 and 2026-04-23T12-00-10, and keep corrected friction accounting as final_pnl - trade_count*cost.",
        "validation": validation_rows,
        "data_scale": {
            "train_file_count": len(base.DEFAULT_TRAIN_FILES),
            "train_files": [str(path) for path in base.DEFAULT_TRAIN_FILES],
            "validation_holdout_count": len(VALIDATION_HOLDOUTS),
            "validation_holdouts": [str(path) for path in VALIDATION_HOLDOUTS],
            "total_unique_2026_files_touched": len(base.DEFAULT_TRAIN_FILES) + len(VALIDATION_HOLDOUTS),
            "capped_vs_uncapped": "uncapped (max_windows_per_file=null)",
            "serious_vs_screening": "serious fresh multi-holdout falsification on the next available 2026 continuation",
            "fresh_2026_backfill_note": "Two new post-exp59 2026 parquets were available, so no older backfill was used in this run.",
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
            "positive_raw_route_count": len(positive_raw_routes),
            "positive_friction_route_count_at_5": len(positive_friction_routes),
            "decision": overall_decision,
            "friction_accounting_note": "Corrected locally from final_pnl - trade_count*cost because exp54-style route file lookup did not carry friction_table_top10 into per-route payloads.",
        },
        "artifacts": {
            "route_and_friction_summary_png": str(summary_png),
            "best_holdout_pngs": [row["best_picture"] for row in validation_rows],
        },
        "runtime_seconds": round(time.perf_counter() - overall_start, 3),
        "holdout_runtimes_seconds": holdout_runtimes,
    }

    result_path = base.RESULTS_DIR / f"{experiment}_{stamp}.json"
    result_path.write_text(json.dumps(summary, indent=2))

    note_path = base.NOTES_DIR / f"{experiment}_{stamp}.md"
    note_lines = [
        f"# {experiment}",
        "",
        f"- timestamp: {timestamp_utc}",
        "- hypothesis: advancing the corrected-friction fixed-route continuation to the freshest available 11:00/12:00 holdouts should reveal whether the branch reactivates after the raw-positive but cost-5-dead 09:00/10:00 checkpoint",
        f"- champion_route: {champion_route}",
        f"- champion_payload: {json.dumps(champion_payload, sort_keys=True)}",
        f"- route_totals: {json.dumps(route_totals, sort_keys=True)}",
        f"- route_friction_totals: {json.dumps(route_friction_totals, sort_keys=True)}",
        f"- route_win_counts: {json.dumps(route_win_counts, sort_keys=True)}",
        f"- decision: `{overall_decision}`",
        f"- summary_png: {summary_png}",
        f"- result_json: {result_path}",
    ]
    for row in validation_rows:
        note_lines.extend(
            [
                f"- {row['holdout_tag']}: features=(impulse={row['feature_book_impulse_top5_mean']:.6f}, range_bps={row['feature_range_bps_top5_mean']:.6f}, score={row['feature_score_top5_mean']:.6f}) best_raw={row['best_route']} best_raw_pnl={row['best_pnl']:.5f}",
                f"  - best_fixed_png: {row['best_picture']}",
            ]
        )
    note_path.write_text("\n".join(note_lines) + "\n")

    row_id = base.register_experiment_run(
        experiment=experiment,
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
            "route_win_count": champion_payload["win_count"],
            "adjusted_pnl_1": champion_payload["total_adjusted_pnl"][1.0],
            "adjusted_pnl_2": champion_payload["total_adjusted_pnl"][2.0],
            "adjusted_pnl_5": champion_payload["total_adjusted_pnl"][5.0],
            "adjusted_pnl_10": champion_payload["total_adjusted_pnl"][10.0],
            **route_totals,
        },
        notes=f"notes_file={note_path}",
    )
    summary["experiment_db_row_id"] = row_id
    result_path.write_text(json.dumps(summary, indent=2))

    route_totals_text = ", ".join(f"{route}={route_totals[route]:.2f}" for route in ROUTES)
    friction_text = "; ".join(
        f"{route}: c1={route_friction_totals[route][1.0]:.2f}, c2={route_friction_totals[route][2.0]:.2f}, c5={route_friction_totals[route][5.0]:.2f}, c10={route_friction_totals[route][10.0]:.2f}"
        for route in ROUTES
    )
    per_holdout_text = '; '.join(
        f"{row['holdout_tag']} (impulse={row['feature_book_impulse_top5_mean']:.4f}, range={row['feature_range_bps_top5_mean']:.2f}) -> best raw {row['best_route']}={row['best_pnl']:.2f}"
        for row in validation_rows
    )
    journal_text = f"""## {timestamp_utc}
- **Continuity**: picked up from exp59. The queue wanted the same corrected-friction fixed-route comparison extended forward chronologically if more fresh 2026 appeared, and two new parquets did appear: `2026-04-23T11-00-01` and `2026-04-23T12-00-10`. Ran that freshest honest multi-holdout continuation instead of changing selector logic or retraining the mapper again.
- **Hypothesis**: if the frozen stress-conditioned short branch is still alive after the raw-positive but cost-5-dead 09:00/10:00 checkpoint, extending the same corrected-friction fixed-route comparison to the next two chronological fresh holdouts (11:00 and 12:00) should show whether the branch reactivates on a broader fresh intraday slice or whether exp59 was just another local wobble.
- **What changed**: validation scope only. Kept the broadened fixed-route family and corrected friction accounting unchanged; advanced the holdout set from `2026-04-23T09-00-00` and `2026-04-23T10-00-11` to the next two chronological fresh files `2026-04-23T11-00-01` and `2026-04-23T12-00-10`.
- **What stayed fixed**: recent 24-file 2026 train pool (`2026-04-16`..`2026-04-21` at `00/06/12/18`), frozen `regonly_l1_2d_h8` branch, short-only direction, `1000ms`, `rolling=1024`, `look_ahead=64`, observable stress filter, uncapped loading.
- **Runtime / hardware**: CPU, total `{summary['runtime_seconds']:.3f}s`; per-holdout runtimes `{holdout_runtimes}`. Serious fresh multi-holdout continuation on the next available 2026 files.
- **Data scale**:
  - 2026 coverage: 24 train files + holdouts `2026-04-23T11-00-01` and `2026-04-23T12-00-10`.
  - Total unique 2026 files touched across this mutation: {summary['data_scale']['total_unique_2026_files_touched']}.
  - Capped vs uncapped: uncapped (`max_windows_per_file=null`).
  - Screening vs serious: serious fresh multi-holdout falsification on the next available 2026 continuation.
- **Main result**:
  - raw route totals `{route_totals_text}`.
  - corrected friction totals `{friction_text}`.
  - corrected champion `{champion_route}` with raw pnl `{champion_payload['total_pnl']:.2f}`, cost-5 pnl `{champion_payload['total_adjusted_pnl'][5.0]:.2f}`, trades `{champion_payload['trade_count']}`, precision `{champion_payload['mean_precision']:.4f}`, f1 `{champion_payload['mean_f1']:.4f}`, rmse_ratio `{champion_payload['mean_rmse_ratio']:.4f}`.
  - per-holdout: {per_holdout_text}.
- **Pictures**:
  - route/friction summary: `{summary_png}`
  - validation diagnostic 1: `{validation_rows[0]['best_picture']}`
  - validation diagnostic 2: `{validation_rows[-1]['best_picture']}`
- **Interpretation**: this remains a falsification-first step and directly tests whether exp59's execution fragility was temporary or persistent. It can still demote routes decisively; promotion would still need later-date confirmation.
- **Decision**: `{overall_decision}`.
- **Exact next mutation**: keep the mapper frozen; if more `2026-04-23` intraday files appear, extend the same corrected-friction fixed-route comparison forward chronologically again before any selector or mapper change. If fresh 2026 stalls, state the gap explicitly and then test a simple present-time regime gate between the surviving fast route and the least-bad persistence route on later recent-2026 blocks.
"""
    base._append_text(base.HOURLY_JOURNAL, journal_text)

    lab_text = f"""## {timestamp_utc} exp60 fresh 11:00/12:00 route continuation with corrected friction

Status: DONE

### Mutation from prior run
- Kept the recent 24-file 2026 stress-conditioned short mapper fixed.
- Kept short-only fixed-route evaluation and corrected-friction accounting fixed.
- Changed one research layer only: advanced the freshest available holdout set from the 09:00/10:00 checkpoint to the next two chronological fresh files (11:00 and 12:00) with the same broadened route family.

### Hypothesis
- If the branch is real, it should reactivate on the next two chronological fresh holdouts rather than staying friction-dead after the raw-positive but cost-5-dead 09:00/10:00 checkpoint.

### Artifacts
- JSON: `{result_path}`
- Summary picture: `{summary_png}`
- Notes: `{note_path}`
- DB row: `experiment_runs.id = {row_id}`

### Outcome
- Raw route totals: `{route_totals_text}`
- Corrected friction totals: `{friction_text}`
- Champion: `{champion_route}` with raw pnl `{champion_payload['total_pnl']:.5f}`, cost-5 pnl `{champion_payload['total_adjusted_pnl'][5.0]:.5f}`, trades `{champion_payload['trade_count']}`
- Decision: `{overall_decision}`

### Interpretation
- This is another freshness/falsification checkpoint. It matters only insofar as it shows whether the frozen execution family still expresses any tradable short branch across the next two fresh 2026 hours.
"""
    base._append_text(base.LAB_LOG, lab_text)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

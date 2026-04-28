"""exp64 — broad-day recent-2026 training with active-core short continuation.

Goal
----
Stop pretending the branch is trained on only a few scattered hours.
This run broadens train coverage to full contiguous recent days, while keeping
validation chronological and route search narrow.

Protocol
--------
- mapper fixed at `regonly_l1_2d_h8`
- short-only
- active route core only:
  q95_p1_hold48, q90_p1_hold48, q90_p2_hold96, q90_persistence2_fast_exit
- train coverage: every hourly parquet from 2026-04-14 through 2026-04-20
- validation: first two chronological holdouts from 2026-04-21
- train windows are capped per file for machine fit; test windows are uncapped
- corrected friction accounting = final_pnl - trade_count * cost

Usage
-----
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. ./.venv/bin/python scripts/exp64_broad_day_recent2026_active_core_short.py
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from pathlib import Path

import scripts.exp54_recent24_stress_short_next_date_friction_retest as base
from deep_orderbook.research_core import ACTIVE_SHORT_ROUTE_NAMES, select_files_for_date_range
from scripts.exp22_trigger_sweep import main as exp22_main

DATA_DIR = Path("/mnt/data/repos/gaelreinaudi/crypto")
FRICTION_COSTS = (1.0, 2.0, 5.0, 10.0)
LABEL_PREFIX = "exp64_broad_day_recent2026_active_core_short"
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
TRAIN_START = "2026-04-14"
TRAIN_END = "2026-04-20"
VALIDATION_DAY = "2026-04-21"
TRAIN_SAMPLE_CAP_PER_FILE = 256


def _corrected_adjusted_pnl(final_pnl: float, trade_count: int) -> dict[float, float]:
    return {cost: float(final_pnl - (trade_count * cost)) for cost in FRICTION_COSTS}


def _train_files() -> list[Path]:
    return select_files_for_date_range(DATA_DIR.glob("2026-*.parquet"), start_date=TRAIN_START, end_date=TRAIN_END)


def _validation_holdouts() -> list[Path]:
    day_files = select_files_for_date_range(DATA_DIR.glob("2026-*.parquet"), start_date=VALIDATION_DAY, end_date=VALIDATION_DAY)
    return day_files[:2]


async def _run_one_holdout(holdout: Path, train_files: list[Path]) -> dict:
    holdout_tag = holdout.stem.replace(":", "").replace("-", "")
    label = f"{LABEL_PREFIX}_{holdout_tag}"
    await exp22_main(
        variants=["regonly_l1_2d_h8"],
        train_files=train_files,
        test_files=[holdout],
        label=label,
        directions=["short"],
        n_samples_per_file=None,
        load_every="1000ms",
        shaper_config_override=base.SHAPER,
        strategy_names=ROUTES,
        train_n_samples_per_file=TRAIN_SAMPLE_CAP_PER_FILE,
        test_n_samples_per_file=None,
    )
    route_payloads: dict[str, dict] = {}
    feature_impulse = None
    feature_range = None
    feature_score = None
    for route in ROUTES:
        path = base._find_latest_result(holdout_tag, route)
        data = base._load_json(path)
        route_payloads[route] = base._route_result_view(data)
        if feature_impulse is None:
            feature_impulse, feature_range, feature_score = base._top5_feature_means(data)
    best_route, best_payload = max(route_payloads.items(), key=lambda kv: kv[1]["final_pnl"])
    return {
        "label": label,
        "holdout_tag": holdout_tag,
        "holdout_path": str(holdout),
        "feature_book_impulse_top5_mean": feature_impulse,
        "feature_range_bps_top5_mean": feature_range,
        "feature_score_top5_mean": feature_score,
        "route_results": route_payloads,
        "best_route": best_route,
        "best_pnl": float(best_payload["final_pnl"]),
        "best_picture": best_payload["artifacts"]["dashboard_fixed_png"],
    }


async def main() -> None:
    overall_start = time.perf_counter()
    timestamp_utc, stamp = base._utc_stamp()
    git_commit = base._git_commit()
    train_files = _train_files()
    validation_holdouts = _validation_holdouts()
    if len(validation_holdouts) < 2:
        raise RuntimeError(f"Need at least 2 holdouts on {VALIDATION_DAY}, found {validation_holdouts}")

    base.ROUTES = ROUTES
    base.ROUTE_LABELS = ROUTE_LABELS
    base.ROUTE_COLORS = ROUTE_COLORS
    base.LABEL_PREFIX = LABEL_PREFIX

    validation_rows = []
    holdout_runtimes = {}
    for holdout in validation_holdouts:
        start = time.perf_counter()
        row = await _run_one_holdout(holdout, train_files)
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
        "hypothesis": "If the short branch is real beyond a few scattered hours, it should still survive on the first contiguous post-train holdout pair when trained across full recent days with only the active-core routes.",
        "expected_failure_mode": "If prior wins were mostly sparse-hour luck, broadening train coverage to full contiguous days should wash them out and the active-core routes should fail even on the first validation pair.",
        "mutation": "Broaden train coverage from scattered 24-file slices to every hourly parquet from 2026-04-14..2026-04-20, keep only the active-core routes, cap train windows per file for fit, and validate on the first two chronological 2026-04-21 holdouts.",
        "data_protocol": {
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "train_file_count": len(train_files),
            "validation_day": VALIDATION_DAY,
            "validation_holdout_count": len(validation_holdouts),
            "train_sample_cap_per_file": TRAIN_SAMPLE_CAP_PER_FILE,
            "test_sample_cap_per_file": None,
            "coverage_note": "Broad-day train coverage with capped train windows for machine fit; uncapped holdout evaluation.",
        },
        "validation": validation_rows,
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
                f"- train_files: {len(train_files)} from {TRAIN_START}..{TRAIN_END}",
                f"- validation_holdouts: {json.dumps([str(path) for path in validation_holdouts])}",
                f"- train_cap_per_file: {TRAIN_SAMPLE_CAP_PER_FILE}",
                f"- champion_route: {champion_route}",
                f"- champion_payload: {json.dumps(champion_payload, sort_keys=True)}",
                f"- route_totals: {json.dumps(route_totals, sort_keys=True)}",
                f"- route_friction_totals: {json.dumps(route_friction_totals, sort_keys=True)}",
                f"- decision: `{decision}`",
                f"- summary_png: {summary_png}",
                f"- result_json: {result_path}",
            ]
        ) + "\n"
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
- **Continuity**: protocol upgrade from scattered-hour train slices to broad-day train coverage. Used all hourly files from `{TRAIN_START}`..`{TRAIN_END}` ({len(train_files)} files), capped train windows per file at `{TRAIN_SAMPLE_CAP_PER_FILE}` for fit, and validated on the first two chronological `{VALIDATION_DAY}` holdouts.
- **Hypothesis**: if the branch is real beyond a few scattered hours, it should still survive on the first contiguous post-train holdout pair under the active-core route set.
- **Main result**: corrected champion `{champion_route}` raw pnl `{champion_payload['total_pnl']:.2f}`, cost-5 `{champion_payload['total_adjusted_pnl'][5.0]:.2f}`, trades `{champion_payload['trade_count']}`, precision `{champion_payload['mean_precision']:.4f}`, f1 `{champion_payload['mean_f1']:.4f}`, rmse_ratio `{champion_payload['mean_rmse_ratio']:.4f}`.
- **Decision**: `{decision}`.
- **Exact next mutation**: keep the active-core route set fixed and continue through the next contiguous `{VALIDATION_DAY}` holdouts before any selector or mapper change.
"""
    base._append_text(base.HOURLY_JOURNAL, journal_text)
    base._append_text(
        base.LAB_LOG,
        f"## {timestamp_utc} exp64 broad-day recent-2026 active-core short\n\nStatus: DONE\n\n- Train coverage: `{TRAIN_START}`..`{TRAIN_END}` ({len(train_files)} hourly files)\n- Validation holdouts: `{[str(path) for path in validation_holdouts]}`\n- Train cap per file: `{TRAIN_SAMPLE_CAP_PER_FILE}`\n- Champion: `{champion_route}` cost5 `{champion_payload['total_adjusted_pnl'][5.0]:.5f}`\n- Decision: `{decision}`\n",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

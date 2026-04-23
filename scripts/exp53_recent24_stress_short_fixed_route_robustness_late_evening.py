"""exp53 — fixed-route robustness comparison on adjacent later-evening holdouts.

Hypothesis
----------
If the frozen stress-conditioned short mapper branch is still real but selector geometry is overfit,
then adjacent later-evening holdouts should reveal one least-fragile fixed execution family.
That family should stay competitive across multiple holdouts under honest friction while the others
show clearer hour-to-hour collapse.

Single main mutation vs exp52:
- keep mapper family fixed at `regonly_l1_2d_h8`
- keep cadence / geometry fixed at 1000ms, rolling=1024, look_ahead=64
- keep observable present-time stress filtering fixed
- keep the recent 24-file 2026 train pool fixed (2026-04-16..2026-04-21 at 00/06/12/18)
- stop selector refits entirely
- mutate only execution evaluation by running fixed-route robustness comparisons on adjacent later-evening holdouts:
  2026-04-22T20:00:11, 21:00:01, 22:00:00, and 23:00:01

Usage
-----
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. ./.venv/bin/python scripts/exp53_recent24_stress_short_fixed_route_robustness_late_evening.py
"""

from __future__ import annotations

import asyncio
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from deep_orderbook.experiment_tracking import register_experiment_run
from scripts.exp16_batch_h64_tcn import DEFAULT_SHAPER_CONFIG_2026
from scripts.exp22_trigger_sweep import main as exp22_main

DATA_DIR = Path("/mnt/data/repos/gaelreinaudi/crypto")
RESULTS_DIR = Path("experiments/results")
PICTURES_DIR = Path("experiments/pictures")
NOTES_DIR = Path("experiments/notes")
HOURLY_JOURNAL = Path("experiments/hourly_journal.md")
LAB_LOG = Path("experiments/image_prediction_lab_log.md")

DEFAULT_TRAIN_FILES = [
    DATA_DIR / "2026-04-16T00-00-00.parquet",
    DATA_DIR / "2026-04-16T06-00-01.parquet",
    DATA_DIR / "2026-04-16T12-00-01.parquet",
    DATA_DIR / "2026-04-16T18-00-01.parquet",
    DATA_DIR / "2026-04-17T00-00-01.parquet",
    DATA_DIR / "2026-04-17T06-00-01.parquet",
    DATA_DIR / "2026-04-17T12-00-00.parquet",
    DATA_DIR / "2026-04-17T18-00-11.parquet",
    DATA_DIR / "2026-04-18T00-00-01.parquet",
    DATA_DIR / "2026-04-18T06-00-00.parquet",
    DATA_DIR / "2026-04-18T12-00-00.parquet",
    DATA_DIR / "2026-04-18T18-00-00.parquet",
    DATA_DIR / "2026-04-19T00-00-00.parquet",
    DATA_DIR / "2026-04-19T06-00-00.parquet",
    DATA_DIR / "2026-04-19T12-00-00.parquet",
    DATA_DIR / "2026-04-19T18-00-00.parquet",
    DATA_DIR / "2026-04-20T00-00-04.parquet",
    DATA_DIR / "2026-04-20T06-00-00.parquet",
    DATA_DIR / "2026-04-20T12-00-00.parquet",
    DATA_DIR / "2026-04-20T18-00-01.parquet",
    DATA_DIR / "2026-04-21T00-00-00.parquet",
    DATA_DIR / "2026-04-21T06-00-00.parquet",
    DATA_DIR / "2026-04-21T12-00-00.parquet",
    DATA_DIR / "2026-04-21T18-00-10.parquet",
]

VALIDATION_HOLDOUTS = [
    DATA_DIR / "2026-04-22T20-00-11.parquet",
    DATA_DIR / "2026-04-22T21-00-01.parquet",
    DATA_DIR / "2026-04-22T22-00-00.parquet",
    DATA_DIR / "2026-04-22T23-00-01.parquet",
]

ROUTES = [
    "q95_p1_hold48",
    "q90_p1_hold48",
    "q90_persistence2_fast_exit",
    "q95_p2_hold96",
    "q90_p2_hold96",
]

ROUTE_LABELS = {
    "q95_p1_hold48": "q95 fast",
    "q90_p1_hold48": "q90 fast",
    "q90_persistence2_fast_exit": "q90 persistence-fast-exit",
    "q95_p2_hold96": "q95 persistence2 hold96",
    "q90_p2_hold96": "q90 persistence2 hold96",
}

ROUTE_COLORS = {
    "q95_p1_hold48": "#4C78A8",
    "q90_p1_hold48": "#F58518",
    "q90_persistence2_fast_exit": "#54A24B",
    "q95_p2_hold96": "#E45756",
    "q90_p2_hold96": "#72B7B2",
}

LABEL_PREFIX = "exp53_recent24_stress_short_fixed_route_robustness_late_evening"

SHAPER = DEFAULT_SHAPER_CONFIG_2026.but(
    rolling_window_size=1024,
    look_ahead=64,
    window_stride=4,
)


def _utc_stamp() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M:%SZ"), now.strftime("%Y%m%dT%H%M%SZ")


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        return out or None
    except Exception:
        return None


def _append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        if path.exists() and path.stat().st_size > 0:
            fh.write("\n")
        fh.write(text.rstrip() + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _find_latest_result(holdout_tag: str, strategy_name: str) -> Path:
    matches = sorted(RESULTS_DIR.glob(f"*{holdout_tag}*{strategy_name}_*.json"))
    if not matches:
        raise FileNotFoundError(f"No result json found for holdout_tag={holdout_tag} strategy={strategy_name}")
    return matches[-1]


def _top5_feature_means(result: dict[str, Any]) -> tuple[float, float, float]:
    ranked = result["event_filter"]["test"]["top_ranked"][:5]
    impulse_mean = statistics.mean(row["features"]["book_impulse"] for row in ranked)
    range_mean = statistics.mean(row["features"]["range_bps"] for row in ranked)
    score_mean = statistics.mean(row["score"] for row in ranked)
    return float(impulse_mean), float(range_mean), float(score_mean)


def _route_result_view(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    strat = result["strategy_metrics"]
    return {
        "strategy_name": result["strategy_name"],
        "result_json": str(result.get("result_json_path", "")) if result.get("result_json_path") else None,
        "final_pnl": float(strat["final_pnl"]),
        "trade_count": int(strat["trade_count"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "rmse": float(metrics["rmse"]),
        "rmse_ratio": float(result["rmse_ratio"]),
        "route_score": float(result["route_score"]),
        "runtime_seconds": float(result.get("runtime_seconds", 0.0)),
        "artifacts": result["artifacts"],
        "data_scale": result["data_scale"],
        "event_filter_test": result["event_filter"]["test"],
    }


async def _run_one_holdout(holdout: Path) -> dict[str, Any]:
    holdout_tag = holdout.stem.replace(":", "").replace("-", "")
    label = f"{LABEL_PREFIX}_{holdout_tag}"
    await exp22_main(
        variants=["regonly_l1_2d_h8"],
        train_files=DEFAULT_TRAIN_FILES,
        test_files=[holdout],
        label=label,
        directions=["short"],
        n_samples_per_file=None,
        load_every="1000ms",
        shaper_config_override=SHAPER,
    )
    route_payloads: dict[str, dict[str, Any]] = {}
    feature_impulse = None
    feature_range = None
    feature_score = None
    for route in ROUTES:
        path = _find_latest_result(holdout_tag, route)
        data = _load_json(path)
        route_payloads[route] = _route_result_view(data)
        if feature_impulse is None:
            feature_impulse, feature_range, feature_score = _top5_feature_means(data)
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


def _summary_picture(validation_rows: list[dict[str, Any]], route_totals: dict[str, float], out_path: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[1.1, 1.0])

    labels = [row["holdout_tag"][-6:] for row in validation_rows]
    x = list(range(len(labels)))
    width = 0.14
    offsets = [(-2 + i) * width for i in range(len(ROUTES))]
    for offset, route in zip(offsets, ROUTES):
        pnl_values = [row["route_results"][route]["final_pnl"] for row in validation_rows]
        axes[0].bar(
            [xi + offset for xi in x],
            pnl_values,
            width=width,
            label=ROUTE_LABELS[route],
            color=ROUTE_COLORS[route],
        )
    for idx, row in enumerate(validation_rows):
        axes[0].annotate(
            f"best={ROUTE_LABELS[row['best_route']]}",
            (idx, row["best_pnl"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("final pnl")
    axes[0].set_title("exp53 fixed-route robustness across adjacent later-evening holdouts")
    axes[0].legend(ncol=2, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    total_vals = [route_totals[route] for route in ROUTES]
    axes[1].bar(
        [ROUTE_LABELS[r] for r in ROUTES],
        total_vals,
        color=[ROUTE_COLORS[r] for r in ROUTES],
    )
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_ylabel("aggregate pnl across holdouts")
    axes[1].set_title("Aggregate robustness: least-fragile route wins by total pnl, trade activity, and graceful degradation")
    axes[1].tick_params(axis="x", rotation=18)
    for i, route in enumerate(ROUTES):
        wins = sum(1 for row in validation_rows if row["best_route"] == route)
        trade_count = sum(row["route_results"][route]["trade_count"] for row in validation_rows)
        axes[1].annotate(
            f"wins={wins}\ntrades={trade_count}",
            (i, total_vals[i]),
            textcoords="offset points",
            xytext=(0, 8 if total_vals[i] >= 0 else -28),
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


async def main() -> None:
    overall_start = time.perf_counter()
    timestamp_utc, stamp = _utc_stamp()
    git_commit = _git_commit()
    experiment = LABEL_PREFIX

    validation_rows = []
    holdout_runtimes = {}
    for holdout in VALIDATION_HOLDOUTS:
        start = time.perf_counter()
        validation_rows.append(await _run_one_holdout(holdout))
        holdout_runtimes[holdout.name] = round(time.perf_counter() - start, 3)

    route_totals = {route: float(sum(row["route_results"][route]["final_pnl"] for row in validation_rows)) for route in ROUTES}
    route_trade_totals = {route: int(sum(row["route_results"][route]["trade_count"] for row in validation_rows)) for route in ROUTES}
    route_win_counts = {route: sum(1 for row in validation_rows if row["best_route"] == route) for route in ROUTES}
    route_mean_precision = {route: float(statistics.mean(row["route_results"][route]["precision"] for row in validation_rows)) for route in ROUTES}
    route_mean_f1 = {route: float(statistics.mean(row["route_results"][route]["f1"] for row in validation_rows)) for route in ROUTES}
    route_mean_rmse_ratio = {route: float(statistics.mean(row["route_results"][route]["rmse_ratio"] for row in validation_rows)) for route in ROUTES}

    champion_route = max(
        ROUTES,
        key=lambda route: (
            route_totals[route],
            route_win_counts[route],
            -abs(route_mean_rmse_ratio[route] - 1.0),
            route_trade_totals[route],
        ),
    )
    champion_payload = {
        "route": champion_route,
        "total_pnl": route_totals[champion_route],
        "win_count": route_win_counts[champion_route],
        "trade_count": route_trade_totals[champion_route],
        "mean_precision": route_mean_precision[champion_route],
        "mean_f1": route_mean_f1[champion_route],
        "mean_rmse_ratio": route_mean_rmse_ratio[champion_route],
    }
    positive_routes = [route for route in ROUTES if route_totals[route] > 0]
    overall_decision = "promising" if len(positive_routes) >= 2 and champion_payload["win_count"] >= 2 else "not promising yet"

    summary_png = _summary_picture(validation_rows, route_totals, PICTURES_DIR / f"{experiment}_route_summary_{stamp}.png")

    summary = {
        "experiment": experiment,
        "timestamp_utc": timestamp_utc,
        "git_commit": git_commit,
        "device": "cpu",
        "hypothesis": "If selector geometry was the overfit layer rather than the mapper, adjacent later-evening holdouts should expose one least-fragile fixed execution family without any selector refit.",
        "mutation": "Keep mapper/train pool/stress filter frozen and stop selector refits; compare fixed short routes q95 fast, q90 fast, q90 persistence-fast-exit, q95 persistence2 hold96, and q90 persistence2 hold96 on adjacent later-evening holdouts 20:00, 21:00, 22:00, 23:00.",
        "validation": validation_rows,
        "data_scale": {
            "train_file_count": len(DEFAULT_TRAIN_FILES),
            "train_files": [str(path) for path in DEFAULT_TRAIN_FILES],
            "validation_holdout_count": len(VALIDATION_HOLDOUTS),
            "validation_holdouts": [str(path) for path in VALIDATION_HOLDOUTS],
            "total_unique_2026_files_touched": len(DEFAULT_TRAIN_FILES) + len(VALIDATION_HOLDOUTS),
            "capped_vs_uncapped": "uncapped (max_windows_per_file=null)",
            "serious_vs_screening": "serious validation",
        },
        "aggregate_results": {
            "route_totals": route_totals,
            "route_trade_totals": route_trade_totals,
            "route_win_counts": route_win_counts,
            "route_mean_precision": route_mean_precision,
            "route_mean_f1": route_mean_f1,
            "route_mean_rmse_ratio": route_mean_rmse_ratio,
            "champion": champion_payload,
            "positive_route_count": len(positive_routes),
            "decision": overall_decision,
        },
        "artifacts": {
            "route_summary_png": str(summary_png),
            "best_holdout_pngs": [row["best_picture"] for row in validation_rows],
        },
        "runtime_seconds": round(time.perf_counter() - overall_start, 3),
        "holdout_runtimes_seconds": holdout_runtimes,
    }

    result_path = RESULTS_DIR / f"{experiment}_{stamp}.json"
    result_path.write_text(json.dumps(summary, indent=2))

    note_path = NOTES_DIR / f"{experiment}_{stamp}.md"
    note_lines = [
        f"# {experiment}",
        "",
        f"- timestamp: {timestamp_utc}",
        "- hypothesis: adjacent later-evening holdouts will reveal the least-fragile fixed route after selector failure",
        f"- champion_route: {champion_route}",
        f"- champion_payload: {json.dumps(champion_payload, sort_keys=True)}",
        f"- route_totals: {json.dumps(route_totals, sort_keys=True)}",
        f"- route_win_counts: {json.dumps(route_win_counts, sort_keys=True)}",
        f"- decision: `{overall_decision}`",
        f"- summary_png: {summary_png}",
        f"- result_json: {result_path}",
    ]
    for row in validation_rows:
        note_lines.extend(
            [
                f"- {row['holdout_tag']}: features=(impulse={row['feature_book_impulse_top5_mean']:.6f}, range_bps={row['feature_range_bps_top5_mean']:.6f}, score={row['feature_score_top5_mean']:.6f}) best={row['best_route']} best_pnl={row['best_pnl']:.5f}",
                f"  - best_fixed_png: {row['best_picture']}",
            ]
        )
    note_path.write_text("\n".join(note_lines) + "\n")

    row_id = register_experiment_run(
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
            "recall": float(statistics.mean(row["route_results"][champion_route]["recall"] for row in validation_rows)),
            "f1": champion_payload["mean_f1"],
            "rmse": float(statistics.mean(row["route_results"][champion_route]["rmse"] for row in validation_rows)),
            "rmse_ratio": champion_payload["mean_rmse_ratio"],
            "final_pnl": champion_payload["total_pnl"],
            "trade_count": champion_payload["trade_count"],
            "route_win_count": champion_payload["win_count"],
            **route_totals,
        },
        notes=f"notes_file={note_path}",
    )
    summary["experiment_db_row_id"] = row_id
    result_path.write_text(json.dumps(summary, indent=2))

    route_totals_text = ", ".join(f"{route}={route_totals[route]:.2f}" for route in ROUTES)
    per_holdout_text = "; ".join(
        f"{row['holdout_tag'][-6:]} (impulse={row['feature_book_impulse_top5_mean']:.4f}, range={row['feature_range_bps_top5_mean']:.2f}) -> best {row['best_route']}={row['best_pnl']:.2f}"
        for row in validation_rows
    )
    journal_text = f"""## {timestamp_utc}
- **Continuity**: executed the queued post-exp52 fixed-route robustness mutation with mapper/train pool/stress filter frozen and no selector refit. Advanced serious validation to adjacent later-evening holdouts `2026-04-22T20-00-11`, `21-00-01`, `22-00-00`, and `23-00-01`.
- **Hypothesis**: if selector geometry was the overfit layer rather than the mapper, adjacent later-evening holdouts should reveal one least-fragile fixed execution family without any selector refit.
- **What changed**: execution evaluation only. Stopped selector logic entirely and compared fixed short routes `q95_p1_hold48`, `q90_p1_hold48`, `q90_persistence2_fast_exit`, `q95_p2_hold96`, and `q90_p2_hold96`.
- **What stayed fixed**: recent 24-file 2026 train pool (`2026-04-16`..`2026-04-21` at `00/06/12/18`), frozen `regonly_l1_2d_h8` branch, short-only direction, `1000ms`, `rolling=1024`, `look_ahead=64`, observable stress filter, uncapped loading.
- **Runtime / hardware**: CPU, total `{summary['runtime_seconds']:.3f}s`; per-holdout runtimes `{holdout_runtimes}`. Serious validation, not screening.
- **Data scale**:
  - 2026 serious-validation coverage: 24 train files + holdouts `2026-04-22T20-00-11`, `21-00-01`, `22-00-00`, and `23-00-01`.
  - Total unique 2026 files touched across this mutation: {summary['data_scale']['total_unique_2026_files_touched']}.
  - Capped vs uncapped: uncapped (`max_windows_per_file=null`).
- **Main result**:
  - route totals `{route_totals_text}`.
  - champion route `{champion_route}` with total pnl `{champion_payload['total_pnl']:.2f}`, wins `{champion_payload['win_count']}/{len(validation_rows)}`, trades `{champion_payload['trade_count']}`, mean precision `{champion_payload['mean_precision']:.4f}`, mean f1 `{champion_payload['mean_f1']:.4f}`, mean rmse_ratio `{champion_payload['mean_rmse_ratio']:.4f}`.
  - per-holdout: {per_holdout_text}.
- **Pictures**:
  - route summary: `{summary_png}`
  - validation diagnostic 1: `{validation_rows[0]['best_picture']}`
  - validation diagnostic 2: `{validation_rows[-1]['best_picture']}`
- **Interpretation**: {'a fixed route family finally looked least-fragile across adjacent later holdouts, so execution may be stabilizing without mapper retrain.' if overall_decision == 'promising' else 'no fixed route cleanly dominated enough adjacent later-evening holdouts to call execution solved; the mapper branch may still be alive, but strategy fragility remains the blocker.'}
- **Decision**: `{overall_decision}`.
- **Exact next mutation**: keep the mapper frozen; if `{champion_route}` stays the least-fragile route, broaden to the next chronological date block and retest higher friction sensitivity before any retrain. If it fails immediately, consider a minimal observable regime gate only between the top two fixed routes rather than another broad selector refit.
"""
    _append_text(HOURLY_JOURNAL, journal_text)

    lab_text = f"""## {timestamp_utc} exp53 fixed-route later-evening robustness

Status: DONE

### Mutation from prior run
- Kept the recent 24-file 2026 stress-conditioned short branch fixed.
- Stopped selector refits after exp52 failure.
- Changed one practical layer only: compared fixed execution families on adjacent later-evening holdouts `20:00`, `21:00`, `22:00`, `23:00`.

### Hypothesis
- If selector geometry was the overfit layer rather than the mapper, adjacent later-evening holdouts should reveal one least-fragile fixed route family without retraining the mapper.

### Artifacts
- JSON: `{result_path}`
- Summary picture: `{summary_png}`
- Notes: `{note_path}`
- DB row: `experiment_runs.id = {row_id}`

### Outcome
- Route totals: `{route_totals_text}`
- Champion: `{champion_route}` with pnl `{champion_payload['total_pnl']:.5f}`, wins `{champion_payload['win_count']}/{len(validation_rows)}`, trades `{champion_payload['trade_count']}`
- Decision: `{overall_decision}`

### Interpretation
- {'A fixed route family emerged as the least-fragile execution candidate on adjacent later-evening holdouts.' if overall_decision == 'promising' else 'Fixed-route robustness was still too unstable on adjacent later-evening holdouts, so execution remains the active blocker even without selector overfit in the loop.'}
"""
    _append_text(LAB_LOG, lab_text)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

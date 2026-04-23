"""exp49 — validate a simple observable execution selector for the recent stress-conditioned short branch.

Hypothesis
----------
If exp48's hour-to-hour execution drift reflects a simple present-time stress split rather than mapper death,
then a selector using current-window stress intensity should choose between:
- strict fast short execution: q95_p1_hold48
- relaxed fast short execution: q90_p1_hold48
and beat or match a fixed-route choice on the next later chronological holdouts.

Single main mutation vs exp48:
- keep mapper family fixed at `regonly_l1_2d_h8`
- keep cadence / geometry fixed at 1000ms, rolling=1024, look_ahead=64
- keep observable present-time stress filtering fixed
- keep the recent 24-file 2026 train pool fixed (2026-04-16..2026-04-21 at 00/06/12/18)
- mutate only execution-policy selection using a simple book-impulse gate calibrated on exp48 holdouts 00:00 and 01:00
- validate on the next two chronological holdouts: 2026-04-22T02:00:00 and 2026-04-22T03:00:01

Usage
-----
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. ./.venv/bin/python scripts/exp49_recent24_stress_short_selector_validation.py
"""

from __future__ import annotations

import asyncio
import json
import statistics
import subprocess
import time
from dataclasses import dataclass
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

CALIBRATION_HOLDOUT_KEYS = ["20260422T000000", "20260422T010000"]
VALIDATION_HOLDOUTS = [
    DATA_DIR / "2026-04-22T02-00-00.parquet",
    DATA_DIR / "2026-04-22T03-00-01.parquet",
]
SELECTOR_ROUTES = ["q95_p1_hold48", "q90_p1_hold48"]
LABEL_PREFIX = "exp49_recent24_stress_short_selector_validation"

SHAPER = DEFAULT_SHAPER_CONFIG_2026.but(
    rolling_window_size=1024,
    look_ahead=64,
    window_stride=4,
)


@dataclass
class HoldoutRoutePair:
    holdout_tag: str
    holdout_path: str
    feature_book_impulse_top5_mean: float
    feature_score_top5_mean: float
    route_results: dict[str, dict[str, Any]]
    selected_route: str
    selected_pnl: float
    alternative_route: str
    alternative_pnl: float
    pnl_edge_vs_alternative: float
    pictures: dict[str, str]
    decision: str


def _utc_stamp() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M:%SZ"), now.strftime("%Y%m%dT%H%M%SZ")


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        return out or None
    except Exception:
        return None


def _find_latest_result(label: str, strategy_name: str) -> Path:
    matches = sorted(RESULTS_DIR.glob(f"{label}_*{strategy_name}_*.json"))
    if not matches:
        raise FileNotFoundError(f"No result json found for label={label} strategy={strategy_name}")
    return matches[-1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _top5_feature_means(result: dict[str, Any]) -> tuple[float, float]:
    ranked = result["event_filter"]["test"]["top_ranked"][:5]
    impulse_mean = statistics.mean(row["features"]["book_impulse"] for row in ranked)
    score_mean = statistics.mean(row["score"] for row in ranked)
    return float(impulse_mean), float(score_mean)


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


def _load_existing_calibration() -> list[dict[str, Any]]:
    rows = []
    for holdout_tag in CALIBRATION_HOLDOUT_KEYS:
        label = f"exp48_recent24_stress_short_post23_multiholdout_execution_falsification_{holdout_tag}"
        route_payloads = {}
        feature_impulse = None
        feature_score = None
        for route in SELECTOR_ROUTES:
            path = _find_latest_result(label, route)
            data = _load_json(path)
            route_payloads[route] = _route_result_view(data)
            if feature_impulse is None:
                feature_impulse, feature_score = _top5_feature_means(data)
        selected = max(route_payloads.values(), key=lambda row: row["final_pnl"])
        rows.append(
            {
                "holdout_tag": holdout_tag,
                "holdout_path": str(DATA_DIR / f"{holdout_tag[:4]}-{holdout_tag[4:6]}-{holdout_tag[6:8]}T{holdout_tag[9:11]}-00-00.parquet"),
                "feature_book_impulse_top5_mean": feature_impulse,
                "feature_score_top5_mean": feature_score,
                "winner": selected["strategy_name"],
                "winner_pnl": selected["final_pnl"],
                "route_results": route_payloads,
            }
        )
    return rows


def _selector_threshold(calibration_rows: list[dict[str, Any]]) -> float:
    hi = None
    lo = None
    for row in calibration_rows:
        if row["winner"] == "q95_p1_hold48":
            hi = row["feature_book_impulse_top5_mean"]
        elif row["winner"] == "q90_p1_hold48":
            lo = row["feature_book_impulse_top5_mean"]
    if hi is None or lo is None:
        raise RuntimeError("Calibration rows did not separate q95 and q90 winners; cannot build selector threshold.")
    return float((hi + lo) / 2.0)


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
    feature_score = None
    for route in SELECTOR_ROUTES:
        path = _find_latest_result(label, route)
        data = _load_json(path)
        route_payloads[route] = _route_result_view(data)
        if feature_impulse is None:
            feature_impulse, feature_score = _top5_feature_means(data)
    return {
        "label": label,
        "holdout_tag": holdout_tag,
        "holdout_path": str(holdout),
        "feature_book_impulse_top5_mean": feature_impulse,
        "feature_score_top5_mean": feature_score,
        "route_results": route_payloads,
    }


def _choose_route(impulse_feature: float, threshold: float) -> str:
    return "q95_p1_hold48" if impulse_feature >= threshold else "q90_p1_hold48"


def _selector_decision_text(selected_pnl: float, alt_pnl: float) -> str:
    if selected_pnl > alt_pnl:
        return "keep"
    if selected_pnl == alt_pnl:
        return "neutral"
    return "reject"


def _summary_picture(calibration_rows: list[dict[str, Any]], validation_pairs: list[HoldoutRoutePair], threshold: float, out_path: Path) -> Path:
    labels = [row["holdout_tag"][-6:] for row in calibration_rows] + [pair.holdout_tag[-6:] for pair in validation_pairs]
    impulses = [row["feature_book_impulse_top5_mean"] for row in calibration_rows] + [pair.feature_book_impulse_top5_mean for pair in validation_pairs]
    q95 = [row["route_results"]["q95_p1_hold48"]["final_pnl"] for row in calibration_rows] + [pair.route_results["q95_p1_hold48"]["final_pnl"] for pair in validation_pairs]
    q90 = [row["route_results"]["q90_p1_hold48"]["final_pnl"] for row in calibration_rows] + [pair.route_results["q90_p1_hold48"]["final_pnl"] for pair in validation_pairs]
    chosen = [row["winner"] for row in calibration_rows] + [pair.selected_route for pair in validation_pairs]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 1.6], sharex=True)
    x = list(range(len(labels)))

    axes[0].plot(x, impulses, marker="o", color="tab:purple", label="top5 mean book_impulse")
    axes[0].axhline(threshold, color="black", linestyle="--", linewidth=1.2, label=f"selector threshold={threshold:.4f}")
    for idx, route in enumerate(chosen):
        axes[0].annotate(route.replace("_p1_hold48", ""), (x[idx], impulses[idx]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    axes[0].set_ylabel("book_impulse")
    axes[0].set_title("exp49 selector calibration + validation")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    width = 0.36
    axes[1].bar([xi - width / 2 for xi in x], q95, width=width, label="q95_p1_hold48 pnl", color="#4C78A8")
    axes[1].bar([xi + width / 2 for xi in x], q90, width=width, label="q90_p1_hold48 pnl", color="#F58518")
    for idx, route in enumerate(chosen):
        pnl = q95[idx] if route == "q95_p1_hold48" else q90[idx]
        axes[1].scatter([x[idx]], [pnl], s=90, color="black", marker="*", zorder=5)
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_ylabel("final pnl")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("holdout HHMMSS suffix (calibration first, validation last)")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def _append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        if path.exists() and path.stat().st_size > 0:
            fh.write("\n")
        fh.write(text.rstrip() + "\n")


async def main() -> None:
    overall_start = time.perf_counter()
    timestamp_utc, stamp = _utc_stamp()
    git_commit = _git_commit()
    experiment = LABEL_PREFIX

    calibration_rows = _load_existing_calibration()
    threshold = _selector_threshold(calibration_rows)

    validation_raw = []
    holdout_runtimes = {}
    for holdout in VALIDATION_HOLDOUTS:
        start = time.perf_counter()
        validation_raw.append(await _run_one_holdout(holdout))
        holdout_runtimes[holdout.name] = round(time.perf_counter() - start, 3)

    validation_pairs: list[HoldoutRoutePair] = []
    for raw in validation_raw:
        selected_route = _choose_route(raw["feature_book_impulse_top5_mean"], threshold)
        alternative_route = [route for route in SELECTOR_ROUTES if route != selected_route][0]
        selected_payload = raw["route_results"][selected_route]
        alt_payload = raw["route_results"][alternative_route]
        validation_pairs.append(
            HoldoutRoutePair(
                holdout_tag=raw["holdout_tag"],
                holdout_path=raw["holdout_path"],
                feature_book_impulse_top5_mean=float(raw["feature_book_impulse_top5_mean"]),
                feature_score_top5_mean=float(raw["feature_score_top5_mean"]),
                route_results=raw["route_results"],
                selected_route=selected_route,
                selected_pnl=float(selected_payload["final_pnl"]),
                alternative_route=alternative_route,
                alternative_pnl=float(alt_payload["final_pnl"]),
                pnl_edge_vs_alternative=float(selected_payload["final_pnl"] - alt_payload["final_pnl"]),
                pictures={
                    "selected_fixed_png": selected_payload["artifacts"]["dashboard_fixed_png"],
                    "selected_best_png": selected_payload["artifacts"]["dashboard_best_png"],
                    "alternative_fixed_png": alt_payload["artifacts"]["dashboard_fixed_png"],
                },
                decision=_selector_decision_text(float(selected_payload["final_pnl"]), float(alt_payload["final_pnl"])),
            )
        )

    selector_total_pnl = float(sum(pair.selected_pnl for pair in validation_pairs))
    q95_total_pnl = float(sum(pair.route_results["q95_p1_hold48"]["final_pnl"] for pair in validation_pairs))
    q90_total_pnl = float(sum(pair.route_results["q90_p1_hold48"]["final_pnl"] for pair in validation_pairs))
    correct_choice_count = sum(1 for pair in validation_pairs if pair.selected_pnl >= pair.alternative_pnl)
    overall_decision = "promising" if selector_total_pnl >= max(q95_total_pnl, q90_total_pnl) and correct_choice_count == len(validation_pairs) else "not promising yet"

    summary_png = _summary_picture(
        calibration_rows,
        validation_pairs,
        threshold,
        PICTURES_DIR / f"{experiment}_selector_summary_{stamp}.png",
    )

    summary = {
        "experiment": experiment,
        "timestamp_utc": timestamp_utc,
        "git_commit": git_commit,
        "device": "cpu",
        "hypothesis": "A simple present-time book-impulse selector can choose between strict q95 and relaxed q90 fast short execution for the recent stress-conditioned branch.",
        "mutation": "Calibrate a single threshold on exp48 top-5 holdout book_impulse means (00:00 vs 01:00), then validate that selector on 02:00 and 03:00 while keeping train pool, mapper family, cadence, geometry, and stress filter fixed.",
        "selector": {
            "feature": "top5_mean_book_impulse_from_event_filter_test_top_ranked",
            "threshold": threshold,
            "rule": "if feature >= threshold choose q95_p1_hold48 else choose q90_p1_hold48",
            "calibration_rows": calibration_rows,
        },
        "validation": [
            {
                "holdout_tag": pair.holdout_tag,
                "holdout_path": pair.holdout_path,
                "feature_book_impulse_top5_mean": pair.feature_book_impulse_top5_mean,
                "feature_score_top5_mean": pair.feature_score_top5_mean,
                "selected_route": pair.selected_route,
                "selected_pnl": pair.selected_pnl,
                "alternative_route": pair.alternative_route,
                "alternative_pnl": pair.alternative_pnl,
                "pnl_edge_vs_alternative": pair.pnl_edge_vs_alternative,
                "decision": pair.decision,
                "route_results": pair.route_results,
                "pictures": pair.pictures,
            }
            for pair in validation_pairs
        ],
        "data_scale": {
            "train_file_count": len(DEFAULT_TRAIN_FILES),
            "train_files": [str(path) for path in DEFAULT_TRAIN_FILES],
            "validation_holdout_count": len(VALIDATION_HOLDOUTS),
            "validation_holdouts": [str(path) for path in VALIDATION_HOLDOUTS],
            "calibration_holdout_count": len(calibration_rows),
            "calibration_holdout_keys": CALIBRATION_HOLDOUT_KEYS,
            "total_unique_2026_files_touched": len(DEFAULT_TRAIN_FILES) + len(VALIDATION_HOLDOUTS) + len(calibration_rows),
            "capped_vs_uncapped": "uncapped (max_windows_per_file=null)",
            "serious_vs_screening": "serious validation",
        },
        "aggregate_results": {
            "selector_total_pnl": selector_total_pnl,
            "q95_total_pnl": q95_total_pnl,
            "q90_total_pnl": q90_total_pnl,
            "correct_choice_count": correct_choice_count,
            "validation_count": len(validation_pairs),
            "decision": overall_decision,
        },
        "artifacts": {
            "selector_summary_png": str(summary_png),
            "validation_selected_fixed_pngs": [pair.pictures["selected_fixed_png"] for pair in validation_pairs],
            "validation_selected_best_pngs": [pair.pictures["selected_best_png"] for pair in validation_pairs],
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
        "- hypothesis: simple present-time book-impulse gate selects strict vs relaxed fast short execution",
        f"- threshold: {threshold:.6f}",
        f"- selector_total_pnl: {selector_total_pnl:.5f}",
        f"- q95_total_pnl: {q95_total_pnl:.5f}",
        f"- q90_total_pnl: {q90_total_pnl:.5f}",
        f"- decision: `{overall_decision}`",
        f"- summary_png: {summary_png}",
        f"- result_json: {result_path}",
    ]
    for pair in validation_pairs:
        note_lines.extend(
            [
                f"- {pair.holdout_tag}: feature={pair.feature_book_impulse_top5_mean:.6f} selected={pair.selected_route} pnl={pair.selected_pnl:.5f} alt={pair.alternative_route} alt_pnl={pair.alternative_pnl:.5f}",
                f"  - selected_fixed_png: {pair.pictures['selected_fixed_png']}",
                f"  - selected_best_png: {pair.pictures['selected_best_png']}",
            ]
        )
    note_path.write_text("\n".join(note_lines) + "\n")

    row_id = register_experiment_run(
        experiment=experiment,
        variant="selector_book_impulse_q95_vs_q90",
        timestamp_utc=timestamp_utc,
        git_commit=git_commit,
        symbol="BTC-USD",
        cadence="1000ms",
        look_ahead=64,
        model="regonly_l1_2d_h8",
        result_json_path=result_path,
        picture_path=summary_png,
        metrics={
            "precision": statistics.mean(pair.route_results[pair.selected_route]["precision"] for pair in validation_pairs),
            "recall": statistics.mean(pair.route_results[pair.selected_route]["recall"] for pair in validation_pairs),
            "f1": statistics.mean(pair.route_results[pair.selected_route]["f1"] for pair in validation_pairs),
            "rmse": statistics.mean(pair.route_results[pair.selected_route]["rmse"] for pair in validation_pairs),
            "rmse_ratio": statistics.mean(pair.route_results[pair.selected_route]["rmse_ratio"] for pair in validation_pairs),
            "final_pnl": selector_total_pnl,
            "trade_count": sum(pair.route_results[pair.selected_route]["trade_count"] for pair in validation_pairs),
            "correct_choice_count": correct_choice_count,
            "validation_count": len(validation_pairs),
            "q95_total_pnl": q95_total_pnl,
            "q90_total_pnl": q90_total_pnl,
        },
        notes=f"notes_file={note_path}",
    )
    summary["experiment_db_row_id"] = row_id
    result_path.write_text(json.dumps(summary, indent=2))

    journal_text = f"""## {timestamp_utc}
- **Continuity**: executed the queued post-exp48 selector mutation with the mapper/train pool/stress filter frozen. Calibration reused exp48 holdouts `00:00` and `01:00`; validation advanced to the next two chronological holdouts `2026-04-22T02-00-00` and `2026-04-22T03-00-01`.
- **Hypothesis**: the hour-to-hour execution flip from exp48 can be partially explained by present-time stress intensity, so a simple observable gate on top-ranked holdout `book_impulse` should choose between strict `q95_p1_hold48` and relaxed `q90_p1_hold48` better than a fixed route.
- **What changed**: execution policy only. The selector feature was `top5 mean book_impulse` from the test event-filter ranking; threshold `{threshold:.6f}` was calibrated from exp48 (`00:00` strict winner vs `01:00` relaxed winner).
- **What stayed fixed**: recent 24-file 2026 train pool (`2026-04-16`..`2026-04-21` at `00/06/12/18`), frozen `regonly_l1_2d_h8` branch, short-only direction, `1000ms`, `rolling=1024`, `look_ahead=64`, observable stress filter, uncapped loading.
- **Runtime / hardware**: CPU, total `{summary['runtime_seconds']:.3f}s`; per-holdout runtimes `{holdout_runtimes}`. Serious validation, not screening.
- **Data scale**:
  - 2026 serious-validation coverage: 24 train files + holdouts `2026-04-22T02-00-00` and `2026-04-22T03-00-01`.
  - Calibration continuity used prior exp48 holdouts `2026-04-22T00-00-00` and `2026-04-22T01-00-00` only for selector fitting.
  - Total unique 2026 files touched across calibration + validation: {summary['data_scale']['total_unique_2026_files_touched']}.
  - Capped vs uncapped: uncapped (`max_windows_per_file=null`).
- **Main result**:
  - selector total pnl `{selector_total_pnl:.2f}` vs fixed `q95_total_pnl={q95_total_pnl:.2f}` and `q90_total_pnl={q90_total_pnl:.2f}` on the two validation holdouts.
  - selector correct choices: `{correct_choice_count}/{len(validation_pairs)}`.
  - per-holdout: {'; '.join(f"{pair.holdout_tag[-6:]} feature={pair.feature_book_impulse_top5_mean:.4f} -> {pair.selected_route} pnl={pair.selected_pnl:.2f} vs {pair.alternative_route}={pair.alternative_pnl:.2f}" for pair in validation_pairs)}.
- **Pictures**:
  - selector summary: `{summary_png}`
  - validation diagnostic 1: `{validation_pairs[0].pictures['selected_fixed_png']}`
  - validation diagnostic 2: `{validation_pairs[-1].pictures['selected_fixed_png']}`
- **Interpretation**: {'the simple book-impulse selector survived the next holdouts and outperformed both fixed routes.' if overall_decision == 'promising' else 'the simple book-impulse selector did not cleanly solve the execution drift on the next holdouts; treat selector quality as unproven.'}
- **Decision**: `{overall_decision}`.
- **Exact next mutation**: keep the mapper frozen; if selector underperformed, test a slightly richer but still observable 2-feature gate (`book_impulse` + `range_bps` or `score`) before any retrain. If selector held up, validate on later day/evening holdouts before promotion.
"""
    _append_text(HOURLY_JOURNAL, journal_text)

    lab_text = f"""## {timestamp_utc} exp49 selector validation

Status: DONE

### Mutation from prior run
- Kept the recent 24-file 2026 stress-conditioned short branch fixed.
- Changed one practical layer only: execution selection between `q95_p1_hold48` and `q90_p1_hold48` using a single present-time observable (`top5 mean book_impulse` on the holdout event-filter ranking).
- Calibration reused exp48 holdouts `00:00` and `01:00`; serious validation moved to `02:00` and `03:00`.

### Hypothesis
- The post-23:00 execution flip is partly explainable by current-window stress intensity, so a simple book-impulse gate can choose the better fast short policy without retraining the mapper.

### Artifacts
- JSON: `{result_path}`
- Summary picture: `{summary_png}`
- Notes: `{note_path}`
- DB row: `experiment_runs.id = {row_id}`

### Outcome
- Selector threshold: `{threshold:.6f}`
- Selector total pnl on validation: `{selector_total_pnl:.5f}`
- Fixed q95 total pnl: `{q95_total_pnl:.5f}`
- Fixed q90 total pnl: `{q90_total_pnl:.5f}`
- Correct route selections: `{correct_choice_count}/{len(validation_pairs)}`
- Decision: `{overall_decision}`

### Interpretation
- {'The selector adds evidence that execution drift may be partly observable in present-time stress without changing the mapper.' if overall_decision == 'promising' else 'The selector did not generalize cleanly enough on the next holdouts, so execution drift remains the blocker and the mapper should stay frozen while the execution layer is falsified further.'}
"""
    _append_text(LAB_LOG, lab_text)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

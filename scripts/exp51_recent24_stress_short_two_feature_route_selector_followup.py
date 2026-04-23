"""exp51 — falsify the frozen exp50 two-feature execution selector on later chronological holdouts.

Hypothesis
----------
If the exp50 two-feature selector captured a real present-time execution-policy geometry rather than just tying the best fixed route on one active holdout,
then the SAME frozen selector calibrated on 00:00..03:00 should continue to choose among:
- strict fast short execution: q95_p1_hold48
- relaxed fast short execution: q90_p1_hold48
- persistence-fast-exit short execution: q90_persistence2_fast_exit
at least as well as the best fixed route on the next later chronological holdouts 06:00 and 07:00.

Single main mutation vs exp50:
- keep mapper family fixed at `regonly_l1_2d_h8`
- keep cadence / geometry fixed at 1000ms, rolling=1024, look_ahead=64
- keep observable present-time stress filtering fixed
- keep the recent 24-file 2026 train pool fixed (2026-04-16..2026-04-21 at 00/06/12/18)
- keep the exp50 selector calibration frozen on existing holdouts 00:00..03:00
- mutate only validation scope by pushing the same selector onto the next later chronological holdouts: 2026-04-22T06:00:00 and 2026-04-22T07:00:00

Usage
-----
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. ./.venv/bin/python scripts/exp51_recent24_stress_short_two_feature_route_selector_followup.py
"""

from __future__ import annotations

import asyncio
import json
import math
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

CALIBRATION_HOLDOUT_KEYS = [
    "20260422T000000",
    "20260422T010000",
    "20260422T020000",
    "20260422T030001",
]
VALIDATION_HOLDOUTS = [
    DATA_DIR / "2026-04-22T06-00-00.parquet",
    DATA_DIR / "2026-04-22T07-00-00.parquet",
]
SELECTOR_ROUTES = [
    "q95_p1_hold48",
    "q90_p1_hold48",
    "q90_persistence2_fast_exit",
]
LABEL_PREFIX = "exp51_recent24_stress_short_two_feature_route_selector_followup"

SHAPER = DEFAULT_SHAPER_CONFIG_2026.but(
    rolling_window_size=1024,
    look_ahead=64,
    window_stride=4,
)


@dataclass
class HoldoutRouteChoice:
    holdout_tag: str
    holdout_path: str
    feature_book_impulse_top5_mean: float
    feature_range_bps_top5_mean: float
    feature_score_top5_mean: float
    route_results: dict[str, dict[str, Any]]
    selected_route: str
    selected_pnl: float
    best_route: str
    best_pnl: float
    pnl_edge_vs_best: float
    pictures: dict[str, str]
    decision: str
    distance_by_route: dict[str, float]


@dataclass
class FeatureScaler:
    impulse_mean: float
    impulse_std: float
    range_mean: float
    range_std: float


def _utc_stamp() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M:%SZ"), now.strftime("%Y%m%dT%H%M%SZ")


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        return out or None
    except Exception:
        return None


def _find_latest_result(holdout_tag: str, strategy_name: str) -> Path:
    matches = sorted(RESULTS_DIR.glob(f"*{holdout_tag}*{strategy_name}_*.json"))
    if not matches:
        raise FileNotFoundError(f"No result json found for holdout_tag={holdout_tag} strategy={strategy_name}")
    return matches[-1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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


def _append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        if path.exists() and path.stat().st_size > 0:
            fh.write("\n")
        fh.write(text.rstrip() + "\n")


def _holdout_path_from_key(holdout_tag: str) -> str:
    y = holdout_tag[:4]
    m = holdout_tag[4:6]
    d = holdout_tag[6:8]
    hh = holdout_tag[9:11]
    mm = holdout_tag[11:13]
    ss = holdout_tag[13:15]
    return str(DATA_DIR / f"{y}-{m}-{d}T{hh}-{mm}-{ss}.parquet")


def _load_existing_calibration() -> list[dict[str, Any]]:
    rows = []
    for holdout_tag in CALIBRATION_HOLDOUT_KEYS:
        route_payloads = {}
        feature_impulse = None
        feature_range = None
        feature_score = None
        for route in SELECTOR_ROUTES:
            path = _find_latest_result(holdout_tag, route)
            data = _load_json(path)
            route_payloads[route] = _route_result_view(data)
            if feature_impulse is None:
                feature_impulse, feature_range, feature_score = _top5_feature_means(data)
        winner = max(route_payloads.values(), key=lambda row: row["final_pnl"])
        rows.append(
            {
                "holdout_tag": holdout_tag,
                "holdout_path": _holdout_path_from_key(holdout_tag),
                "feature_book_impulse_top5_mean": feature_impulse,
                "feature_range_bps_top5_mean": feature_range,
                "feature_score_top5_mean": feature_score,
                "winner": winner["strategy_name"],
                "winner_pnl": winner["final_pnl"],
                "route_results": route_payloads,
            }
        )
    return rows


def _build_scaler(rows: list[dict[str, Any]]) -> FeatureScaler:
    impulses = [row["feature_book_impulse_top5_mean"] for row in rows]
    ranges = [row["feature_range_bps_top5_mean"] for row in rows]
    return FeatureScaler(
        impulse_mean=float(statistics.mean(impulses)),
        impulse_std=float(statistics.pstdev(impulses) or 1.0),
        range_mean=float(statistics.mean(ranges)),
        range_std=float(statistics.pstdev(ranges) or 1.0),
    )


def _scaled_point(impulse: float, range_bps: float, scaler: FeatureScaler) -> tuple[float, float]:
    return (
        (impulse - scaler.impulse_mean) / scaler.impulse_std,
        (range_bps - scaler.range_mean) / scaler.range_std,
    )


def _build_centroids(rows: list[dict[str, Any]], scaler: FeatureScaler) -> dict[str, dict[str, float]]:
    by_route: dict[str, list[tuple[float, float]]] = {route: [] for route in SELECTOR_ROUTES}
    for row in rows:
        by_route[row["winner"]].append(
            _scaled_point(
                row["feature_book_impulse_top5_mean"],
                row["feature_range_bps_top5_mean"],
                scaler,
            )
        )
    centroids: dict[str, dict[str, float]] = {}
    for route, points in by_route.items():
        if not points:
            continue
        centroids[route] = {
            "scaled_book_impulse": float(statistics.mean(p[0] for p in points)),
            "scaled_range_bps": float(statistics.mean(p[1] for p in points)),
            "count": len(points),
        }
    return centroids


def _choose_route(impulse: float, range_bps: float, scaler: FeatureScaler, centroids: dict[str, dict[str, float]]) -> tuple[str, dict[str, float]]:
    x_impulse, x_range = _scaled_point(impulse, range_bps, scaler)
    distance_by_route = {}
    for route, centroid in centroids.items():
        dist = math.sqrt(
            (x_impulse - centroid["scaled_book_impulse"]) ** 2
            + (x_range - centroid["scaled_range_bps"]) ** 2
        )
        distance_by_route[route] = float(dist)
    selected_route = min(distance_by_route, key=distance_by_route.get)
    return selected_route, distance_by_route


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
    for route in SELECTOR_ROUTES:
        path = _find_latest_result(holdout_tag, route)
        data = _load_json(path)
        route_payloads[route] = _route_result_view(data)
        if feature_impulse is None:
            feature_impulse, feature_range, feature_score = _top5_feature_means(data)
    return {
        "label": label,
        "holdout_tag": holdout_tag,
        "holdout_path": str(holdout),
        "feature_book_impulse_top5_mean": feature_impulse,
        "feature_range_bps_top5_mean": feature_range,
        "feature_score_top5_mean": feature_score,
        "route_results": route_payloads,
    }


def _summary_picture(
    calibration_rows: list[dict[str, Any]],
    validation_pairs: list[HoldoutRouteChoice],
    scaler: FeatureScaler,
    centroids: dict[str, dict[str, float]],
    out_path: Path,
) -> Path:
    route_colors = {
        "q95_p1_hold48": "#4C78A8",
        "q90_p1_hold48": "#F58518",
        "q90_persistence2_fast_exit": "#54A24B",
    }
    route_labels = {
        "q95_p1_hold48": "q95 fast",
        "q90_p1_hold48": "q90 fast",
        "q90_persistence2_fast_exit": "q90 persistence-fast-exit",
    }

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1.15, 1.0])

    for row in calibration_rows:
        route = row["winner"]
        axes[0].scatter(
            row["feature_book_impulse_top5_mean"],
            row["feature_range_bps_top5_mean"],
            color=route_colors[route],
            s=90,
            marker="o",
            alpha=0.9,
        )
        axes[0].annotate(
            f"cal {row['holdout_tag'][-6:]}",
            (row["feature_book_impulse_top5_mean"], row["feature_range_bps_top5_mean"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

    for pair in validation_pairs:
        route = pair.selected_route
        axes[0].scatter(
            pair.feature_book_impulse_top5_mean,
            pair.feature_range_bps_top5_mean,
            color=route_colors[route],
            s=120,
            marker="*",
            alpha=0.95,
        )
        axes[0].annotate(
            f"val {pair.holdout_tag[-6:]} → {route_labels[route]}",
            (pair.feature_book_impulse_top5_mean, pair.feature_range_bps_top5_mean),
            textcoords="offset points",
            xytext=(6, -12),
            fontsize=8,
        )

    for route, centroid in centroids.items():
        raw_impulse = centroid["scaled_book_impulse"] * scaler.impulse_std + scaler.impulse_mean
        raw_range = centroid["scaled_range_bps"] * scaler.range_std + scaler.range_mean
        axes[0].scatter(raw_impulse, raw_range, color=route_colors[route], s=180, marker="X", edgecolor="black")
        axes[0].annotate(
            f"centroid {route_labels[route]}",
            (raw_impulse, raw_range),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8,
        )

    axes[0].set_title("exp51 frozen two-feature selector follow-up: calibration clusters + later validation choices")
    axes[0].set_xlabel("top5 mean book_impulse")
    axes[0].set_ylabel("top5 mean range_bps")
    axes[0].grid(alpha=0.25)

    labels = [pair.holdout_tag[-6:] for pair in validation_pairs]
    x = list(range(len(labels)))
    width = 0.23
    for offset, route in zip([-width, 0.0, width], SELECTOR_ROUTES):
        pnl_values = [pair.route_results[route]["final_pnl"] for pair in validation_pairs]
        axes[1].bar(
            [xi + offset for xi in x],
            pnl_values,
            width=width,
            label=route_labels[route],
            color=route_colors[route],
        )
    for idx, pair in enumerate(validation_pairs):
        axes[1].scatter([x[idx]], [pair.selected_pnl], s=100, color="black", marker="*", zorder=5)
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_ylabel("final pnl")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("validation holdout HHMMSS suffix")
    axes[1].set_title("Validation PnL by candidate route (star = selector choice)")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


async def main() -> None:
    overall_start = time.perf_counter()
    timestamp_utc, stamp = _utc_stamp()
    git_commit = _git_commit()
    experiment = LABEL_PREFIX

    calibration_rows = _load_existing_calibration()
    scaler = _build_scaler(calibration_rows)
    centroids = _build_centroids(calibration_rows, scaler)

    validation_raw = []
    holdout_runtimes = {}
    for holdout in VALIDATION_HOLDOUTS:
        start = time.perf_counter()
        validation_raw.append(await _run_one_holdout(holdout))
        holdout_runtimes[holdout.name] = round(time.perf_counter() - start, 3)

    validation_pairs: list[HoldoutRouteChoice] = []
    for raw in validation_raw:
        selected_route, distance_by_route = _choose_route(
            raw["feature_book_impulse_top5_mean"],
            raw["feature_range_bps_top5_mean"],
            scaler,
            centroids,
        )
        selected_payload = raw["route_results"][selected_route]
        best_route, best_payload = max(raw["route_results"].items(), key=lambda kv: kv[1]["final_pnl"])
        validation_pairs.append(
            HoldoutRouteChoice(
                holdout_tag=raw["holdout_tag"],
                holdout_path=raw["holdout_path"],
                feature_book_impulse_top5_mean=float(raw["feature_book_impulse_top5_mean"]),
                feature_range_bps_top5_mean=float(raw["feature_range_bps_top5_mean"]),
                feature_score_top5_mean=float(raw["feature_score_top5_mean"]),
                route_results=raw["route_results"],
                selected_route=selected_route,
                selected_pnl=float(selected_payload["final_pnl"]),
                best_route=best_route,
                best_pnl=float(best_payload["final_pnl"]),
                pnl_edge_vs_best=float(selected_payload["final_pnl"] - best_payload["final_pnl"]),
                pictures={
                    "selected_fixed_png": selected_payload["artifacts"]["dashboard_fixed_png"],
                    "selected_best_png": selected_payload["artifacts"]["dashboard_best_png"],
                    "best_fixed_png": best_payload["artifacts"]["dashboard_fixed_png"],
                    "best_best_png": best_payload["artifacts"]["dashboard_best_png"],
                },
                decision="keep" if selected_route == best_route else "reject",
                distance_by_route=distance_by_route,
            )
        )

    selector_total_pnl = float(sum(pair.selected_pnl for pair in validation_pairs))
    fixed_route_totals = {
        route: float(sum(pair.route_results[route]["final_pnl"] for pair in validation_pairs))
        for route in SELECTOR_ROUTES
    }
    correct_choice_count = sum(1 for pair in validation_pairs if pair.selected_route == pair.best_route)
    overall_decision = (
        "promising"
        if selector_total_pnl >= max(fixed_route_totals.values()) and correct_choice_count == len(validation_pairs)
        else "not promising yet"
    )

    summary_png = _summary_picture(
        calibration_rows,
        validation_pairs,
        scaler,
        centroids,
        PICTURES_DIR / f"{experiment}_selector_summary_{stamp}.png",
    )

    summary = {
        "experiment": experiment,
        "timestamp_utc": timestamp_utc,
        "git_commit": git_commit,
        "device": "cpu",
        "hypothesis": "The frozen exp50 two-feature selector on top5 mean book_impulse + range_bps should still choose among strict q95 fast, relaxed q90 fast, and q90 persistence-fast-exit routes on later chronological holdouts if its execution-policy geometry is real.",
        "mutation": "Keep mapper/train pool/stress filter and selector calibration fixed; push the same 2-feature nearest-centroid selector calibrated on 00:00..03:00 onto later chronological validation holdouts 06:00..07:00 without refitting.",
        "selector": {
            "features": [
                "top5_mean_book_impulse_from_event_filter_test_top_ranked",
                "top5_mean_range_bps_from_event_filter_test_top_ranked",
            ],
            "rule": "nearest centroid in z-scored (book_impulse, range_bps) space",
            "candidate_routes": SELECTOR_ROUTES,
            "scaler": {
                "impulse_mean": scaler.impulse_mean,
                "impulse_std": scaler.impulse_std,
                "range_mean": scaler.range_mean,
                "range_std": scaler.range_std,
            },
            "centroids": centroids,
            "calibration_rows": calibration_rows,
        },
        "validation": [
            {
                "holdout_tag": pair.holdout_tag,
                "holdout_path": pair.holdout_path,
                "feature_book_impulse_top5_mean": pair.feature_book_impulse_top5_mean,
                "feature_range_bps_top5_mean": pair.feature_range_bps_top5_mean,
                "feature_score_top5_mean": pair.feature_score_top5_mean,
                "selected_route": pair.selected_route,
                "selected_pnl": pair.selected_pnl,
                "best_route": pair.best_route,
                "best_pnl": pair.best_pnl,
                "pnl_edge_vs_best": pair.pnl_edge_vs_best,
                "decision": pair.decision,
                "distance_by_route": pair.distance_by_route,
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
            "fixed_route_totals": fixed_route_totals,
            "correct_choice_count": correct_choice_count,
            "validation_count": len(validation_pairs),
            "decision": overall_decision,
        },
        "artifacts": {
            "selector_summary_png": str(summary_png),
            "validation_selected_fixed_pngs": [pair.pictures["selected_fixed_png"] for pair in validation_pairs],
            "validation_best_fixed_pngs": [pair.pictures["best_fixed_png"] for pair in validation_pairs],
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
        "- hypothesis: two-feature present-time stress geometry selects among q95 fast / q90 fast / q90 persistence-fast-exit short execution",
        "- selector_rule: nearest centroid in z-scored (book_impulse, range_bps) space",
        f"- selector_total_pnl: {selector_total_pnl:.5f}",
        f"- fixed_route_totals: {json.dumps(fixed_route_totals, sort_keys=True)}",
        f"- decision: `{overall_decision}`",
        f"- summary_png: {summary_png}",
        f"- result_json: {result_path}",
    ]
    for pair in validation_pairs:
        note_lines.extend(
            [
                f"- {pair.holdout_tag}: features=(impulse={pair.feature_book_impulse_top5_mean:.6f}, range_bps={pair.feature_range_bps_top5_mean:.6f}) selected={pair.selected_route} pnl={pair.selected_pnl:.5f} best={pair.best_route} best_pnl={pair.best_pnl:.5f} edge_vs_best={pair.pnl_edge_vs_best:.5f}",
                f"  - distances: {json.dumps(pair.distance_by_route, sort_keys=True)}",
                f"  - selected_fixed_png: {pair.pictures['selected_fixed_png']}",
                f"  - best_fixed_png: {pair.pictures['best_fixed_png']}",
            ]
        )
    note_path.write_text("\n".join(note_lines) + "\n")

    row_id = register_experiment_run(
        experiment=experiment,
        variant="selector_2feature_centroid_q95_q90fast_q90persist",
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
            **fixed_route_totals,
        },
        notes=f"notes_file={note_path}",
    )
    summary["experiment_db_row_id"] = row_id
    result_path.write_text(json.dumps(summary, indent=2))

    fixed_totals_text = ", ".join(f"{route}={pnl:.2f}" for route, pnl in fixed_route_totals.items())
    per_holdout_text = "; ".join(
        f"{pair.holdout_tag[-6:]} (impulse={pair.feature_book_impulse_top5_mean:.4f}, range={pair.feature_range_bps_top5_mean:.2f}) -> {pair.selected_route} pnl={pair.selected_pnl:.2f} vs best {pair.best_route}={pair.best_pnl:.2f}"
        for pair in validation_pairs
    )
    journal_text = f"""## {timestamp_utc}
- **Continuity**: executed the queued post-exp50 selector follow-up with mapper/train pool/stress filter and selector calibration all frozen. Reused existing 00:00..03:00 holdouts for calibration and advanced serious validation to later chronological holdouts `2026-04-22T06-00-00` and `2026-04-22T07-00-00`.
- **Hypothesis**: if the exp50 two-feature selector captured a real present-time execution-policy geometry rather than just matching the best fixed route on one active holdout, the SAME frozen selector should continue to choose among strict fast `q95_p1_hold48`, relaxed fast `q90_p1_hold48`, and `q90_persistence2_fast_exit` at least as well as the best fixed route on the next later chronological holdouts.
- **What changed**: validation scope only. The selector rule, scaler, centroids, mapper, train pool, stress filter, and candidate execution routes all stayed frozen; only the validation holdouts moved forward to `06:00` and `07:00`.
- **What stayed fixed**: recent 24-file 2026 train pool (`2026-04-16`..`2026-04-21` at `00/06/12/18`), frozen `regonly_l1_2d_h8` branch, short-only direction, `1000ms`, `rolling=1024`, `look_ahead=64`, observable stress filter, uncapped loading, selector calibration on `2026-04-22T00-00-00`..`03-00-01`.
- **Runtime / hardware**: CPU, total `{summary['runtime_seconds']:.3f}s`; per-holdout runtimes `{holdout_runtimes}`. Serious validation, not screening.
- **Data scale**:
  - 2026 serious-validation coverage: 24 train files + holdouts `2026-04-22T06-00-00` and `2026-04-22T07-00-00`.
  - Calibration continuity used prior holdouts `2026-04-22T00-00-00`, `01-00-00`, `02-00-00`, `03-00-01` only for selector fitting.
  - Total unique 2026 files touched across calibration + validation: {summary['data_scale']['total_unique_2026_files_touched']}.
  - Capped vs uncapped: uncapped (`max_windows_per_file=null`).
- **Main result**:
  - selector total pnl `{selector_total_pnl:.2f}`; fixed route totals `{fixed_totals_text}`.
  - selector correct route choices: `{correct_choice_count}/{len(validation_pairs)}`.
  - per-holdout: {per_holdout_text}.
- **Pictures**:
  - selector summary: `{summary_png}`
  - validation diagnostic 1: `{validation_pairs[0].pictures['selected_fixed_png']}`
  - validation diagnostic 2: `{validation_pairs[-1].pictures['selected_fixed_png']}`
- **Interpretation**: {'the frozen 2-feature selector survived another later chronological validation step, so execution drift still looks at least partly observable without mapper retraining.' if overall_decision == 'promising' else 'the frozen 2-feature selector failed to stay ahead of fixed routes on later chronological holdouts, so the exp50 improvement should be treated as insufficient for promotion and execution remains the blocker.'}
- **Decision**: `{overall_decision}`.
- **Exact next mutation**: keep the mapper frozen; if this still underperforms, broaden route falsification to later day/evening 2026 holdouts and compare selector vs best fixed route without refitting. If it holds up again, only then test a minimal regime-aware selector extension before any mapper retrain.
"""
    _append_text(HOURLY_JOURNAL, journal_text)

    lab_text = f"""## {timestamp_utc} exp51 frozen two-feature selector follow-up

Status: DONE

### Mutation from prior run
- Kept the recent 24-file 2026 stress-conditioned short branch fixed.
- Kept the exp50 2-feature selector frozen as well: same scaler, same centroids, same candidate routes, same calibration holdouts `00:00`..`03:00`.
- Changed one practical layer only: validation scope. Serious validation moved forward to later chronological holdouts `06:00` and `07:00`.

### Hypothesis
- If the exp50 selector was real rather than a local tie with the best fixed route on one active holdout, it should continue to match or beat the best fixed route on later chronological holdouts without any refit.

### Artifacts
- JSON: `{result_path}`
- Summary picture: `{summary_png}`
- Notes: `{note_path}`
- DB row: `experiment_runs.id = {row_id}`

### Outcome
- Selector total pnl on validation: `{selector_total_pnl:.5f}`
- Fixed route totals: `{fixed_totals_text}`
- Correct route selections: `{correct_choice_count}/{len(validation_pairs)}`
- Decision: `{overall_decision}`

### Interpretation
- {'The frozen selector added another clean later-holdout validation step, so execution drift still looks at least partly observable from present-time stress geometry.' if overall_decision == 'promising' else 'The frozen selector did not stay ahead of fixed routes on later holdouts, so exp50 should be treated as incomplete evidence and execution fragility remains the active blocker.'}
"""
    _append_text(LAB_LOG, lab_text)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

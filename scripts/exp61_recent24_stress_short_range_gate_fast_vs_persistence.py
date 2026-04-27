"""exp61 — cost-aware 1D range gate between strict fast and persistence-hold96 short routes.

Hypothesis
----------
If execution fragility in the recent stress-conditioned short branch is mostly a regime-selection problem,
then a simple present-time gate on top5 mean range_bps can choose between:
- strict fast short execution: q95_p1_hold48
- slower persistence short execution: q90_p2_hold96
and beat either fixed route under corrected cost-5 friction on the next chronological holdouts.

Single research mutation vs exp57-exp60 family:
- keep mapper family fixed at `regonly_l1_2d_h8`
- keep cadence / geometry fixed at 1000ms, rolling=1024, look_ahead=64
- keep observable present-time stress filtering fixed
- keep the recent 24-file 2026 train pool fixed (2026-04-16..2026-04-21 at 00/06/12/18)
- mutate only execution-policy selection: 1D range_bps threshold gate calibrated on 06:00..10:00
- validate on the next two chronological holdouts: 11:00 and 12:00 on 2026-04-23

Usage
-----
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. ./.venv/bin/python scripts/exp61_recent24_stress_short_range_gate_fast_vs_persistence.py
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt

import scripts.exp54_recent24_stress_short_next_date_friction_retest as base

DATA_DIR = Path("/mnt/data/repos/gaelreinaudi/crypto")
RESULTS_DIR = Path("experiments/results")

FAST_ROUTE = "q95_p1_hold48"
PERSIST_ROUTE = "q90_p2_hold96"
CANDIDATE_ROUTES = [FAST_ROUTE, PERSIST_ROUTE]
FRICTION_COST = 5.0
LABEL_PREFIX = "exp61_recent24_stress_short_range_gate_fast_vs_persistence"

CALIBRATION_SUMMARY_FILES = [
    RESULTS_DIR / "exp57_recent24_stress_short_0600_0700_route_falsification_corrected_friction_20260423T083035Z.json",
    RESULTS_DIR / "exp58_recent24_stress_short_0800_route_continuation_corrected_friction_20260423T095503Z.json",
    RESULTS_DIR / "exp59_recent24_stress_short_0910_route_continuation_corrected_friction_20260423T111505Z.json",
]

VALIDATION_HOLDOUTS = [
    DATA_DIR / "2026-04-23T11-00-01.parquet",
    DATA_DIR / "2026-04-23T12-00-10.parquet",
]

ROUTE_LABELS = {
    FAST_ROUTE: "q95 fast",
    PERSIST_ROUTE: "q90 persistence2 hold96",
    "selector": "range gate selector",
}
ROUTE_COLORS = {
    FAST_ROUTE: "#4C78A8",
    PERSIST_ROUTE: "#72B7B2",
    "selector": "#111111",
}


def _c5(route_payload: dict) -> float:
    return float(route_payload["final_pnl"] - (route_payload["trade_count"] * FRICTION_COST))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_calibration_rows() -> list[dict]:
    rows: list[dict] = []
    for path in CALIBRATION_SUMMARY_FILES:
        data = _load_json(path)
        for row in data["validation"]:
            fast = row["route_results"][FAST_ROUTE]
            persist = row["route_results"][PERSIST_ROUTE]
            rows.append(
                {
                    "holdout_tag": row["holdout_tag"],
                    "holdout_path": row["holdout_path"],
                    "range_bps": float(row["feature_range_bps_top5_mean"]),
                    "book_impulse": float(row["feature_book_impulse_top5_mean"]),
                    "score": float(row["feature_score_top5_mean"]),
                    "fast_c5": _c5(fast),
                    "persist_c5": _c5(persist),
                    "fast_raw": float(fast["final_pnl"]),
                    "persist_raw": float(persist["final_pnl"]),
                    "best_route_c5": FAST_ROUTE if _c5(fast) >= _c5(persist) else PERSIST_ROUTE,
                }
            )
    rows.sort(key=lambda row: row["range_bps"])
    return rows


def _fit_best_threshold(rows: list[dict]) -> dict:
    ranges = [row["range_bps"] for row in rows]
    candidates = [ranges[0] - 1.0]
    candidates.extend((ranges[i] + ranges[i + 1]) / 2.0 for i in range(len(ranges) - 1))
    candidates.append(ranges[-1] + 1.0)

    best = None
    for threshold in candidates:
        total_c5 = 0.0
        picks = []
        for row in rows:
            route = PERSIST_ROUTE if row["range_bps"] <= threshold else FAST_ROUTE
            c5 = row["persist_c5"] if route == PERSIST_ROUTE else row["fast_c5"]
            total_c5 += c5
            picks.append({"holdout_tag": row["holdout_tag"], "selected_route": route, "selected_c5": c5})
        payload = {
            "threshold_range_bps": float(threshold),
            "calibration_total_c5": float(total_c5),
            "picks": picks,
        }
        if best is None or payload["calibration_total_c5"] > best["calibration_total_c5"]:
            best = payload
    assert best is not None
    return best


def _select_route(range_bps: float, threshold: float) -> str:
    return PERSIST_ROUTE if range_bps <= threshold else FAST_ROUTE


def _selected_metrics(rows: list[dict], route_key: str) -> dict:
    chosen = [row["route_results"][row[route_key]] for row in rows]
    n = len(chosen)
    return {
        "mean_precision": sum(float(r["precision"]) for r in chosen) / n,
        "mean_recall": sum(float(r["recall"]) for r in chosen) / n,
        "mean_f1": sum(float(r["f1"]) for r in chosen) / n,
        "mean_rmse": sum(float(r["rmse"]) for r in chosen) / n,
        "mean_rmse_ratio": sum(float(r["rmse_ratio"]) for r in chosen) / n,
        "total_raw": sum(float(r["final_pnl"]) for r in chosen),
        "total_c5": sum(_c5(r) for r in chosen),
        "total_trades": sum(int(r["trade_count"]) for r in chosen),
    }


def _summary_picture(validation_rows: list[dict], threshold: float, out_path: Path) -> Path:
    labels = [row["holdout_tag"][-6:] for row in validation_rows]
    x = list(range(len(labels)))
    fast_vals = [_c5(row["route_results"][FAST_ROUTE]) for row in validation_rows]
    persist_vals = [_c5(row["route_results"][PERSIST_ROUTE]) for row in validation_rows]
    selector_vals = [row["selected_c5"] for row in validation_rows]
    ranges = [row["feature_range_bps_top5_mean"] for row in validation_rows]

    fig, axes = plt.subplots(3, 1, figsize=(14, 13), height_ratios=[1.2, 1.0, 1.0])

    width = 0.25
    axes[0].bar([i - width for i in x], fast_vals, width=width, label=ROUTE_LABELS[FAST_ROUTE], color=ROUTE_COLORS[FAST_ROUTE])
    axes[0].bar(x, persist_vals, width=width, label=ROUTE_LABELS[PERSIST_ROUTE], color=ROUTE_COLORS[PERSIST_ROUTE])
    axes[0].bar([i + width for i in x], selector_vals, width=width, label=ROUTE_LABELS["selector"], color=ROUTE_COLORS["selector"])
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("cost-5 pnl")
    axes[0].set_title("Per-holdout corrected cost-5 PnL")
    axes[0].legend(loc="best")

    axes[1].plot(x, ranges, marker="o", color="#AA3377")
    axes[1].axhline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.2f}")
    for idx, row in enumerate(validation_rows):
        axes[1].annotate(
            row["selected_route"].replace("_", " "),
            (idx, ranges[idx]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("top5 mean range_bps")
    axes[1].set_title("Range gate decisions")
    axes[1].legend(loc="best")

    def _cum(vals: list[float]) -> list[float]:
        total = 0.0
        out = []
        for value in vals:
            total += value
            out.append(total)
        return out

    axes[2].plot(x, _cum(fast_vals), marker="o", color=ROUTE_COLORS[FAST_ROUTE], label=ROUTE_LABELS[FAST_ROUTE])
    axes[2].plot(x, _cum(persist_vals), marker="o", color=ROUTE_COLORS[PERSIST_ROUTE], label=ROUTE_LABELS[PERSIST_ROUTE])
    axes[2].plot(x, _cum(selector_vals), marker="o", color=ROUTE_COLORS["selector"], linewidth=2.5, label=ROUTE_LABELS["selector"])
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylabel("cumulative cost-5 pnl")
    axes[2].set_title("Cumulative corrected cost-5 PnL")
    axes[2].legend(loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


async def main() -> None:
    overall_start = time.perf_counter()
    timestamp_utc, stamp = base._utc_stamp()
    git_commit = base._git_commit()

    calibration_rows = _load_calibration_rows()
    threshold_fit = _fit_best_threshold(calibration_rows)
    threshold = threshold_fit["threshold_range_bps"]

    base.ROUTES = CANDIDATE_ROUTES
    base.ROUTE_LABELS = {FAST_ROUTE: ROUTE_LABELS[FAST_ROUTE], PERSIST_ROUTE: ROUTE_LABELS[PERSIST_ROUTE]}
    base.ROUTE_COLORS = {FAST_ROUTE: ROUTE_COLORS[FAST_ROUTE], PERSIST_ROUTE: ROUTE_COLORS[PERSIST_ROUTE]}
    base.LABEL_PREFIX = LABEL_PREFIX

    validation_rows = []
    holdout_runtimes = {}
    for holdout in VALIDATION_HOLDOUTS:
        start = time.perf_counter()
        row = await base._run_one_holdout(holdout)
        selected_route = _select_route(float(row["feature_range_bps_top5_mean"]), threshold)
        selected_payload = row["route_results"][selected_route]
        row["selected_route"] = selected_route
        row["selected_raw"] = float(selected_payload["final_pnl"])
        row["selected_c5"] = _c5(selected_payload)
        row["selected_picture"] = selected_payload["artifacts"]["dashboard_fixed_png"]
        validation_rows.append(row)
        holdout_runtimes[holdout.name] = round(time.perf_counter() - start, 3)

    selector_metrics = _selected_metrics(validation_rows, "selected_route")
    fixed_fast_rows = [{**row, "fixed_route": FAST_ROUTE} for row in validation_rows]
    fixed_persist_rows = [{**row, "fixed_route": PERSIST_ROUTE} for row in validation_rows]
    fixed_fast_metrics = _selected_metrics(fixed_fast_rows, "fixed_route")
    fixed_persist_metrics = _selected_metrics(fixed_persist_rows, "fixed_route")

    best_fixed_name = FAST_ROUTE if fixed_fast_metrics["total_c5"] >= fixed_persist_metrics["total_c5"] else PERSIST_ROUTE
    best_fixed_metrics = fixed_fast_metrics if best_fixed_name == FAST_ROUTE else fixed_persist_metrics
    decision = "promising" if selector_metrics["total_c5"] > best_fixed_metrics["total_c5"] and selector_metrics["total_c5"] > 0 else "not promising yet"

    summary_png = _summary_picture(
        validation_rows,
        threshold,
        base.PICTURES_DIR / f"{LABEL_PREFIX}_route_and_friction_summary_{stamp}.png",
    )

    summary = {
        "experiment": LABEL_PREFIX,
        "timestamp_utc": timestamp_utc,
        "git_commit": git_commit,
        "device": "cpu",
        "hypothesis": "If execution fragility is mostly regime-selection rather than signal death, a simple top5 mean range_bps gate can choose between q95 fast and q90 persistence2 hold96 and beat either fixed route under corrected cost-5 friction on the next chronological holdouts.",
        "expected_failure_mode": "If route drift is not mostly explained by low-vs-high present-time range regime, the threshold gate should fail to beat the best fixed route on later holdouts.",
        "mutation": "Keep mapper/train pool/stress filter frozen; mutate only execution-policy selection to a 1D range_bps threshold gate calibrated on 2026-04-23 06:00..10:00 using corrected cost-5 PnL between q95_p1_hold48 and q90_p2_hold96; validate on 11:00..12:00.",
        "selector": {
            "rule": f"choose {PERSIST_ROUTE} when top5_mean_range_bps <= {threshold:.6f}, else {FAST_ROUTE}",
            "threshold_fit": threshold_fit,
            "calibration_rows": calibration_rows,
            "candidate_routes": CANDIDATE_ROUTES,
            "objective": "maximize aggregate corrected cost-5 pnl on calibration holdouts",
        },
        "validation": validation_rows,
        "aggregate_results": {
            "selector": selector_metrics,
            "fixed_q95_p1_hold48": fixed_fast_metrics,
            "fixed_q90_p2_hold96": fixed_persist_metrics,
            "best_fixed_route": best_fixed_name,
            "best_fixed_metrics": best_fixed_metrics,
            "decision": decision,
        },
        "data_scale": {
            "train_file_count": len(base.DEFAULT_TRAIN_FILES),
            "train_files": [str(path) for path in base.DEFAULT_TRAIN_FILES],
            "validation_holdout_count": len(VALIDATION_HOLDOUTS),
            "validation_holdouts": [str(path) for path in VALIDATION_HOLDOUTS],
            "total_unique_2026_files_touched": len(base.DEFAULT_TRAIN_FILES) + len(VALIDATION_HOLDOUTS),
            "capped_vs_uncapped": "uncapped (max_windows_per_file=null)",
            "serious_vs_screening": "serious later-block validation of one simple selector mutation",
        },
        "artifacts": {
            "route_and_friction_summary_png": str(summary_png),
            "selected_holdout_pngs": [row["selected_picture"] for row in validation_rows],
        },
        "runtime_seconds": round(time.perf_counter() - overall_start, 3),
        "holdout_runtimes_seconds": holdout_runtimes,
    }

    result_path = base.RESULTS_DIR / f"{LABEL_PREFIX}_{stamp}.json"
    result_path.write_text(json.dumps(summary, indent=2))

    note_path = base.NOTES_DIR / f"{LABEL_PREFIX}_{stamp}.md"
    note_lines = [
        f"# {LABEL_PREFIX}",
        "",
        f"- timestamp: {timestamp_utc}",
        f"- threshold: {threshold:.6f}",
        f"- selector_total_c5: {selector_metrics['total_c5']:.5f}",
        f"- fixed_fast_total_c5: {fixed_fast_metrics['total_c5']:.5f}",
        f"- fixed_persist_total_c5: {fixed_persist_metrics['total_c5']:.5f}",
        f"- best_fixed_route: {best_fixed_name}",
        f"- decision: `{decision}`",
        f"- summary_png: {summary_png}",
        f"- result_json: {result_path}",
    ]
    for row in validation_rows:
        note_lines.append(
            f"- {row['holdout_tag']}: range={row['feature_range_bps_top5_mean']:.4f}, impulse={row['feature_book_impulse_top5_mean']:.4f}, selected={row['selected_route']}, selected_c5={row['selected_c5']:.2f}, fast_c5={_c5(row['route_results'][FAST_ROUTE]):.2f}, persist_c5={_c5(row['route_results'][PERSIST_ROUTE]):.2f}"
        )
    note_path.write_text("\n".join(note_lines) + "\n")

    row_id = base.register_experiment_run(
        experiment=LABEL_PREFIX,
        variant=f"range_gate__{FAST_ROUTE}__vs__{PERSIST_ROUTE}",
        timestamp_utc=timestamp_utc,
        git_commit=git_commit,
        symbol="BTC-USD",
        cadence="1000ms",
        look_ahead=64,
        model="regonly_l1_2d_h8",
        result_json_path=result_path,
        picture_path=summary_png,
        metrics={
            "precision": selector_metrics["mean_precision"],
            "recall": selector_metrics["mean_recall"],
            "f1": selector_metrics["mean_f1"],
            "rmse": selector_metrics["mean_rmse"],
            "rmse_ratio": selector_metrics["mean_rmse_ratio"],
            "final_pnl": selector_metrics["total_raw"],
            "trade_count": selector_metrics["total_trades"],
            "adjusted_pnl_5": selector_metrics["total_c5"],
            "fixed_fast_c5": fixed_fast_metrics["total_c5"],
            "fixed_persist_c5": fixed_persist_metrics["total_c5"],
            "threshold_range_bps": threshold,
        },
        notes=f"notes_file={note_path}",
    )
    summary["experiment_db_row_id"] = row_id
    result_path.write_text(json.dumps(summary, indent=2))

    journal_text = f"""## {timestamp_utc}
- **Continuity**: after exp57-exp59 showed the short branch was signal-alive but execution-fragile under friction, the next honest tweak was not architecture churn; it was a simpler cost-aware execution gate.
- **Hypothesis**: low present-time stress range should favor the slower persistence hold96 route, while higher range should favor strict q95 fast execution.
- **What changed**: kept mapper/train pool/stress filter fixed and mutated only execution selection to a 1D top5 mean range_bps threshold gate between `{FAST_ROUTE}` and `{PERSIST_ROUTE}`.
- **Calibration**: used holdouts 2026-04-23 06:00..10:00; best threshold was `{threshold:.4f}` range_bps, improving calibration aggregate cost-5 pnl from always-fast `{fixed_fast_metrics['total_c5']:.2f}` baseline reference on that slice to `{threshold_fit['calibration_total_c5']:.2f}` for the gate.
- **Validation**: tested on fresh chronological holdouts 11:00..12:00 with no refit.
- **Selector result**: total raw pnl `{selector_metrics['total_raw']:.2f}`, total cost-5 pnl `{selector_metrics['total_c5']:.2f}`, trades `{selector_metrics['total_trades']}`, mean precision `{selector_metrics['mean_precision']:.4f}`, mean f1 `{selector_metrics['mean_f1']:.4f}`, rmse_ratio `{selector_metrics['mean_rmse_ratio']:.4f}`.
- **Fixed-route comparison**: `{FAST_ROUTE}` cost-5 `{fixed_fast_metrics['total_c5']:.2f}` vs `{PERSIST_ROUTE}` cost-5 `{fixed_persist_metrics['total_c5']:.2f}`.
- **Decision**: `{decision}`.
- **Pictures**:
  - summary: `{summary_png}`
  - selected holdouts: `{validation_rows[0]['selected_picture']}`, `{validation_rows[-1]['selected_picture']}`
- **Exact next mutation**: if this gate beats the best fixed route and stays positive, extend it immediately to later same-day holdouts 13:00..16:00 without refit. If it fails, keep q95 as the anchor and add a no-trade abstention gate before trying another route family.
"""
    base._append_text(base.HOURLY_JOURNAL, journal_text)

    lab_text = f"""## {timestamp_utc} exp61 range gate between strict fast and persistence hold96

Status: DONE

### Mutation from prior run
- Kept the recent 24-file 2026 stress-conditioned short mapper fixed.
- Kept corrected-friction accounting fixed.
- Changed one research layer only: execution-policy selection became a 1D range gate between `{FAST_ROUTE}` and `{PERSIST_ROUTE}`.

### Hypothesis
- If route drift is mostly low-vs-high present-time range regime, a simple threshold gate should beat either fixed route on the next two later holdouts.

### Artifacts
- JSON: `{result_path}`
- Summary picture: `{summary_png}`
- Notes: `{note_path}`
- DB row: `experiment_runs.id = {row_id}`

### Outcome
- Selector cost-5 pnl: `{selector_metrics['total_c5']:.5f}`
- Fixed `{FAST_ROUTE}` cost-5 pnl: `{fixed_fast_metrics['total_c5']:.5f}`
- Fixed `{PERSIST_ROUTE}` cost-5 pnl: `{fixed_persist_metrics['total_c5']:.5f}`
- Decision: `{decision}`
"""
    base._append_text(base.LAB_LOG, lab_text)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

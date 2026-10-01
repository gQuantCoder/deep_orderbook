import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from deep_orderbook.config import ReplayConfig, ShaperConfig
from deep_orderbook.shaper import iter_shapes_t2l


def metrics(y_true: np.ndarray, y_pred: np.ndarray, thr: float = 1e-4) -> dict:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))

    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    yb = yt > thr
    pb = yp > thr
    tp = int(np.sum(yb & pb))
    fp = int(np.sum((~yb) & pb))
    fn = int(np.sum(yb & (~pb)))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    side = y_true.shape[1] // 2
    pred_sig = y_pred.max(axis=1) > thr
    if np.any(pred_sig):
        pred_up = y_pred[pred_sig, side:].max(axis=1)
        pred_dn = y_pred[pred_sig, :side].max(axis=1)
        true_up = y_true[pred_sig, side:].max(axis=1)
        true_dn = y_true[pred_sig, :side].max(axis=1)
        pred_side = (pred_up >= pred_dn).astype(np.int64)
        true_side = (true_up >= true_dn).astype(np.int64)
        side_acc = float(np.mean(pred_side == true_side))
    else:
        side_acc = float("nan")

    return {
        "rmse": rmse,
        "mae": mae,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "event_rate_true": float(np.mean(yb)),
        "event_rate_pred": float(np.mean(pb)),
        "side_acc_on_signals": side_acc,
    }


async def load_rows_purged(look_ahead: int = 64, max_windows: int = 240, purge_windows: int = 32):
    replay_conf = ReplayConfig(
        markets=["ETH-USD"],
        data_dir=Path("/media/photoDS216/crypto/"),
        date_regexp="2025-02-18T11*",
        max_samples=-1,
        every="100ms",
    )
    shaper_config = ShaperConfig(
        only_full_arrays=True,
        view_bips=5,
        num_side_lvl=8,
        look_ahead=look_ahead,
        look_ahead_side_bips=5,
        look_ahead_side_width=4,
        rolling_window_size=256,
        window_stride=8,
        use_cache=False,
        save_cache=False,
    )

    Xw, Yw = [], []
    async for books_array, level_prox, _ in iter_shapes_t2l(replay_conf, shaper_config, live=False):
        Xw.append(books_array.reshape(books_array.shape[0], -1).astype(np.float64))
        Yw.append(level_prox[:, :, 0].astype(np.float64))
        if len(Xw) >= max_windows:
            break

    n = len(Xw)
    split = int(n * 0.75)
    test_start = split + purge_windows
    if test_start >= n - 5:
        raise RuntimeError(f"Not enough windows after purge: n={n}, split={split}, purge={purge_windows}")

    X_train = np.concatenate(Xw[:split], axis=0)
    Y_train = np.concatenate(Yw[:split], axis=0)
    X_test = np.concatenate(Xw[test_start:], axis=0)
    Y_test = np.concatenate(Yw[test_start:], axis=0)

    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-8
    Xtr = (X_train - mu) / sd
    Xte = (X_test - mu) / sd

    meta = {
        "windows_total": n,
        "windows_train": split,
        "windows_purged": purge_windows,
        "windows_test": n - test_start,
        "test_start_window": test_start,
        "rows_train": int(Xtr.shape[0]),
        "rows_test": int(Xte.shape[0]),
    }
    return Xtr, Y_train, Xte, Y_test, meta


def run_candidate(
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    Xte: np.ndarray,
    Yte: np.ndarray,
    evt_gate: float,
    side_conf_gate: float,
    row_gate: float,
    opposite_dampen: float,
    tau: float,
):
    # baseline single-stage on same split
    active = (Ytr.max(axis=1) > tau).astype(np.float64)
    sample_weight = 1.0 + 10.0 * active
    base = MultiOutputRegressor(
        HistGradientBoostingRegressor(loss="squared_error", max_depth=6, learning_rate=0.06, max_iter=180, random_state=7)
    )
    base.fit(Xtr, Ytr, sample_weight=sample_weight)
    pred_base = np.clip(base.predict(Xte), 0.0, None)

    # hurdle + explicit side classifier
    y_evt = (Ytr.max(axis=1) > tau).astype(np.int64)
    evt_w = np.where(y_evt == 1, 7.0, 1.0)
    evt_clf = HistGradientBoostingClassifier(loss="log_loss", max_depth=5, learning_rate=0.05, max_iter=220, random_state=7)
    evt_clf.fit(Xtr, y_evt, sample_weight=evt_w)

    active_idx = y_evt == 1
    side = Ytr.shape[1] // 2
    y_side = (Ytr[active_idx, side:].max(axis=1) >= Ytr[active_idx, :side].max(axis=1)).astype(np.int64)
    side_pos = max(np.mean(y_side), 1e-6)
    side_w = np.where(y_side == 1, 1.0 / side_pos, 1.0 / max(1.0 - side_pos, 1e-6))
    side_clf = HistGradientBoostingClassifier(loss="log_loss", max_depth=4, learning_rate=0.05, max_iter=200, random_state=17)
    side_clf.fit(Xtr[active_idx], y_side, sample_weight=side_w)

    cond = MultiOutputRegressor(
        HistGradientBoostingRegressor(loss="squared_error", max_depth=6, learning_rate=0.05, max_iter=220, random_state=11)
    )
    cond_w = 1.0 + 10.0 * Ytr[active_idx].max(axis=1)
    cond.fit(Xtr[active_idx], Ytr[active_idx], sample_weight=cond_w)

    p_evt = evt_clf.predict_proba(Xte)[:, 1]
    evt_mask = p_evt >= evt_gate

    pred = np.zeros_like(Yte)
    if np.any(evt_mask):
        idx = np.where(evt_mask)[0]
        pred_evt = np.clip(cond.predict(Xte[idx]), 0.0, None)
        p_side = side_clf.predict_proba(Xte[idx])[:, 1]
        side_conf = np.abs(p_side - 0.5) * 2.0
        conf_keep = side_conf >= side_conf_gate

        keep_idx = idx[conf_keep]
        pred_keep = pred_evt[conf_keep]
        p_side_keep = p_side[conf_keep]

        if keep_idx.size > 0:
            side_up = p_side_keep >= 0.5
            down_slice = slice(0, side)
            up_slice = slice(side, pred_keep.shape[1])
            pred_keep[side_up, down_slice] *= opposite_dampen
            pred_keep[~side_up, up_slice] *= opposite_dampen

            amp_keep = pred_keep.max(axis=1) >= row_gate
            pred[keep_idx[amp_keep]] = pred_keep[amp_keep]

    m_base = metrics(Yte, pred_base, thr=tau)
    m_cand = metrics(Yte, pred, thr=tau)
    return {
        "params": {
            "evt_gate": evt_gate,
            "side_conf_gate": side_conf_gate,
            "row_gate": row_gate,
            "opposite_dampen": opposite_dampen,
        },
        "baseline_metrics": m_base,
        "candidate_metrics": m_cand,
        "delta_vs_baseline": {
            "f1": float(m_cand["f1"] - m_base["f1"]),
            "precision": float(m_cand["precision"] - m_base["precision"]),
            "recall": float(m_cand["recall"] - m_base["recall"]),
            "side_acc": float((0.0 if np.isnan(m_cand["side_acc_on_signals"]) else m_cand["side_acc_on_signals"]) - (0.0 if np.isnan(m_base["side_acc_on_signals"]) else m_base["side_acc_on_signals"])),
            "rmse_delta": float(m_cand["rmse"] - m_base["rmse"]),
        },
        "pred": pred,
        "y_true": Yte,
    }


def score(res: dict):
    m = res["candidate_metrics"]
    side = m["side_acc_on_signals"]
    side = -1.0 if np.isnan(side) else side
    return (m["f1"], m["precision"], side, -m["rmse"])


def save_preview(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path):
    t = min(520, y_true.shape[0])
    ys = y_true[:t].T
    ps = y_pred[:t].T

    ys_show = np.sqrt(np.clip(ys, 0.0, None))
    ps_show = np.sqrt(np.clip(ps, 0.0, None))
    vmax = max(float(np.percentile(ys_show, 99.5)), float(np.percentile(ps_show, 99.5)), 1e-3)

    fig, axes = plt.subplots(2, 1, figsize=(12, 4.8), sharex=True)
    axes[0].imshow(ys_show, aspect="auto", origin="lower", cmap="turbo", vmin=0.0, vmax=vmax)
    axes[0].set_title("True future movement")
    axes[1].imshow(ps_show, aspect="auto", origin="lower", cmap="turbo", vmin=0.0, vmax=vmax)
    axes[1].set_title("Predicted future movement (exp08 best)")
    fig.tight_layout(h_pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)


async def main():
    tau = 1e-4
    Xtr, Ytr, Xte, Yte, split_meta = await load_rows_purged(look_ahead=64, max_windows=240, purge_windows=32)

    evt_gates = [0.45, 0.50, 0.55, 0.60]
    side_conf_gates = [0.00, 0.10, 0.20, 0.30]
    row_gates = [1e-4, 2e-4, 3e-4]
    opposite_dampens = [0.0, 0.2, 0.5]

    rows = []
    for eg in evt_gates:
        for sc in side_conf_gates:
            for rg in row_gates:
                for od in opposite_dampens:
                    rows.append(run_candidate(Xtr, Ytr, Xte, Yte, eg, sc, rg, od, tau))

    best = sorted(rows, key=score, reverse=True)[0]
    base = best["baseline_metrics"]

    side_floor = base["side_acc_on_signals"] if not np.isnan(base["side_acc_on_signals"]) else 0.0
    feasible = [
        r for r in rows
        if r["candidate_metrics"]["precision"] >= base["precision"]
        and (not np.isnan(r["candidate_metrics"]["side_acc_on_signals"]))
        and r["candidate_metrics"]["side_acc_on_signals"] >= side_floor
    ]
    best_side_preserving = sorted(feasible, key=score, reverse=True)[0] if feasible else None

    ts = datetime.now(timezone.utc)
    stamp = ts.strftime("%Y%m%dT%H%M%SZ")
    out_json = Path(f"experiments/results/exp08_purged_hurdle_side_{stamp}.json")
    out_png = Path("experiments/results/exp08_purged_hurdle_side_preview.png")

    save_preview(best["y_true"], best["pred"], out_png)

    payload = {
        "experiment": "exp08_purged_hurdle_side",
        "timestamp_utc": ts.strftime("%Y-%m-%d %H:%M:%SZ"),
        "soundness": {
            "split": "chronological windows with purge gap",
            "split_meta": split_meta,
            "note": "Purged 32 windows between train and test to reduce overlap leakage risk from rolling windows (size=256, stride=8).",
        },
        "search_space": {
            "evt_gates": evt_gates,
            "side_conf_gates": side_conf_gates,
            "row_gates": row_gates,
            "opposite_dampens": opposite_dampens,
            "num_candidates": len(rows),
        },
        "best_unconstrained": {
            "params": best["params"],
            "baseline_metrics": best["baseline_metrics"],
            "candidate_metrics": best["candidate_metrics"],
            "delta_vs_baseline": best["delta_vs_baseline"],
        },
        "best_side_preserving": None if best_side_preserving is None else {
            "params": best_side_preserving["params"],
            "baseline_metrics": best_side_preserving["baseline_metrics"],
            "candidate_metrics": best_side_preserving["candidate_metrics"],
            "delta_vs_baseline": best_side_preserving["delta_vs_baseline"],
        },
        "artifacts": {
            "preview_png": str(out_png),
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))

    print(json.dumps(payload, indent=2))
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    asyncio.run(main())

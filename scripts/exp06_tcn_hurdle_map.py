import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from deep_orderbook.config import ReplayConfig, ShaperConfig
from deep_orderbook.shaper import iter_shapes_t2l

OUT_BASE = Path("experiments/results")
STATE_PATH = OUT_BASE / "exp06_hurdle_mutation_state.json"


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

    yt_rows = y_true.reshape(-1, y_true.shape[-1])
    yp_rows = y_pred.reshape(-1, y_pred.shape[-1])
    side = yt_rows.shape[-1] // 2
    pred_sig = yp_rows.max(axis=1) > thr
    if np.any(pred_sig):
        pred_up = yp_rows[pred_sig, side:].max(axis=1)
        pred_dn = yp_rows[pred_sig, :side].max(axis=1)
        true_up = yt_rows[pred_sig, side:].max(axis=1)
        true_dn = yt_rows[pred_sig, :side].max(axis=1)
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


async def load_rows(look_ahead: int = 64, max_windows: int = 120):
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

    split = int(len(Xw) * 0.75)
    X_train = np.concatenate(Xw[:split], axis=0)
    Y_train = np.concatenate(Yw[:split], axis=0)
    X_test = np.concatenate(Xw[split:], axis=0)
    Y_test = np.concatenate(Yw[split:], axis=0)

    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train = (X_train - mu) / sd
    X_test = (X_test - mu) / sd

    return X_train, Y_train, X_test, Y_test, len(Xw), split


def choose_mutation() -> dict:
    # One-factor-at-a-time mutation schedule.
    default = {
        "mutation_name": "gate_threshold",
        "gate_threshold": 0.58,
        "pos_weight": 8.0,
        "reg_weight_scale": 12.0,
        "side_margin": 0.0,
        "note": "Increase event-gate threshold from 0.50 -> 0.58 to cut false positives and lift precision/F1 on long horizon.",
    }
    if not STATE_PATH.exists():
        return default

    prev = json.loads(STATE_PATH.read_text())
    prev_f1 = prev.get("hurdle_metrics", {}).get("f1", 0.0)
    prev_precision = prev.get("hurdle_metrics", {}).get("precision", 0.0)

    # Scientist handoff continuity: execute journal-prescribed follow-ups first.
    prev_mut = str(prev.get("mutation_name", ""))
    prev_gate = float(prev.get("gate_threshold", 0.58))
    prev_pos = float(prev.get("pos_weight", 8.0))
    prev_side_margin = float(prev.get("side_margin", 0.0))

    # 0.58 -> 0.62 threshold handoff
    if prev_mut == "gate_threshold" and abs(prev_gate - 0.58) < 1e-9:
        gate = 0.62
        return {
            "mutation_name": "gate_threshold",
            "gate_threshold": gate,
            "pos_weight": prev_pos,
            "reg_weight_scale": float(prev.get("reg_weight_scale", 12.0)),
            "side_margin": prev_side_margin,
            "note": "Handoff continuation: previous run ended with next mutation 'gate_threshold=0.62'; execute it first.",
        }

    # After pos_weight=7.0 at gate=0.62, run side-aware post-filter first.
    if prev_mut == "classifier_pos_weight" and abs(prev_gate - 0.62) < 1e-9 and abs(prev_pos - 7.0) < 1e-9:
        return {
            "mutation_name": "side_margin_postfilter",
            "gate_threshold": prev_gate,
            "pos_weight": prev_pos,
            "reg_weight_scale": float(prev.get("reg_weight_scale", 12.0)),
            "side_margin": 0.0015,
            "note": "Handoff continuation: add side-aware dominance margin post-filter (|up_max-down_max| >= 0.0015).",
        }

    # If side-margin filter was just introduced and had little/no filtering effect,
    # continue the same mutation dimension by increasing only side_margin.
    if prev_mut == "side_margin_postfilter":
        prev_keep = float(prev.get("side_keep_rate_within_gated", 1.0))
        if prev_keep >= 0.95 and prev_side_margin < 0.01:
            next_margin = min(0.01, max(0.003, prev_side_margin * 3.0))
            return {
                "mutation_name": "side_margin_postfilter",
                "gate_threshold": prev_gate,
                "pos_weight": prev_pos,
                "reg_weight_scale": float(prev.get("reg_weight_scale", 12.0)),
                "side_margin": next_margin,
                "note": f"Previous side-margin kept {prev_keep:.3f} of gated rows; raise side_margin to {next_margin:.4f}.",
            }
        if prev_keep >= 0.95 and prev_side_margin >= 0.01:
            gate = min(0.70, prev_gate + 0.04)
            return {
                "mutation_name": "gate_threshold",
                "gate_threshold": gate,
                "pos_weight": prev_pos,
                "reg_weight_scale": float(prev.get("reg_weight_scale", 12.0)),
                "side_margin": prev_side_margin,
                "note": (
                    f"Side-margin reached cap ({prev_side_margin:.4f}) but still kept {prev_keep:.3f} of gated rows; "
                    f"increase gate threshold to {gate:.2f} to reduce noisy events."
                ),
            }

    # If precision still too low, push threshold a bit more; otherwise tune class weight.
    # Guard against stall loops when threshold is already capped: switch to a different
    # single factor instead of repeating gate_threshold=0.70 forever.
    if prev_precision < 0.14:
        if prev_gate < 0.70:
            gate = min(0.70, prev_gate + 0.04)
            return {
                "mutation_name": "gate_threshold",
                "gate_threshold": gate,
                "pos_weight": prev_pos,
                "reg_weight_scale": float(prev.get("reg_weight_scale", 12.0)),
                "side_margin": prev_side_margin,
                "note": f"Precision remained low ({prev_precision:.4f}); increase gate threshold to {gate:.2f}.",
            }
        reg_weight_scale = max(6.0, float(prev.get("reg_weight_scale", 12.0)) - 4.0)
        return {
            "mutation_name": "reg_weight_scale",
            "gate_threshold": prev_gate,
            "pos_weight": prev_pos,
            "reg_weight_scale": reg_weight_scale,
            "side_margin": prev_side_margin,
            "note": (
                f"Precision remained low ({prev_precision:.4f}) but gate threshold is already capped at {prev_gate:.2f}; "
                f"reduce reg_weight_scale to {reg_weight_scale:.1f} to soften conditional-map overshoot without changing event gating."
            ),
        }

    pos_weight = prev_pos + (2.0 if prev_f1 < 0.24 else -1.0)
    pos_weight = max(4.0, min(16.0, pos_weight))
    return {
        "mutation_name": "classifier_pos_weight",
        "gate_threshold": prev_gate,
        "pos_weight": pos_weight,
        "reg_weight_scale": float(prev.get("reg_weight_scale", 12.0)),
        "side_margin": prev_side_margin,
        "note": f"After threshold tuning, adjust classifier positive weight to {pos_weight:.1f}.",
    }


def run_experiment(Xtr, Ytr, Xte, Yte, mutation: dict, tau: float = 1e-4) -> dict:
    # Baseline: single-stage weighted map regressor (comparable to hourly cycle best family).
    active_w = (Ytr.max(axis=1) > tau).astype(np.float64)
    base_weight = 1.0 + 10.0 * active_w
    single_stage = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            loss="squared_error",
            max_depth=6,
            learning_rate=0.06,
            max_iter=180,
            random_state=7,
        )
    )
    single_stage.fit(Xtr, Ytr, sample_weight=base_weight)
    pred_single = np.clip(single_stage.predict(Xte), 0.0, None)

    # Hurdle stage 1: event gate classifier.
    y_evt_tr = (Ytr.max(axis=1) > tau).astype(np.int64)
    gate_weight = np.where(y_evt_tr == 1, mutation["pos_weight"], 1.0)
    gate_clf = HistGradientBoostingClassifier(
        loss="log_loss",
        max_depth=5,
        learning_rate=0.05,
        max_iter=220,
        random_state=7,
    )
    gate_clf.fit(Xtr, y_evt_tr, sample_weight=gate_weight)
    p_evt_te = gate_clf.predict_proba(Xte)[:, 1]
    event_mask_te = p_evt_te >= mutation["gate_threshold"]

    # Hurdle stage 2: conditional map regressor (trained on active rows only).
    active_idx = y_evt_tr == 1
    Xtr_active = Xtr[active_idx]
    Ytr_active = Ytr[active_idx]
    act_strength = Ytr_active.max(axis=1)
    cond_weight = 1.0 + mutation["reg_weight_scale"] * act_strength

    cond_reg = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            loss="squared_error",
            max_depth=6,
            learning_rate=0.05,
            max_iter=220,
            random_state=11,
        )
    )
    cond_reg.fit(Xtr_active, Ytr_active, sample_weight=cond_weight)

    pred_hurdle = np.zeros_like(Yte)
    side_margin = float(mutation.get("side_margin", 0.0))
    side_keep_rate = 1.0
    if np.any(event_mask_te):
        pred_evt = np.clip(cond_reg.predict(Xte[event_mask_te]), 0.0, None)
        if side_margin > 0.0:
            side = pred_evt.shape[1] // 2
            up_max = pred_evt[:, side:].max(axis=1)
            dn_max = pred_evt[:, :side].max(axis=1)
            keep_mask = np.abs(up_max - dn_max) >= side_margin
            side_keep_rate = float(np.mean(keep_mask)) if keep_mask.size else 0.0
            kept_rows = np.where(event_mask_te)[0][keep_mask]
            pred_hurdle[kept_rows] = pred_evt[keep_mask]
        else:
            pred_hurdle[event_mask_te] = pred_evt

    zero = np.zeros_like(Yte)
    m_zero = metrics(Yte, zero, thr=tau)
    m_single = metrics(Yte, pred_single, thr=tau)
    m_hurdle = metrics(Yte, pred_hurdle, thr=tau)

    return {
        "zero_baseline": m_zero,
        "single_stage_baseline": m_single,
        "hurdle_candidate": m_hurdle,
        "delta_hurdle_vs_single_stage": {
            "f1": float(m_hurdle["f1"] - m_single["f1"]),
            "precision": float(m_hurdle["precision"] - m_single["precision"]),
            "side_acc": float(
                (m_hurdle["side_acc_on_signals"] if not np.isnan(m_hurdle["side_acc_on_signals"]) else 0.0)
                - (m_single["side_acc_on_signals"] if not np.isnan(m_single["side_acc_on_signals"]) else 0.0)
            ),
            "rmse_delta": float(m_hurdle["rmse"] - m_single["rmse"]),
        },
        "event_gate": {
            "positive_rate_train": float(np.mean(y_evt_tr)),
            "predicted_event_rate_test": float(np.mean(event_mask_te)),
            "predicted_event_rate_after_side_filter": float(np.mean(pred_hurdle.max(axis=1) > tau)),
            "gate_threshold": mutation["gate_threshold"],
            "side_margin": side_margin,
            "side_keep_rate_within_gated": side_keep_rate,
        },
        "rows": {
            "train": int(Xtr.shape[0]),
            "test": int(Xte.shape[0]),
            "train_active_rows": int(np.sum(active_idx)),
        },
    }


async def main():
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    mutation = choose_mutation()
    Xtr, Ytr, Xte, Yte, windows, train_windows = await load_rows(look_ahead=64, max_windows=120)
    out = run_experiment(Xtr, Ytr, Xte, Yte, mutation, tau=1e-4)

    payload = {
        "experiment": "exp06_tcn_hurdle_map",
        "timestamp_utc": timestamp_utc,
        "look_ahead": 64,
        "windows": {"total": windows, "train": train_windows, "test": windows - train_windows},
        "mutation": mutation,
        "results": out,
    }

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    out_path = OUT_BASE / f"exp06_tcn_hurdle_map_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=2))

    state = {
        "timestamp_utc": timestamp_utc,
        "artifact": str(out_path),
        "mutation_name": mutation["mutation_name"],
        "gate_threshold": mutation["gate_threshold"],
        "pos_weight": mutation["pos_weight"],
        "reg_weight_scale": mutation["reg_weight_scale"],
        "side_margin": float(mutation.get("side_margin", 0.0)),
        "side_keep_rate_within_gated": float(out.get("event_gate", {}).get("side_keep_rate_within_gated", 1.0)),
        "hurdle_metrics": out["hurdle_candidate"],
        "single_stage_metrics": out["single_stage_baseline"],
    }
    STATE_PATH.write_text(json.dumps(state, indent=2))

    print(json.dumps(payload, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())

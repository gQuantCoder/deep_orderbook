from __future__ import annotations

from pathlib import Path
import struct
import zlib

import numpy as np


BATCH_VARIANTS: dict[str, dict] = {
    "baseline_evt025_pw8_lr2e3_h96_e6": {
        "hidden_dim": 96,
        "epochs": 6,
        "lr": 2e-3,
        "weight_decay": 1e-4,
        "reg_loss": "huber",
        "event_loss_weight": 0.25,
        "pos_weight": 8.0,
        "trade_threshold": 0.04,
        "prediction_cap_quantile": 99.5,
        "active_reg_weight": 1.0,
        "hypothesis": "Reference capped holdout TCN baseline.",
    },
    "precision_evt005_pw2_thr010": {
        "hidden_dim": 96,
        "epochs": 6,
        "lr": 2e-3,
        "weight_decay": 1e-4,
        "reg_loss": "huber",
        "event_loss_weight": 0.05,
        "pos_weight": 2.0,
        "trade_threshold": 0.10,
        "prediction_cap_quantile": 99.5,
        "active_reg_weight": 1.0,
        "hypothesis": "Reduce event spam with weaker event pressure and higher trade threshold.",
    },
    "regonly_huber_thr010": {
        "hidden_dim": 96,
        "epochs": 6,
        "lr": 2e-3,
        "weight_decay": 1e-4,
        "reg_loss": "huber",
        "event_loss_weight": 0.0,
        "pos_weight": 1.0,
        "trade_threshold": 0.10,
        "prediction_cap_quantile": 99.5,
        "active_reg_weight": 1.0,
        "hypothesis": "Pure image-to-image regression may preserve geometry better.",
    },
    "small_h64_regonly": {
        "hidden_dim": 64,
        "epochs": 6,
        "lr": 2e-3,
        "weight_decay": 1e-4,
        "reg_loss": "huber",
        "event_loss_weight": 0.0,
        "pos_weight": 1.0,
        "trade_threshold": 0.10,
        "prediction_cap_quantile": 99.5,
        "active_reg_weight": 1.0,
        "hypothesis": "Smaller TCN may reduce overfit on strict holdout.",
    },
    "medium_h128_regonly_lr1e3": {
        "hidden_dim": 128,
        "epochs": 6,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "reg_loss": "huber",
        "event_loss_weight": 0.0,
        "pos_weight": 1.0,
        "trade_threshold": 0.10,
        "prediction_cap_quantile": 99.5,
        "active_reg_weight": 1.0,
        "hypothesis": "More width plus lower LR may keep detail without spiky extrapolation.",
    },
    "regonly_short_e4": {
        "hidden_dim": 96,
        "epochs": 4,
        "lr": 2e-3,
        "weight_decay": 1e-4,
        "reg_loss": "huber",
        "event_loss_weight": 0.0,
        "pos_weight": 1.0,
        "trade_threshold": 0.10,
        "prediction_cap_quantile": 99.5,
        "active_reg_weight": 1.0,
        "hypothesis": "Earlier stop may help strict holdout if we are overfitting fast.",
    },
    "regonly_long_e10_lr8e4": {
        "hidden_dim": 96,
        "epochs": 10,
        "lr": 8e-4,
        "weight_decay": 1e-4,
        "reg_loss": "huber",
        "event_loss_weight": 0.0,
        "pos_weight": 1.0,
        "trade_threshold": 0.10,
        "prediction_cap_quantile": 99.5,
        "active_reg_weight": 1.0,
        "hypothesis": "Longer, gentler optimization may recover cleaner maps.",
    },
    "mse_evt010_pw4": {
        "hidden_dim": 96,
        "epochs": 6,
        "lr": 1.5e-3,
        "weight_decay": 1e-4,
        "reg_loss": "mse",
        "event_loss_weight": 0.10,
        "pos_weight": 4.0,
        "trade_threshold": 0.08,
        "prediction_cap_quantile": 99.5,
        "active_reg_weight": 1.0,
        "hypothesis": "MSE may sharpen dense structure if event term is kept moderate.",
    },
    "l1_evt005_pw2": {
        "hidden_dim": 96,
        "epochs": 6,
        "lr": 1.5e-3,
        "weight_decay": 1e-4,
        "reg_loss": "l1",
        "event_loss_weight": 0.05,
        "pos_weight": 2.0,
        "trade_threshold": 0.10,
        "prediction_cap_quantile": 99.5,
        "active_reg_weight": 1.0,
        "hypothesis": "L1 can resist outlier explosions and keep maps less washed out.",
    },
    "regonly_activew3": {
        "hidden_dim": 96,
        "epochs": 6,
        "lr": 2e-3,
        "weight_decay": 1e-4,
        "reg_loss": "huber",
        "event_loss_weight": 0.0,
        "pos_weight": 1.0,
        "trade_threshold": 0.10,
        "prediction_cap_quantile": 99.5,
        "active_reg_weight": 3.0,
        "hypothesis": "Weight active target pixels more in regression to preserve sparse structures.",
    },
    "regonly_wd1e3": {
        "hidden_dim": 96,
        "epochs": 6,
        "lr": 2e-3,
        "weight_decay": 1e-3,
        "reg_loss": "huber",
        "event_loss_weight": 0.0,
        "pos_weight": 1.0,
        "trade_threshold": 0.10,
        "prediction_cap_quantile": 99.5,
        "active_reg_weight": 1.0,
        "hypothesis": "Stronger decay may improve generalization without killing map detail.",
    },
    "large_h160_evt002_lr8e4": {
        "hidden_dim": 160,
        "epochs": 8,
        "lr": 8e-4,
        "weight_decay": 2e-4,
        "reg_loss": "huber",
        "event_loss_weight": 0.02,
        "pos_weight": 2.0,
        "trade_threshold": 0.10,
        "prediction_cap_quantile": 99.5,
        "active_reg_weight": 1.0,
        "hypothesis": "Larger TCN with gentler optimization might capture richer motifs without event spam.",
    },
    "l1_evt005_pw2_thr006": {
        "hidden_dim": 96,
        "epochs": 6,
        "lr": 1.5e-3,
        "weight_decay": 1e-4,
        "reg_loss": "l1",
        "event_loss_weight": 0.05,
        "pos_weight": 2.0,
        "trade_threshold": 0.06,
        "prediction_cap_quantile": 99.5,
        "active_reg_weight": 1.0,
        "hypothesis": "Lower trigger threshold may convert conservative L1 maps into actual trades.",
    },
    "l1_evt005_pw2_thr014": {
        "hidden_dim": 96,
        "epochs": 6,
        "lr": 1.5e-3,
        "weight_decay": 1e-4,
        "reg_loss": "l1",
        "event_loss_weight": 0.05,
        "pos_weight": 2.0,
        "trade_threshold": 0.14,
        "prediction_cap_quantile": 99.5,
        "active_reg_weight": 1.0,
        "hypothesis": "Higher threshold may cut noisy event triggers inside violent windows.",
    },
    "l1_evt002_pw2_thr010": {
        "hidden_dim": 96,
        "epochs": 6,
        "lr": 1.5e-3,
        "weight_decay": 1e-4,
        "reg_loss": "l1",
        "event_loss_weight": 0.02,
        "pos_weight": 2.0,
        "trade_threshold": 0.10,
        "prediction_cap_quantile": 99.5,
        "active_reg_weight": 1.0,
        "hypothesis": "Slightly weaker event pressure may reduce overtriggering while preserving L1 geometry.",
    },
    "l1_evt010_pw2_thr010": {
        "hidden_dim": 96,
        "epochs": 6,
        "lr": 1.5e-3,
        "weight_decay": 1e-4,
        "reg_loss": "l1",
        "event_loss_weight": 0.10,
        "pos_weight": 2.0,
        "trade_threshold": 0.10,
        "prediction_cap_quantile": 99.5,
        "active_reg_weight": 1.0,
        "hypothesis": "More event pressure may sharpen timing on reaction windows if geometry survives.",
    },
    "l1_evt005_pw1_thr010": {
        "hidden_dim": 96,
        "epochs": 6,
        "lr": 1.5e-3,
        "weight_decay": 1e-4,
        "reg_loss": "l1",
        "event_loss_weight": 0.05,
        "pos_weight": 1.0,
        "trade_threshold": 0.10,
        "prediction_cap_quantile": 99.5,
        "active_reg_weight": 1.0,
        "hypothesis": "Lower positive-class weight may reduce false positives in event-filtered holdout.",
    },
    "l1_evt005_pw3_thr010": {
        "hidden_dim": 96,
        "epochs": 6,
        "lr": 1.5e-3,
        "weight_decay": 1e-4,
        "reg_loss": "l1",
        "event_loss_weight": 0.05,
        "pos_weight": 3.0,
        "trade_threshold": 0.10,
        "prediction_cap_quantile": 99.5,
        "active_reg_weight": 1.0,
        "hypothesis": "Higher positive-class weight may lift recall on violent windows without exploding RMSE.",
    },
    "l1_evt005_pw2_long_e10": {
        "hidden_dim": 96,
        "epochs": 10,
        "lr": 1.0e-3,
        "weight_decay": 1e-4,
        "reg_loss": "l1",
        "event_loss_weight": 0.05,
        "pos_weight": 2.0,
        "trade_threshold": 0.10,
        "prediction_cap_quantile": 99.5,
        "active_reg_weight": 1.0,
        "hypothesis": "Longer gentler L1 training may capture richer event geometry on filtered windows.",
    },
    "l1_evt005_pw2_short_e4": {
        "hidden_dim": 96,
        "epochs": 4,
        "lr": 1.5e-3,
        "weight_decay": 1e-4,
        "reg_loss": "l1",
        "event_loss_weight": 0.05,
        "pos_weight": 2.0,
        "trade_threshold": 0.10,
        "prediction_cap_quantile": 99.5,
        "active_reg_weight": 1.0,
        "hypothesis": "Earlier stopping may help if event-filtered windows overfit quickly.",
    },
    "l1_evt005_pw2_h64": {
        "hidden_dim": 64,
        "epochs": 6,
        "lr": 1.5e-3,
        "weight_decay": 1e-4,
        "reg_loss": "l1",
        "event_loss_weight": 0.05,
        "pos_weight": 2.0,
        "trade_threshold": 0.10,
        "prediction_cap_quantile": 99.5,
        "active_reg_weight": 1.0,
        "hypothesis": "Smaller L1 TCN may generalize better on concentrated event windows.",
    },
    "l1_evt005_pw2_h128": {
        "hidden_dim": 128,
        "epochs": 6,
        "lr": 1.2e-3,
        "weight_decay": 1e-4,
        "reg_loss": "l1",
        "event_loss_weight": 0.05,
        "pos_weight": 2.0,
        "trade_threshold": 0.10,
        "prediction_cap_quantile": 99.5,
        "active_reg_weight": 1.0,
        "hypothesis": "A slightly wider L1 TCN may preserve more vertical structure in reaction maps.",
    },
    "regonly_huber_thr006": {
        "hidden_dim": 96,
        "epochs": 6,
        "lr": 2e-3,
        "weight_decay": 1e-4,
        "reg_loss": "huber",
        "event_loss_weight": 0.0,
        "pos_weight": 1.0,
        "trade_threshold": 0.06,
        "prediction_cap_quantile": 99.5,
        "active_reg_weight": 1.0,
        "hypothesis": "Lower threshold may unlock profitable triggers from conservative reg-only maps.",
    },
    "regonly_huber_thr014": {
        "hidden_dim": 96,
        "epochs": 6,
        "lr": 2e-3,
        "weight_decay": 1e-4,
        "reg_loss": "huber",
        "event_loss_weight": 0.0,
        "pos_weight": 1.0,
        "trade_threshold": 0.14,
        "prediction_cap_quantile": 99.5,
        "active_reg_weight": 1.0,
        "hypothesis": "Higher threshold may turn reg-only outputs into cleaner high-confidence triggers.",
    },
    "regonly_huber_long_e12": {
        "hidden_dim": 96,
        "epochs": 12,
        "lr": 8e-4,
        "weight_decay": 1e-4,
        "reg_loss": "huber",
        "event_loss_weight": 0.0,
        "pos_weight": 1.0,
        "trade_threshold": 0.10,
        "prediction_cap_quantile": 99.5,
        "active_reg_weight": 1.0,
        "hypothesis": "Longer reg-only optimization may better fit recurring violent-window motifs.",
    },
    "regonly_huber_activew2": {
        "hidden_dim": 96,
        "epochs": 6,
        "lr": 2e-3,
        "weight_decay": 1e-4,
        "reg_loss": "huber",
        "event_loss_weight": 0.0,
        "pos_weight": 1.0,
        "trade_threshold": 0.10,
        "prediction_cap_quantile": 99.5,
        "active_reg_weight": 2.0,
        "hypothesis": "Moderate active-pixel weighting may improve sparse reaction geometry without destabilizing training.",
    },
    "regonly_wd5e4": {
        "hidden_dim": 96,
        "epochs": 6,
        "lr": 2e-3,
        "weight_decay": 5e-4,
        "reg_loss": "huber",
        "event_loss_weight": 0.0,
        "pos_weight": 1.0,
        "trade_threshold": 0.10,
        "prediction_cap_quantile": 99.5,
        "active_reg_weight": 1.0,
        "hypothesis": "Intermediate decay may regularize filtered-window training better than either extreme.",
    },
    "mse_evt005_pw2_thr010": {
        "hidden_dim": 96,
        "epochs": 6,
        "lr": 1.2e-3,
        "weight_decay": 1e-4,
        "reg_loss": "mse",
        "event_loss_weight": 0.05,
        "pos_weight": 2.0,
        "trade_threshold": 0.10,
        "prediction_cap_quantile": 99.5,
        "active_reg_weight": 1.0,
        "hypothesis": "MSE with lighter event pressure may sharpen reaction maps without the heavier baseline penalties.",
    },
}


def list_batch_variant_names() -> list[str]:
    return sorted(BATCH_VARIANTS.keys())


EVENT_FILTERED_SUITE_25: list[str] = [
    "baseline_evt025_pw8_lr2e3_h96_e6",
    "precision_evt005_pw2_thr010",
    "regonly_huber_thr010",
    "small_h64_regonly",
    "medium_h128_regonly_lr1e3",
    "regonly_short_e4",
    "regonly_long_e10_lr8e4",
    "mse_evt010_pw4",
    "l1_evt005_pw2",
    "regonly_activew3",
    "regonly_wd1e3",
    "large_h160_evt002_lr8e4",
    "l1_evt005_pw2_thr006",
    "l1_evt005_pw2_thr014",
    "l1_evt002_pw2_thr010",
    "l1_evt010_pw2_thr010",
    "l1_evt005_pw1_thr010",
    "l1_evt005_pw3_thr010",
    "l1_evt005_pw2_long_e10",
    "l1_evt005_pw2_short_e4",
    "l1_evt005_pw2_h64",
    "l1_evt005_pw2_h128",
    "regonly_huber_thr006",
    "regonly_huber_thr014",
    "regonly_huber_activew2",
]



def list_event_filtered_suite_25() -> list[str]:
    return list(EVENT_FILTERED_SUITE_25)



def get_batch_variant(name: str) -> dict:
    if name not in BATCH_VARIANTS:
        raise KeyError(f"Unknown batch variant: {name}")
    return dict(BATCH_VARIANTS[name])


def _read_chunks(raw: bytes) -> tuple[int, int, int, int, bytes]:
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Not a PNG file")
    idx = 8
    width = height = bit_depth = color_type = None
    idat = bytearray()
    while idx < len(raw):
        length = struct.unpack(">I", raw[idx : idx + 4])[0]
        idx += 4
        chunk_type = raw[idx : idx + 4]
        idx += 4
        chunk_data = raw[idx : idx + length]
        idx += length + 4  # skip crc too
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, compression, filt, interlace = struct.unpack(">IIBBBBB", chunk_data)
            if compression != 0 or filt != 0 or interlace != 0:
                raise ValueError("Unsupported PNG compression/filter/interlace")
        elif chunk_type == b"IDAT":
            idat.extend(chunk_data)
        elif chunk_type == b"IEND":
            break
    if None in (width, height, bit_depth, color_type):
        raise ValueError("Missing IHDR")
    return width, height, bit_depth, color_type, bytes(idat)


def _channels_for_color_type(color_type: int) -> int:
    return {0: 1, 2: 3, 6: 4}[color_type]


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _unfilter_png(raw: bytes, width: int, height: int, stride: int) -> np.ndarray:
    rows = []
    prev = bytearray(stride)
    idx = 0
    for _ in range(height):
        filter_type = raw[idx]
        idx += 1
        row = bytearray(raw[idx : idx + stride])
        idx += stride
        if filter_type == 1:
            for i in range(stride):
                left = row[i - 1] if i > 0 else 0
                row[i] = (row[i] + left) & 0xFF
        elif filter_type == 2:
            for i in range(stride):
                row[i] = (row[i] + prev[i]) & 0xFF
        elif filter_type == 3:
            for i in range(stride):
                left = row[i - 1] if i > 0 else 0
                up = prev[i]
                row[i] = (row[i] + ((left + up) // 2)) & 0xFF
        elif filter_type == 4:
            for i in range(stride):
                left = row[i - 1] if i > 0 else 0
                up = prev[i]
                up_left = prev[i - 1] if i > 0 else 0
                row[i] = (row[i] + _paeth(left, up, up_left)) & 0xFF
        elif filter_type != 0:
            raise ValueError(f"Unsupported PNG filter type: {filter_type}")
        rows.append(bytes(row))
        prev = row
    return np.frombuffer(b"".join(rows), dtype=np.uint8)


def _load_png_array(path: str | Path) -> np.ndarray:
    raw = Path(path).read_bytes()
    width, height, bit_depth, color_type, compressed = _read_chunks(raw)
    if bit_depth != 8:
        raise ValueError(f"Unsupported PNG bit depth: {bit_depth}")
    channels = _channels_for_color_type(color_type)
    stride = width * channels
    data = _unfilter_png(zlib.decompress(compressed), width, height, stride)
    arr = data.reshape(height, width, channels)
    return arr.astype(np.float32) / 255.0


def _to_gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        return np.clip(arr, 0.0, 1.0)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        rgb = arr[..., :3]
        return np.clip(0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2], 0.0, 1.0)
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def compute_png_quality_stats(path: str | Path) -> dict:
    gray = _to_gray(_load_png_array(path))
    gx = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0.0
    gy = np.abs(np.diff(gray, axis=0)).mean() if gray.shape[0] > 1 else 0.0
    stats = {
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "near_black_fraction": float(np.mean(gray < 0.03)),
        "near_white_fraction": float(np.mean(gray > 0.97)),
        "edge_mean_abs_diff": float((gx + gy) / 2.0),
    }
    stats.update(_judge_quality(stats))
    return stats


def _judge_quality(stats: dict) -> dict:
    if stats["near_black_fraction"] > 0.98:
        return {"usable": False, "reason": "mostly_black"}
    if stats["near_white_fraction"] > 0.98:
        return {"usable": False, "reason": "mostly_saturated"}
    if stats["gray_std"] < 0.035:
        return {"usable": False, "reason": "low_contrast"}
    if stats["edge_mean_abs_diff"] < 0.01:
        return {"usable": False, "reason": "too_smooth"}
    return {"usable": True, "reason": "ok"}


def score_holdout_route(result: dict) -> float:
    metrics = result.get("metrics", {})
    zero = result.get("zero_baseline", {})
    image_quality = result.get("image_quality", {})
    pnl = result.get("pnl", {}).get("fixed_slice", {}).get("prediction_final", 0.0)
    rmse = float(metrics.get("rmse", 1e9))
    zero_rmse = max(float(zero.get("rmse", 1e-9)), 1e-9)
    precision = float(metrics.get("precision", 0.0))
    f1 = float(metrics.get("f1", 0.0))
    image_bonus = 0.03 if image_quality.get("usable") else -0.08
    contrast_bonus = min(float(image_quality.get("gray_std", 0.0)), 0.25)
    pnl_bonus = max(min(float(pnl), 20.0), -20.0) / 400.0
    rmse_penalty = max(rmse / zero_rmse - 1.0, 0.0)
    return 3.0 * precision + 2.0 * f1 + image_bonus + 0.4 * contrast_bonus + pnl_bonus - 0.6 * rmse_penalty


def rank_variant_results(results: list[dict]) -> list[dict]:
    ranked = []
    for item in results:
        enriched = dict(item)
        enriched["route_score"] = score_holdout_route(item)
        ranked.append(enriched)
    ranked.sort(key=lambda x: (x["route_score"], x.get("metrics", {}).get("precision", 0.0), x.get("metrics", {}).get("f1", 0.0)), reverse=True)
    return ranked

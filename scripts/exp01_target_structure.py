import asyncio
import json
from pathlib import Path
import numpy as np

from deep_orderbook.config import ReplayConfig, ShaperConfig
from deep_orderbook.shaper import iter_shapes_t2l


async def main() -> None:
    replay_conf = ReplayConfig(
        markets=["ETH-USD"],
        data_dir=Path('/media/photoDS216/crypto/'),
        date_regexp='2025-02-18T11*',
        max_samples=-1,
        every='100ms',
    )
    shaper_config = ShaperConfig(
        only_full_arrays=True,
        view_bips=5,
        num_side_lvl=8,
        look_ahead=32,
        look_ahead_side_bips=5,
        look_ahead_side_width=4,
        rolling_window_size=256,
        window_stride=8,
        use_cache=False,
        save_cache=False,
    )

    max_windows = 80
    windows = 0

    target_vals = []
    input_vals = []
    per_window_peak = []
    per_window_nonzero = []

    async for books_array, level_prox, _pxar in iter_shapes_t2l(
        replay_config=replay_conf,
        shaper_config=shaper_config,
        live=False,
    ):
        windows += 1
        t = level_prox[:, :, 0]  # [time, levels]
        x = books_array[:, :, 0]  # [time, levels]

        target_vals.append(t.reshape(-1))
        input_vals.append(x.reshape(-1))
        per_window_peak.append(float(np.nanmax(t)))
        per_window_nonzero.append(float(np.mean(t > 1e-4)))

        if windows >= max_windows:
            break

    if windows == 0:
        raise RuntimeError("No windows produced. Check replay/date_regexp or shaper settings.")

    target_vals = np.concatenate(target_vals)
    input_vals = np.concatenate(input_vals)

    out = {
        "windows_analyzed": windows,
        "config": {
            "date_regexp": replay_conf.date_regexp,
            "every": replay_conf.every,
            "rolling_window_size": shaper_config.rolling_window_size,
            "window_stride": shaper_config.window_stride,
            "look_ahead": shaper_config.look_ahead,
            "num_side_lvl": shaper_config.num_side_lvl,
            "look_ahead_side_width": shaper_config.look_ahead_side_width,
        },
        "target": {
            "min": float(np.nanmin(target_vals)),
            "max": float(np.nanmax(target_vals)),
            "mean": float(np.nanmean(target_vals)),
            "std": float(np.nanstd(target_vals)),
            "q50": float(np.nanquantile(target_vals, 0.50)),
            "q90": float(np.nanquantile(target_vals, 0.90)),
            "q99": float(np.nanquantile(target_vals, 0.99)),
            "frac_gt_1e-5": float(np.mean(target_vals > 1e-5)),
            "frac_gt_1e-4": float(np.mean(target_vals > 1e-4)),
            "frac_gt_1e-3": float(np.mean(target_vals > 1e-3)),
            "window_peak_mean": float(np.mean(per_window_peak)),
            "window_peak_q90": float(np.quantile(per_window_peak, 0.90)),
            "window_nonzero_mean": float(np.mean(per_window_nonzero)),
        },
        "input_channel0": {
            "min": float(np.nanmin(input_vals)),
            "max": float(np.nanmax(input_vals)),
            "mean": float(np.nanmean(input_vals)),
            "std": float(np.nanstd(input_vals)),
            "q01": float(np.nanquantile(input_vals, 0.01)),
            "q50": float(np.nanquantile(input_vals, 0.50)),
            "q99": float(np.nanquantile(input_vals, 0.99)),
        },
    }

    out_path = Path('experiments/results/exp01_target_structure.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path.resolve()}")


if __name__ == '__main__':
    asyncio.run(main())

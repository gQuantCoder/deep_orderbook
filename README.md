# Deep OrderBook

![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Deep OrderBook** treats cryptocurrency limit order books as spatially correlated images and forecasts the full 2D structure of future price dynamics — not just a direction, not just a number, but a map.

---

## The Idea

Most quantitative approaches to order book data flatten it. They extract a handful of features — mid-price, best bid/ask, depth imbalance — and feed a scalar or vector into a model. This discards the most structurally rich information in the data.

A limit order book is inherently two-dimensional:
- **Time** runs along one axis
- **Price levels** run along the other

When you lay this out as a grid, with pixel intensity encoding liquidity at each price level and time step, you get something genuinely image-like. Orders cluster. Walls form. Price carves channels through the book. The patterns that matter to a trader — absorption events, level breakthroughs, order book imbalance building over time — are *spatially correlated features*, the exact class of pattern that vision models are engineered to detect.

**The insight: use the spatial inductive bias of vision models, applied to a physically meaningful 2D market representation.**

---

## Why This Is Not Naive

The obvious objection: "you're just plotting data and running a CNN." That's not what's happening here.

### 1. The encoding is physically motivated

Price levels are binned relative to the current mid-price, not absolute. The x-axis is time (causal), the y-axis is price distance from mid in basis points. Each cell holds a transformed liquidity signal. The geometry is preserved: nearby price levels really are nearby in the image, so a convolution kernel sliding over price axis *is* detecting structure across correlated price levels. That's not an accident — it's the whole point.

### 2. The target is also a 2D map

The model doesn't predict a scalar or a class. It predicts a **time-to-level proximity map**: for each future timestep and each price level bin, how close does price come to that level? This is a full 2D forecast of what the future price trajectory looks like in (time, price) space.

```
Input:  [T × price_levels × channels]   →   past order book + trades
Output: [T × price_levels]              →   future price proximity map
```

This turns the problem into proper image-to-image regression — the same class of problem as depth estimation, optical flow, and video prediction. All the architectural advances from those fields are directly applicable.

### 3. Multiple color channels encode different market signals

Like RGB layers in a photograph, the input tensor uses multiple channels per (time, price) cell:

- **Channel 0**: Order book depth/pressure (transformed)
- **Channel 1**: Trade flow and imbalance
- **Channel 2**: Derived microstructure signal

The model sees the book *and* the tape simultaneously, in the same spatial coordinate system. A large order sitting at a level and trades piling up below it appear as spatially co-located features. The model can learn that combination.

### 4. The architecture is strictly causal

No future leakage. The time axis runs left (past) to right (now). The model only sees the left side; it predicts what the right side will look like. This is enforced architecturally via causal convolutions, not just by convention in data splits.

### 5. Event-window conditioning — focus on what matters

Markets are quiet most of the time. Training on all windows equally teaches the model to predict "nothing happens." Instead, windows are ranked by observable present-time eventfulness (abs return, range, realized vol, book std, book impulse) and training concentrates on the most active market regimes. This is done without looking at future targets, so there's no leakage — and it dramatically improves event-detection quality on holdout data.

---

## Architecture

```
┌──────────────────────┐
│  Coinbase WebSocket  │   (live) or  parquet replay
│  or Historical Data  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   ArrayShaper        │   bins price levels, computes 3-channel input tensor
│                      │   + time-to-level proximity target tensor
└──────────┬───────────┘
           │  [T × 2*K × 3] input,  [T × 2*W] target
           ▼
┌──────────────────────┐
│   TCN / AttentionTCN │   causal dilated convolutions over time axis
│   / Transformer      │   with optional per-price-level attention
└──────────┬───────────┘
           │  predicted proximity map  [T × 2*W]
           ▼
┌──────────────────────┐
│   Strategy Layer     │   extract up/down max, side margin, persistence
│                      │   → directional trade signals → simulated PnL
└──────────────────────┘
```

The shaper produces tensors shaped `[256, 16, 3]` (256 timesteps × 16 price bins × 3 channels) as input and `[256, 8]` as target. The target encodes, for each of the next 64 timesteps and each of 4 price bins on each side of the book, how close price comes to that level.

---

## Experimental Results

This is an active research project. The experiment log tracks ~100+ runs across:

- 3 model families (TCN, AttentionTCN, Transformer)
- 25+ loss configurations (L1, Huber, MSE + sparse event terms)
- Multiple symbols (BTC-USD, ETH-USD)
- Walk-forward holdout splits across hourly parquet files
- Automated image QC gating every run (contrast, saturation, edge density)

**Key finding so far**: Event-window conditioning consistently lifts holdout precision/F1 vs broad-slice training without causing RMSE collapse. The best runs achieve near-zero-baseline RMSE while maintaining precision >0.29 on active market windows. The bottleneck has shifted from *map reconstruction* to *signal extraction and execution conversion* — which is the right problem to have.

---

## Demo

[![Deep OrderBook Demo](https://img.youtube.com/vi/TUogAa2Y1sU/0.jpg)](https://www.youtube.com/shorts/TUogAa2Y1sU)

Example of the 2D input representation and predicted proximity map:

![Order Book Image Representation](https://raw.githubusercontent.com/gQuantCoder/deep_orderbook/master/images/01.png?raw=true "Order book as a spatially correlated image")

---

## Connection to the Frontier

The image-to-image framing of order book forecasting is gaining ground in academic research. Recent work (2025–2026) includes:

- **Diffusion models** applied to LOB images, treating future states as an inpainting problem
- **LOB-Bench** standardizing 2D representation evaluation
- **Multimodal fusion** of image-based predictions with attention-refined recurrent refinements

What makes this project distinct is the explicit multi-channel physical encoding (not just flattened depth), the 2D proximity-map target (not a scalar midprice direction), and the end-to-end pipeline from live WebSocket data through causal visual model to trade signal.

---

## Structure

```
deep_orderbook/
├── feeds/              WebSocket + replay feed normalization
├── consumers/          Live recorder (hourly parquet rotation)
├── learn/              TCN, AttentionTCN, Transformer + training loop
├── shaper.py           Core image construction (time × price × channel)
├── strategy.py         Rule-based signal extraction from predicted maps
├── event_selection.py  Leakage-safe eventfulness ranking for window selection
├── strategy_search.py  Vectorized directional backtester
├── trigger_search.py   Train-calibrated strategy grid generation
├── btc_search_lab.py   Image QC + holdout route scoring/ranking
└── experiment_tracking.py  SQLite registry + PNG artifact logging
experiments/
├── hourly_journal.md       Running experiment continuity log
├── image_prediction_lab_log.md  Full scientific experiment log (EXP-01 → EXP-22+)
└── scientific_experiment_search.md  Experiment philosophy and staged search protocol
scripts/
└── exp01..exp22*.py    Numbered self-contained experiments
```

See [architecture.md](architecture.md) for the full dependency map.

---

## Installation

```bash
git clone https://github.com/gQuantCoder/deep_orderbook.git
cd deep_orderbook
pip install -r requirements.txt
pip install -e .
```

For live data, create `credentials/coinbase.txt`:
```
api_key="organizations/xxx.../apiKeys/xxx..."
api_secret="-----BEGIN EC PRIVATE KEY-----\n...\n-----END EC PRIVATE KEY-----\n"
```

---

## Usage

**Record live data:**
```bash
python -m deep_orderbook.consumers.recorder
```

**Replay and visualize:**
```python
# replay.ipynb — replay historical parquet + live visualization
```

**Train a model:**
```python
# learn-100ms.ipynb — full training loop with live loss plot
```

**Run an experiment:**
```bash
PYTHONPATH=. python scripts/exp10_scientist_once.py
```

---

## License

MIT. See LICENSE.

# Deep OrderBook

![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)

**Deep OrderBook** is an advanced cryptocurrency order book analysis toolkit that transforms order book data into temporally and spatially local-correlated representations for quantitative analysis and deep learning applications.

## Overview

While conventional technical analysis often relies on price-derived indicators like moving averages, order books contain rich organic information about market microstructure and supply/demand dynamics. Deep OrderBook captures this information by:

1. Processing real-time and historical order book data
2. Converting multi-dimensional order book states into feature-rich representations
3. Enabling visualization and machine learning on these representations

## Example Output

![books](https://raw.githubusercontent.com/gQuantCoder/deep_orderbook/master/images/01.png?raw=true "Orderbooks and alpha")

## Demo Video

Check out our short demo video:

[![Deep OrderBook Demo](https://img.youtube.com/vi/TUogAa2Y1sU/0.jpg)](https://www.youtube.com/shorts/TUogAa2Y1sU)

## Features

- **Live Data Collection**: Connect to cryptocurrency exchanges (Coinbase, etc.) for real-time order book data
- **Historical Replay**: Replay and analyze historical order book data with precise timing
- **Visualization Tools**: Rich visualizations of order book dynamics and patterns
- **Machine Learning Integration**: Pre-process order book data for ML applications
- **Asyncio-based Architecture**: Non-blocking I/O for efficient data processing
- **Type-Safe Implementation**: Fully type-annotated codebase with Pydantic data validation

## Installation

### Prerequisites

- Python 3.12 or higher
- API credentials (for live data)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/gQuantCoder/deep_orderbook.git
   cd deep_orderbook
   ```

2. Set up API credentials:
   Create a file `credentials/coinbase.txt` with your API details:
   ```
   api_key="organizations/xxxxxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx/apiKeys/xxxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxxx"
   api_secret="-----BEGIN EC PRIVATE KEY-----\xxxxxxxxxxxxxxxxx...xxxxxxxxxxxxxxxxxxx\n-----END EC PRIVATE KEY-----\n"
   ```

3. Install with uv:
   ```bash
   uv sync
   ```

4. Run commands with uv-managed environment:
   ```bash
   uv run deepbook --version
   uv run deepbook-record
   ```

## Usage

### Recording Live Data

To capture live order book data from exchanges, use the recorder module:

```bash
# Start the recorder with default settings
python -m deep_orderbook.consumers.recorder

# Alternatively, you can run it directly
python deep_orderbook/consumers/recorder.py
```

The recorder will:
1. Connect to the configured exchanges
2. Save order book updates and trades to parquet files
3. Store files in the configured data directory 
4. Automatically rotate files at midnight

### Data Analysis and Visualization

The project includes several Jupyter notebooks for different analysis scenarios:

- **Live Analysis**: `live.ipynb` - Connect to exchange and visualize live order book
- **Historical Replay**: `replay.ipynb` - Replay and analyze historical order book data
- **Machine Learning**: `learn.ipynb` - Examples of applying ML to order book features

### Example Output

![Order Book Visualization](https://raw.githubusercontent.com/gQuantCoder/deep_orderbook/master/images/01.png?raw=true "Orderbooks and alpha")

## Architecture

Deep OrderBook is built on an event-driven architecture:

```
┌─────────────┐     ┌───────────┐     ┌─────────┐     ┌──────────────┐
│ Data Sources│────►│ Processors│────►│ Shapers │────►│ Visualizers/ │
│ (Exchanges) │     │           │     │         │     │ ML Models    │
└─────────────┘     └───────────┘     └─────────┘     └──────────────┘
```

- **Data Sources**: Exchange APIs, historical data files
- **Processors**: Convert raw data to standard format
- **Shapers**: Transform order book snapshots into feature matrices
- **Consumers**: Visualization tools, ML models, trading signals

## Development

### Testing

Run the test suite:

```bash
pytest
```

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors and the quant trading community

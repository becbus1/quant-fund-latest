# Quant Trading System

A machine-driven quantitative trading system composed of two independent systems:

1. **Trader** - Always-on paper trading runtime
2. **Edge Factory** - Offline ML edge discovery factory

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        QUANT SYSTEM                              │
├─────────────────────────────┬───────────────────────────────────┤
│         TRADER              │         EDGE FACTORY              │
│     (Always Running)        │          (Offline)                │
├─────────────────────────────┼───────────────────────────────────┤
│ • Live data ingestion       │ • Historical data analysis        │
│ • Paper trade execution     │ • Anomaly detection               │
│ • Risk management           │ • Meta-labeling                   │
│ • Position tracking         │ • Walk-forward backtesting        │
│                             │ • Strategy validation             │
├─────────────────────────────┴───────────────────────────────────┤
│                     registry/strategies.json                     │
│              (Only communication between systems)                │
└─────────────────────────────────────────────────────────────────┘
```

## Safety Rules

1. **PAPER TRADING ONLY** - No real money, no private keys, no live orders
2. **HARD SEPARATION** - Trader never trains ML, ML never executes trades
3. **NO PRICE PREDICTION** - ML detects anomalies, not future prices

## Quick Start

### Trader System

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export MODE=paper
export SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT
export DATABASE_URL=sqlite:///./trader.db

# Run the trader
uvicorn trader.app.main:app --host 0.0.0.0 --port 8080
```

### Edge Factory

```bash
# Run edge discovery
python -m edge_factory.train --mode train --days 7

# Review existing strategies
python -m edge_factory.train --mode review
```

## Configuration

All configuration via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| MODE | paper | Trading mode (must be "paper") |
| SYMBOLS | BTCUSDT,ETHUSDT,SOLUSDT | Symbols to trade |
| WINDOW_SIZE | 120 | Feature calculation window (seconds) |
| ENTRY_Z | 2.0 | Entry z-score threshold |
| TP_BPS | 8 | Take profit in basis points |
| SL_BPS | 5 | Stop loss in basis points |
| TIME_STOP_SEC | 60 | Time stop in seconds |
| NOTIONAL_USDT | 25 | Position size in USDT |
| FEES_BPS | 4 | Trading fees in basis points |
| SLIPPAGE_BPS | 2 | Slippage estimate in basis points |
| MAX_DAILY_LOSS_USDT | 100 | Daily loss limit |
| DATABASE_URL | sqlite:///./trader.db | Database connection |
| KILLSWITCH_TOKEN | SECRET | Kill switch auth token |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /health | GET | Health check |
| /metrics | GET | Trading metrics |
| /positions | GET | Open positions |
| /killswitch | GET | Activate kill switch (requires token) |
| /liquidity | GET | Liquidity status |

## Liquidity Gating

Symbols are automatically disabled if they fail liquidity checks:

- MIN_24H_VOLUME_USD ≥ 500M
- MAX_SPREAD_BPS ≤ 5
- MIN_DEPTH_10BPS_USD ≥ 1M

## Strategy Lifecycle

1. **Discovery** - ML detects anomalous market states
2. **Labeling** - Meta-labeling determines if anomalies are profitable
3. **Backtesting** - Walk-forward validation with fees/slippage
4. **Validation** - Check Sharpe, win rate, drawdown thresholds
5. **Registration** - Approved strategies written to registry
6. **Execution** - Trader reads registry and executes signals
7. **Review** - Periodic comparison of expected vs realized performance
8. **Expiration** - Strategies expire after lifetime (default 7 days)

## Deployment

### Docker

```bash
cd trader
docker build -t paper-trader .
docker run -p 8080:8080 -e MODE=paper paper-trader
```

### Railway

The trader is configured for Railway deployment via `railway.toml`.

## Project Structure

```
quant-system/
├─ trader/                    # Paper trading runtime
│  ├─ app/
│  │  ├─ api/                # FastAPI routes
│  │  ├─ common/             # Config, DB, models
│  │  ├─ executor/           # Paper execution
│  │  ├─ ingest/             # Bybit WebSocket
│  │  ├─ risk/               # Risk management
│  │  └─ main.py             # Application entry
│  ├─ Dockerfile
│  └─ railway.toml
├─ edge_factory/              # ML edge discovery
│  ├─ data/                  # Data fetching
│  ├─ features/              # Feature engineering
│  ├─ models/                # Anomaly detection, meta-labeling
│  ├─ backtests/             # Walk-forward backtesting
│  ├─ selection/             # Strategy evaluation
│  └─ train.py               # Training pipeline
├─ shared/                    # Shared definitions
│  ├─ feature_defs.py        # Feature specifications
│  └─ schemas.py             # Data schemas
├─ registry/
│  └─ strategies.json        # Strategy registry
├─ requirements.txt
└─ README.md
```

## License

Proprietary - All rights reserved.

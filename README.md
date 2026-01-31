# 👁️ Argus

> *Named after Argus Panoptes, the all-seeing giant of Greek mythology who had 100 eyes and never slept*

**Argus** is a 24/7 crypto market monitoring system that detects trading opportunities across 6 different strategy types. It runs in observation mode to gather real market data, then the best opportunity is selected for automation.

## 🎯 Strategy Types

### Bot-Tradeable (Automated Candidates)
1. **Funding Rate Mean Reversion** - Perpetual funding rates spike, then revert to mean
2. **Spot-Perp Basis Arbitrage** - Price gaps between spot and perpetual contracts
3. **Cross-Exchange Latency Arb** - Same asset at different prices on different exchanges
4. **Post-Liquidation Snapback** - Price spikes during liquidation cascades, then bounces

### Human-Tradeable (Manual)
5. **BTC Options IV Spike** - Implied volatility spikes >80% during panic (sell premium)
6. **Volatility Regime Shifts** - Sudden volatility expansion/compression events

## 📊 Architecture

```
Market Data Sources (WebSocket/REST)
    ↓
Data Normalization Layer
    ↓
6 Opportunity Detectors (Independent Modules)
    ↓
SQLite Database (Logging Everything)
    ↓
Alert System (Telegram) + Analysis Engine
```

## 🛠️ Setup

### Prerequisites
- Python 3.10+
- API keys for: Bybit, Binance, OKX, Deribit, Coinglass
- Telegram bot token

### Installation

```powershell
# Navigate to project
cd C:\Users\Oliver\Desktop\Desktop\Projects\argus

# Create virtual environment
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Copy secrets template
copy config\secrets.example.yaml config\secrets.yaml

# Edit secrets.yaml with your API keys

# Initialize database
python scripts\init_database.py

# Run Argus
python run.py
```

## 📁 Project Structure

```
argus/
├── config/
│   ├── config.yaml          # Main configuration
│   ├── secrets.yaml         # API keys (gitignored)
│   └── thresholds.yaml      # Detection thresholds
├── src/
│   ├── core/                # Database, logging, utilities
│   ├── connectors/          # Exchange WebSocket/REST clients
│   ├── detectors/           # 6 opportunity detectors
│   ├── alerts/              # Telegram notifications
│   └── analysis/            # Performance tracking
├── data/
│   ├── argus.db            # SQLite database
│   └── logs/               # Daily log files
├── tests/                   # Unit tests
├── notebooks/               # Jupyter analysis notebooks
└── scripts/                 # Utility scripts
```

## 📈 Timeline

| Week | Phase |
|------|-------|
| 1-2 | Build and run detector (observation only) |
| 3 | Analyze data and select best strategy |
| 4-5 | Build trading bot for winning strategy |
| 6+ | Deploy and monitor |

## ⚠️ Important Rules

1. **90-Day Rule**: After bot deployment, NO parameter changes for 90 days
2. **Circuit Breakers**: Auto-pause on 5% daily loss or 5 consecutive losses
3. **Observation First**: Always observe before trading

## 📞 Alert Tiers

| Tier | Type | Example |
|------|------|---------|
| 🚨 1 | Immediate | Options IV >80%, Liquidations >$5M |
| 📊 2 | FYI | Funding extremes, Basis arb |
| 📝 3 | Background | Minor events (logged only) |

## 📜 License

Private project - not for redistribution.

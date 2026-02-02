# 👁️ Argus

> *Named after Argus Panoptes, the all-seeing giant of Greek mythology who had 100 eyes and never slept*

**Argus** is a 24/7 crypto market monitoring system that powers manual trading recommendations. It runs in observation mode to gather real market data, then paper trades a wide range of parameter combinations to identify the best-performing strategy to follow manually.

## 🎯 Strategy Types (Manual Recommendations)

1. **BTC Options IV Spike** - Implied volatility spikes during panic (sell premium)
2. **Volatility Regime Shifts** - Sudden volatility expansion/compression events
3. **IBIT Options Put Spreads** - BTC IV + ETF drawdown triggers for Robinhood trades
4. **BITO Options Put Spreads** - Same framework, more opportunity coverage

## 📊 Architecture

```
Market Data Sources (WebSocket/REST)
    ↓
Data Normalization Layer
    ↓
Manual Opportunity Detectors (Independent Modules)
    ↓
SQLite Database (Logging Everything)
    ↓
Alert System (Telegram) + Paper Trader Analysis Engine
```

## 🛠️ Setup

### Prerequisites
- Python 3.10+
- API keys for: Bybit, Deribit (optional for IV data)
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
python main.py
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
│   ├── detectors/           # Manual opportunity detectors
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
| 3 | Analyze data and select best paper trader |
| 4-5 | Follow the top paper trader manually (no automation) |
| 6+ | Continue monitoring and performance reviews |

## ⚠️ Important Rules

1. **90-Day Rule**: After adopting a strategy, NO parameter changes for 90 days
2. **Circuit Breakers**: Auto-pause on 5% daily loss or 5 consecutive losses
3. **Observation First**: Always observe before trading

## 📞 Alert Tiers

| Tier | Type | Example |
|------|------|---------|
| 🚨 1 | Immediate | IBIT/BITO put spread signal |
| 📊 2 | FYI | IV spike confirmations |
| 📝 3 | Background | Minor events (logged only) |

## 📜 License

Private project - not for redistribution.

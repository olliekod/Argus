---
description: Complete setup and implementation workflow for Argus - the crypto market monitoring system
---

# Argus Market Monitor - Implementation Workflow

## Project Overview
**Argus** (named after the all-seeing giant of Greek mythology) is a unified crypto market monitoring system that watches markets 24/7 and detects 6 types of trading opportunities.

---

## Phase 1: Project Setup (Day 1)

### Step 1.1: Verify Python Installation
Open PowerShell and run:
```powershell
python --version
```
Expected: Python 3.10 or higher. If not installed, download from https://python.org

### Step 1.2: Create Virtual Environment
```powershell
cd C:\Users\Oliver\Desktop\Desktop\Projects\argus
python -m venv venv
```

### Step 1.3: Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```
Note: If you get an execution policy error, run PowerShell as Admin and execute:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 1.4: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 1.5: Configure API Keys
1. Copy `config\secrets.example.yaml` to `config\secrets.yaml`
2. Fill in your API keys (see API Keys Guide below)

### Step 1.6: Initialize Database
```powershell
python scripts\init_database.py
```

### Step 1.7: Test Telegram Bot
```powershell
python scripts\test_telegram.py
```

---

## Phase 2: Build Connectors (Days 2-3)

### Step 2.1: Test Bybit Connection
```powershell
python -m pytest tests\test_bybit.py -v
```

### Step 2.2: Test Binance Connection
```powershell
python -m pytest tests\test_binance.py -v
```

### Step 2.3: Test OKX Connection
```powershell
python -m pytest tests\test_okx.py -v
```

### Step 2.4: Test Deribit Connection (Testnet)
```powershell
python -m pytest tests\test_deribit.py -v
```

### Step 2.5: Test Coinglass Connection
```powershell
python -m pytest tests\test_coinglass.py -v
```

---

## Phase 3: Build Detectors (Days 4-5)

### Step 3.1: Test Funding Rate Detector
```powershell
python -m pytest tests\test_funding_detector.py -v
```

### Step 3.2: Test Basis Detector
```powershell
python -m pytest tests\test_basis_detector.py -v
```

### Step 3.3: Test All Detectors
```powershell
python -m pytest tests\test_detectors.py -v
```

---

## Phase 4: Integration (Days 6-7)

### Step 4.1: Run Full System Test
```powershell
python scripts\integration_test.py
```

### Step 4.2: Start Argus (Observation Mode)
```powershell
python run.py
```
Keep this running 24/7. Use `Ctrl+C` to gracefully stop.

---

## Phase 5: Week 2 - Options & Analysis

### Step 5.1: Verify Options Detector
```powershell
python -m pytest tests\test_options_detector.py -v
```

### Step 5.2: Check Database Health
```powershell
python scripts\db_health_check.py
```

### Step 5.3: Generate Weekly Report
```powershell
python scripts\generate_report.py --days 7
```

---

## Phase 6: Week 3 - Analysis

### Step 6.1: Export Data for Analysis
```powershell
python scripts\export_analysis_data.py
```

### Step 6.2: Open Jupyter Notebook
```powershell
jupyter notebook notebooks\data_exploration.ipynb
```

### ⚠️ HANDOFF POINT: Strategy Selection
After analyzing 2 weeks of data, consult with Sonnet 4.5 to:
- Review detection frequency and edge statistics
- Select the winning strategy to automate
- Discuss risk parameters for the trading bot

---

## API Keys Required

See the secrets.yaml file for where to place each key.

### 1. Bybit (https://www.bybit.com)
- Login → API Management → Create New Key
- Permissions needed: Read-only (no trading permissions for observation)
- Copy: API Key and API Secret

### 2. Binance (https://www.binance.com)
- Login → API Management → Create API
- Permissions: Enable Reading only
- Copy: API Key and Secret Key

### 3. OKX (https://www.okx.com)
- Login → API → Create API Key
- Permissions: Read-only
- Copy: API Key, Secret Key, and Passphrase

### 4. Deribit (https://www.deribit.com)
- Use TESTNET first: https://test.deribit.com
- Account → API → Add New Key
- Permissions: Read-only
- Copy: Client ID and Client Secret

### 5. Coinglass (https://www.coinglass.com)
- Create free account
- Go to API section
- Copy: API Key

### 6. Telegram Bot
- Open Telegram, search for @BotFather
- Send: /newbot
- Follow prompts to create bot
- Copy: Bot Token (looks like: 123456789:ABCdefGHI...)
- Create a channel/group, add your bot
- Get Chat ID: Send a message to your bot, then visit:
  https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
- Copy: Chat ID (negative number for groups)

---

## Troubleshooting

### WebSocket Disconnections
Check logs at: `data\logs\YYYYMMDD.log`
Argus auto-reconnects, but if persistent, restart with `python run.py`

### Database Locked Errors
Stop Argus, then run:
```powershell
python scripts\db_repair.py
```

### Telegram Not Sending
Verify bot token and chat ID in secrets.yaml
Test with: `python scripts\test_telegram.py`

---

## Daily Operations

**Morning Check (5 min):**
1. Check Telegram for overnight alerts
2. Verify Argus is running: Check console or `data\logs\` for recent entries

**Evening Check (5 min):**
1. Review daily summary (sent at 8 PM UTC)
2. Check hypothetical P&L

**Weekly Review:**
```powershell
python scripts\generate_report.py --days 7
```

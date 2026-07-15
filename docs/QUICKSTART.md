# Quick start

## 1. Install

Linux/macOS:

```bash
./install.sh --dev
```

Windows PowerShell:

```powershell
.\install.ps1 --dev
```

## 2. Add data

Place the supported BTCUSDT CSV files under `data/` or `data/raw/`.

## 3. Prepare Parquet

```bash
./prepare-data.sh
```

Windows:

```powershell
.\prepare-data.ps1
```

## 4. Start the dashboard

```bash
./run.sh
```

Windows:

```powershell
.\run.ps1
```

Open `http://127.0.0.1:8000`.

## 5. Run a backtest

1. Open **Backtest lab**.
2. Confirm the dataset is prepared.
3. Choose a strategy and timeframe.
4. Set fees, slippage, spread, leverage, and shorting.
5. Run the authoritative backtest.

## 6. Start Kraken paper trading

1. Open **Kraken paper**.
2. Choose a strategy and supported live timeframe.
3. Set bootstrap bars and paper execution assumptions.
4. Press **Start paper**.

Meteor Quant consumes public Kraken data and never sends a live order.

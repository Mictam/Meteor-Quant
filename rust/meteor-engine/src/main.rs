use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use arrow_array::{Array, Float64Array, Int64Array, RecordBatch};
use clap::{Args, Parser, Subcommand};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(name = "aegis-engine", version, about = "Aegis Quant streaming backtest engine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Backtest(BacktestArgs),
}

#[derive(Args, Debug, Clone)]
struct BacktestArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 1)]
    timeframe_seconds: u64,
    #[arg(long, default_value_t = 10_000.0)]
    initial_equity: f64,
    #[arg(long, default_value_t = 10.0)]
    fee_bps: f64,
    #[arg(long, default_value_t = 1.0)]
    slippage_bps: f64,
    #[arg(long, default_value_t = 1.0)]
    spread_bps: f64,
    #[arg(long, default_value_t = 10.0)]
    minimum_order_notional: f64,
    #[arg(long, default_value_t = 1.0)]
    max_leverage: f64,
    #[arg(long, default_value_t = false)]
    allow_short: bool,
    #[arg(long, default_value_t = 10_000)]
    max_equity_points: usize,
    #[arg(long, default_value_t = 100_000)]
    max_fills_returned: usize,
}

#[derive(Debug, Serialize)]
struct Fill {
    timestamp: i64,
    side: &'static str,
    quantity: f64,
    price: f64,
    fee: f64,
    position_after: f64,
    cash_after: f64,
    reason: &'static str,
}

#[derive(Debug, Serialize)]
struct EquityPoint {
    timestamp: i64,
    equity: f64,
    drawdown_pct: f64,
}

#[derive(Debug, Serialize)]
struct Metrics {
    initial_equity: f64,
    final_equity: f64,
    total_return_pct: f64,
    max_drawdown_pct: f64,
    sharpe_ratio: Option<f64>,
    annualized_volatility_pct: Option<f64>,
    buy_and_hold_return_pct: f64,
    fill_count: u64,
    fills_returned: usize,
    fees_paid: f64,
    bar_count: u64,
    invested_bar_fraction: f64,
}

#[derive(Debug, Serialize)]
struct BacktestResult {
    engine: &'static str,
    engine_version: &'static str,
    metrics: Metrics,
    fills: Vec<Fill>,
    equity: Vec<EquityPoint>,
    first_timestamp: i64,
    last_timestamp: i64,
}

struct EngineState {
    cash: f64,
    quantity: f64,
    fees_paid: f64,
    pending_target: Option<f64>,
    last_signal_target: Option<f64>,
    peak_equity: f64,
    max_drawdown: f64,
    first_close: Option<f64>,
    last_close: f64,
    first_timestamp: i64,
    last_timestamp: i64,
    previous_equity: Option<f64>,
    return_count: u64,
    return_mean: f64,
    return_m2: f64,
    bar_count: u64,
    invested_count: u64,
    fill_count: u64,
    fills: Vec<Fill>,
    equity: Vec<EquityPoint>,
    sample_stride: u64,
}

impl EngineState {
    fn new(args: &BacktestArgs, total_rows: u64) -> Self {
        let point_limit = args.max_equity_points.max(100) as u64;
        Self {
            cash: args.initial_equity,
            quantity: 0.0,
            fees_paid: 0.0,
            pending_target: None,
            last_signal_target: None,
            peak_equity: args.initial_equity,
            max_drawdown: 0.0,
            first_close: None,
            last_close: 0.0,
            first_timestamp: 0,
            last_timestamp: 0,
            previous_equity: None,
            return_count: 0,
            return_mean: 0.0,
            return_m2: 0.0,
            bar_count: 0,
            invested_count: 0,
            fill_count: 0,
            fills: Vec::with_capacity(args.max_fills_returned.min(100_000)),
            equity: Vec::with_capacity(args.max_equity_points.max(100)),
            sample_stride: total_rows.div_ceil(point_limit).max(1),
        }
    }

    fn process_row(
        &mut self,
        timestamp: i64,
        open: f64,
        close: f64,
        target: Option<f64>,
        args: &BacktestArgs,
        is_last: bool,
    ) -> Result<()> {
        if !open.is_finite() || !close.is_finite() || open <= 0.0 || close <= 0.0 {
            bail!("invalid price at timestamp {timestamp}");
        }
        if self.bar_count == 0 {
            self.first_close = Some(close);
            self.first_timestamp = timestamp;
        }
        if let Some(pending) = self.pending_target.take() {
            self.rebalance(pending, open, timestamp, args);
        }
        if let Some(value) = target.filter(|value| value.is_finite()) {
            let changed = self
                .last_signal_target
                .map(|previous| (value - previous).abs() > 1e-12)
                .unwrap_or(true);
            if changed {
                self.pending_target = Some(value);
                self.last_signal_target = Some(value);
            }
        }

        let equity = self.cash + self.quantity * close;
        if self.quantity.abs() > 1e-15 {
            self.invested_count += 1;
        }
        self.peak_equity = self.peak_equity.max(equity);
        let drawdown = if self.peak_equity > 0.0 {
            (equity / self.peak_equity - 1.0) * 100.0
        } else {
            -100.0
        };
        self.max_drawdown = self.max_drawdown.min(drawdown);
        if let Some(previous) = self.previous_equity.filter(|previous| *previous > 0.0) {
            let value = equity / previous - 1.0;
            self.return_count += 1;
            let delta = value - self.return_mean;
            self.return_mean += delta / self.return_count as f64;
            self.return_m2 += delta * (value - self.return_mean);
        }
        self.previous_equity = Some(equity);
        if self.bar_count % self.sample_stride == 0 || is_last {
            self.equity.push(EquityPoint {
                timestamp,
                equity,
                drawdown_pct: drawdown,
            });
        }
        self.last_close = close;
        self.last_timestamp = timestamp;
        self.bar_count += 1;
        Ok(())
    }

    fn rebalance(&mut self, target: f64, open: f64, timestamp: i64, args: &BacktestArgs) {
        let lower = if args.allow_short { -args.max_leverage } else { 0.0 };
        let target = target.clamp(lower, args.max_leverage);
        let equity = self.cash + self.quantity * open;
        if equity <= 0.0 {
            return;
        }
        let desired_quantity = target * equity / open;
        let mut delta = desired_quantity - self.quantity;
        if (delta * open).abs() < args.minimum_order_notional {
            return;
        }
        let buy = delta > 0.0;
        let friction = (args.spread_bps / 2.0 + args.slippage_bps) / 10_000.0;
        let execution_price = open * if buy { 1.0 + friction } else { 1.0 - friction };
        let fee_rate = args.fee_bps / 10_000.0;
        if buy && !args.allow_short && args.max_leverage <= 1.0 {
            let affordable = self.cash.max(0.0) / (execution_price * (1.0 + fee_rate));
            delta = delta.min(affordable);
            if (delta * open).abs() < args.minimum_order_notional {
                return;
            }
        }
        let fee = (delta * execution_price).abs() * fee_rate;
        self.cash -= delta * execution_price + fee;
        self.quantity += delta;
        self.fees_paid += fee;
        self.fill_count += 1;
        if self.fills.len() < args.max_fills_returned {
            self.fills.push(Fill {
                timestamp,
                side: if buy { "buy" } else { "sell" },
                quantity: delta.abs(),
                price: execution_price,
                fee,
                position_after: self.quantity,
                cash_after: self.cash,
                reason: "target_rebalance",
            });
        }
    }

    fn finish(self, args: &BacktestArgs) -> Result<BacktestResult> {
        if self.bar_count < 2 {
            bail!("backtest requires at least two rows");
        }
        let final_equity = self.previous_equity.context("missing final equity")?;
        let first_close = self.first_close.context("missing first close")?;
        let periods_per_year = 365.0 * 24.0 * 3600.0 / args.timeframe_seconds as f64;
        let (sharpe_ratio, annualized_volatility_pct) = if self.return_count > 1 {
            let variance = (self.return_m2 / (self.return_count - 1) as f64).max(0.0);
            let deviation = variance.sqrt();
            if deviation > 0.0 {
                (
                    Some(self.return_mean / deviation * periods_per_year.sqrt()),
                    Some(deviation * periods_per_year.sqrt() * 100.0),
                )
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };
        let metrics = Metrics {
            initial_equity: args.initial_equity,
            final_equity,
            total_return_pct: (final_equity / args.initial_equity - 1.0) * 100.0,
            max_drawdown_pct: self.max_drawdown,
            sharpe_ratio,
            annualized_volatility_pct,
            buy_and_hold_return_pct: (self.last_close / first_close - 1.0) * 100.0,
            fill_count: self.fill_count,
            fills_returned: self.fills.len(),
            fees_paid: self.fees_paid,
            bar_count: self.bar_count,
            invested_bar_fraction: self.invested_count as f64 / self.bar_count as f64,
        };
        Ok(BacktestResult {
            engine: "rust-arrow",
            engine_version: env!("CARGO_PKG_VERSION"),
            metrics,
            fills: self.fills,
            equity: self.equity,
            first_timestamp: self.first_timestamp,
            last_timestamp: self.last_timestamp,
        })
    }
}

fn typed_column<'a, T: Array + 'static>(batch: &'a RecordBatch, name: &str) -> Result<&'a T> {
    let index = batch
        .schema()
        .index_of(name)
        .with_context(|| format!("missing required column: {name}"))?;
    batch
        .column(index)
        .as_any()
        .downcast_ref::<T>()
        .with_context(|| format!("column {name} has an unexpected Arrow type"))
}

fn run_backtest(args: BacktestArgs) -> Result<()> {
    validate_args(&args)?;
    let file = File::open(&args.input)
        .with_context(|| format!("cannot open input parquet: {}", args.input.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("cannot read parquet metadata")?;
    let total_rows = builder.metadata().file_metadata().num_rows() as u64;
    let mut reader = builder
        .with_batch_size(262_144)
        .build()
        .context("cannot create parquet batch reader")?;
    let mut state = EngineState::new(&args, total_rows);
    let mut processed = 0_u64;

    for batch in &mut reader {
        let batch = batch.context("cannot decode parquet record batch")?;
        let timestamps = typed_column::<Int64Array>(&batch, "timestamp")?;
        let opens = typed_column::<Float64Array>(&batch, "open")?;
        let closes = typed_column::<Float64Array>(&batch, "close")?;
        let targets = typed_column::<Float64Array>(&batch, "target_fraction")?;
        for row in 0..batch.num_rows() {
            if timestamps.is_null(row) || opens.is_null(row) || closes.is_null(row) {
                continue;
            }
            let target = if targets.is_null(row) {
                None
            } else {
                Some(targets.value(row))
            };
            processed += 1;
            state.process_row(
                timestamps.value(row),
                opens.value(row),
                closes.value(row),
                target,
                &args,
                processed == total_rows,
            )?;
        }
    }

    let result = state.finish(&args)?;
    write_json(&args.output, &result)
}

fn write_json(path: &Path, result: &BacktestResult) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("cannot create output directory: {}", parent.display()))?;
    }
    let file = File::create(path)
        .with_context(|| format!("cannot create output file: {}", path.display()))?;
    serde_json::to_writer(BufWriter::new(file), result).context("cannot serialize result")
}

fn validate_args(args: &BacktestArgs) -> Result<()> {
    if args.initial_equity <= 0.0 {
        bail!("initial-equity must be positive");
    }
    if args.timeframe_seconds == 0 {
        bail!("timeframe-seconds must be positive");
    }
    if args.max_leverage <= 0.0 || args.max_leverage > 10.0 {
        bail!("max-leverage must be in (0, 10]");
    }
    if [args.fee_bps, args.slippage_bps, args.spread_bps]
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        bail!("fees, slippage, and spread must be finite and non-negative");
    }
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Backtest(args) => run_backtest(args),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_args() -> BacktestArgs {
        BacktestArgs {
            input: PathBuf::new(),
            output: PathBuf::new(),
            timeframe_seconds: 1,
            initial_equity: 10_000.0,
            fee_bps: 0.0,
            slippage_bps: 0.0,
            spread_bps: 0.0,
            minimum_order_notional: 1.0,
            max_leverage: 1.0,
            allow_short: false,
            max_equity_points: 100,
            max_fills_returned: 100,
        }
    }

    fn assert_single_fill(targets: [Option<f64>; 5]) {
        let args = test_args();
        let prices = [100.0, 200.0, 50.0, 300.0, 80.0];
        let last_index = prices.len() - 1;
        let mut state = EngineState::new(&args, prices.len() as u64);

        for (index, (price, target)) in prices.into_iter().zip(targets).enumerate() {
            state
                .process_row(
                    index as i64 + 1,
                    price,
                    price,
                    target,
                    &args,
                    index == last_index,
                )
                .expect("row should be processed");
        }

        assert_eq!(state.fill_count, 1);
        assert_eq!(state.fills.len(), 1);
        assert_eq!(state.fills[0].timestamp, 2);
        assert!(state.pending_target.is_none());
    }

    #[test]
    fn unchanged_or_sparse_target_is_executed_only_once() {
        assert_single_fill([Some(0.5); 5]);
        assert_single_fill([Some(0.5), None, None, None, None]);
    }
}


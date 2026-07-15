export interface DatasetDescriptor {
  key: string;
  symbol: string;
  base_timeframe_seconds: number;
  raw_files: string[];
  canonical_path: string;
  prepared: boolean;
  row_count: number | null;
  first_timestamp: number | null;
  last_timestamp: number | null;
  source_bytes: number;
  prepared_bytes: number;
  updated_at: string | null;
}

export interface Bar {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  quote_asset_volume?: number;
  number_of_trades?: number;
  taker_buy_base_volume?: number;
  taker_buy_quote_volume?: number;
  symbol?: string;
  timeframe_seconds?: number;
}

export interface Fill {
  timestamp: number;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  fee: number;
  reason: string;
  strategy_key?: string;
  position_after: number;
  cash_after: number;
}

export interface EquityPoint {
  timestamp: number;
  equity: number;
  drawdown_pct: number;
}

export interface PlotPoint {
  timestamp: number;
  values: Record<string, number | null>;
}

export interface IndicatorSpec {
  key: string;
  label: string;
  pane: 'price' | 'indicator' | 'equity';
  format: 'price' | 'number' | 'percent';
  line_width: number;
  time_offset_seconds?: number;
}

export interface StrategyMetadata {
  key: string;
  name: string;
  description: string;
  minimum_bars: number;
  required_timeframe_seconds?: number | null;
  execution_mode: string;
  parameter_schema: {
    properties?: Record<string, {
      title?: string;
      description?: string;
      type?: string;
      default?: unknown;
      minimum?: number;
      maximum?: number;
      enum?: unknown[];
    }>;
  };
  indicator_specs: IndicatorSpec[];
  source: string;
}

export interface BacktestResult {
  id: string;
  engine: string;
  engine_version: string;
  strategy_key: string;
  strategy_name: string;
  parameters: Record<string, unknown>;
  dataset_key: string;
  timeframe_seconds: number;
  start_timestamp: number;
  end_timestamp: number;
  signal_cache_key: string;
  wall_time_seconds: number;
  metrics: Record<string, number | null>;
  fills: Fill[];
  equity: EquityPoint[];
}

export interface ChartPayload {
  bars: Bar[];
  plots: PlotPoint[];
  plot_columns: string[];
  bucket_seconds: number;
  source_row_count: number;
}

export interface Account {
  timestamp: number;
  cash: number;
  quantity: number;
  avg_entry_price: number;
  mark_price: number;
  equity: number;
  realized_pnl: number;
  unrealized_pnl: number;
  fees_paid: number;
  exposure_fraction: number;
}

export interface SessionState {
  id?: string;
  status: string;
  status_message?: string;
  strategy_key?: string;
  strategy_name?: string;
  parameters?: Record<string, unknown>;
  symbol?: string;
  timeframe_seconds?: number;
  bootstrap_source?: string | null;
  bootstrap_bars?: number;
  bootstrap_trade_pages?: number;
  account?: Account | null;
  indicator_specs?: IndicatorSpec[];
}

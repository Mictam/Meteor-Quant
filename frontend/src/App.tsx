import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, websocketUrl } from './api';
import TradingChart from './TradingChart';
import MarketHybridTab from './MarketHybridTab';
import MarketLMTab from './MarketLMTab';
import type {
  Account,
  BacktestResult,
  Bar,
  ChartPayload,
  DatasetDescriptor,
  EquityPoint,
  Fill,
  IndicatorSpec,
  PlotPoint,
  SessionState,
  StrategyMetadata,
} from './types';

type Tab = 'backtest' | 'paper' | 'plugins' | 'marketlm' | 'markethybrid';

interface StrategyResponse {
  strategies: StrategyMetadata[];
  errors: Array<{ file: string; error: string }>;
}

interface PaperCapabilities {
  timeframes_seconds: number[];
  default_bootstrap_bars: number;
  maximum_bootstrap_bars: number;
  trade_bootstrap_max_pages: number;
  subminute_source: string;
}

function defaultsFor(strategy?: StrategyMetadata): Record<string, unknown> {
  const output: Record<string, unknown> = {};
  for (const [key, property] of Object.entries(strategy?.parameter_schema.properties ?? {})) {
    if (property.default !== undefined) output[key] = property.default;
  }
  return output;
}

function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  return `${(value / 1024 ** index).toFixed(index > 1 ? 2 : 0)} ${units[index]}`;
}

function money(value?: number | null): string {
  if (value === undefined || value === null || !Number.isFinite(value)) return '—';
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 }).format(value);
}

function number(value?: number | null, digits = 2): string {
  if (value === undefined || value === null || !Number.isFinite(value)) return '—';
  return value.toLocaleString('en-US', { maximumFractionDigits: digits });
}

function percent(value?: number | null): string {
  if (value === undefined || value === null || !Number.isFinite(value)) return '—';
  return `${value.toFixed(2)}%`;
}

function dateLabel(timestamp?: number | null): string {
  if (!timestamp) return '—';
  return new Date(timestamp * 1000).toLocaleString();
}

function timeframeLabel(seconds: number): string {
  if (seconds < 60) return `${seconds} second${seconds === 1 ? '' : 's'}`;
  if (seconds % 3600 === 0) {
    const hours = seconds / 3600;
    return `${hours} hour${hours === 1 ? '' : 's'}`;
  }
  const minutes = seconds / 60;
  return `${minutes} minute${minutes === 1 ? '' : 's'}`;
}

function asTimestamp(value: string): number | null {
  if (!value) return null;
  const parsed = new Date(value).getTime();
  return Number.isFinite(parsed) ? Math.floor(parsed / 1000) : null;
}

function upsertBar(previous: Bar[], incoming: Bar): Bar[] {
  const next = [...previous];
  const index = next.findIndex((bar) => bar.timestamp === incoming.timestamp);
  if (index >= 0) next[index] = incoming;
  else next.push(incoming);
  next.sort((a, b) => a.timestamp - b.timestamp);
  return next.slice(-5_000);
}

function upsertPlot(previous: PlotPoint[], incoming: PlotPoint): PlotPoint[] {
  const next = [...previous];
  const index = next.findIndex((point) => point.timestamp === incoming.timestamp);
  if (index >= 0) next[index] = incoming;
  else next.push(incoming);
  next.sort((a, b) => a.timestamp - b.timestamp);
  return next.slice(-5_000);
}

function Field({ label, children, hint }: { label: string; children: React.ReactNode; hint?: string }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
      {hint && <small>{hint}</small>}
    </label>
  );
}

function StrategyParameters({
  strategy,
  values,
  onChange,
}: {
  strategy?: StrategyMetadata;
  values: Record<string, unknown>;
  onChange: (values: Record<string, unknown>) => void;
}) {
  const properties = strategy?.parameter_schema.properties ?? {};
  if (Object.keys(properties).length === 0) return <div className="empty-inline">No configurable parameters.</div>;
  return (
    <div className="parameter-grid">
      {Object.entries(properties).map(([key, property]) => {
        const label = property.title ?? key.replaceAll('_', ' ');
        if (property.type === 'boolean') {
          return (
            <label className="toggle-field" key={key}>
              <input
                type="checkbox"
                checked={Boolean(values[key])}
                onChange={(event) => onChange({ ...values, [key]: event.target.checked })}
              />
              <span>{label}</span>
            </label>
          );
        }
        if (property.enum) {
          return (
            <Field key={key} label={label} hint={property.description}>
              <select value={String(values[key] ?? property.enum[0] ?? '')} onChange={(event) => onChange({ ...values, [key]: event.target.value })}>
                {property.enum.map((item) => <option value={String(item)} key={String(item)}>{String(item)}</option>)}
              </select>
            </Field>
          );
        }
        return (
          <Field key={key} label={label} hint={property.description}>
            <input
              type={property.type === 'integer' || property.type === 'number' ? 'number' : 'text'}
              min={property.minimum}
              max={property.maximum}
              step={property.type === 'integer' ? 1 : 'any'}
              value={String(values[key] ?? '')}
              onChange={(event) => {
                const raw = event.target.value;
                const next = property.type === 'integer'
                  ? Number.parseInt(raw, 10)
                  : property.type === 'number'
                    ? Number.parseFloat(raw)
                    : raw;
                onChange({ ...values, [key]: typeof next === 'number' && Number.isNaN(next) ? raw : next });
              }}
            />
          </Field>
        );
      })}
    </div>
  );
}

function Metric({ label, value, tone }: { label: string; value: string; tone?: 'good' | 'bad' }) {
  return <div className={`metric ${tone ?? ''}`}><span>{label}</span><strong>{value}</strong></div>;
}

function BacktestMetrics({ result }: { result: BacktestResult | null }) {
  const metrics = result?.metrics;
  return (
    <div className="metrics-grid">
      <Metric label="Final equity" value={money(metrics?.final_equity)} />
      <Metric label="Total return" value={percent(metrics?.total_return_pct)} tone={(metrics?.total_return_pct ?? 0) >= 0 ? 'good' : 'bad'} />
      <Metric label="Max drawdown" value={percent(metrics?.max_drawdown_pct)} tone="bad" />
      <Metric label="Sharpe" value={number(metrics?.sharpe_ratio)} />
      <Metric label="Buy & hold" value={percent(metrics?.buy_and_hold_return_pct)} />
      <Metric label="Fees" value={money(metrics?.fees_paid)} />
      <Metric label="Fills" value={number(metrics?.fill_count, 0)} />
      <Metric label="Bars" value={number(metrics?.bar_count, 0)} />
      <Metric label="Engine" value={result?.engine ?? '—'} />
      <Metric label="Wall time" value={result ? `${result.wall_time_seconds.toFixed(2)}s` : '—'} />
    </div>
  );
}

function AccountMetrics({ account }: { account?: Account | null }) {
  return (
    <div className="metrics-grid">
      <Metric label="Equity" value={money(account?.equity)} />
      <Metric label="Cash" value={money(account?.cash)} />
      <Metric label="Position BTC" value={number(account?.quantity, 8)} />
      <Metric label="Exposure" value={account ? percent(account.exposure_fraction * 100) : '—'} />
      <Metric label="Unrealized P&L" value={money(account?.unrealized_pnl)} tone={(account?.unrealized_pnl ?? 0) >= 0 ? 'good' : 'bad'} />
      <Metric label="Realized P&L" value={money(account?.realized_pnl)} tone={(account?.realized_pnl ?? 0) >= 0 ? 'good' : 'bad'} />
      <Metric label="Fees" value={money(account?.fees_paid)} />
      <Metric label="Mark" value={money(account?.mark_price)} />
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>('backtest');
  const [datasets, setDatasets] = useState<DatasetDescriptor[]>([]);
  const [strategies, setStrategies] = useState<StrategyMetadata[]>([]);
  const [pluginErrors, setPluginErrors] = useState<Array<{ file: string; error: string }>>([]);
  const [selectedKey, setSelectedKey] = useState('sma_cross');
  const selected = useMemo(() => strategies.find((item) => item.key === selectedKey), [strategies, selectedKey]);
  const [parameters, setParameters] = useState<Record<string, unknown>>({});
  const [timeframeSeconds, setTimeframeSeconds] = useState(60);
  const [paperBootstrapBars, setPaperBootstrapBars] = useState(500);
  const [paperCapabilities, setPaperCapabilities] = useState<PaperCapabilities>({
    timeframes_seconds: [1, 5, 15, 30, 60, 300, 900, 1800, 3600],
    default_bootstrap_bars: 500,
    maximum_bootstrap_bars: 20000,
    trade_bootstrap_max_pages: 120,
    subminute_source: 'Kraken recent trades',
  });
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [initialEquity, setInitialEquity] = useState(10_000);
  const [feeBps, setFeeBps] = useState(10);
  const [slippageBps, setSlippageBps] = useState(1);
  const [spreadBps, setSpreadBps] = useState(1);
  const [allowShort, setAllowShort] = useState(false);
  const [maxLeverage, setMaxLeverage] = useState(1);
  const [engine, setEngine] = useState<'auto' | 'rust' | 'python'>('auto');
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [chart, setChart] = useState<ChartPayload | null>(null);
  const [session, setSession] = useState<SessionState>({ status: 'stopped' });
  const [liveBars, setLiveBars] = useState<Bar[]>([]);
  const [liveFills, setLiveFills] = useState<Fill[]>([]);
  const [livePlots, setLivePlots] = useState<PlotPoint[]>([]);
  const [liveEquity, setLiveEquity] = useState<EquityPoint[]>([]);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const websocket = useRef<WebSocket | null>(null);

  const loadMetadata = useCallback(async () => {
    const [datasetResponse, strategyResponse, state, capabilities] = await Promise.all([
      api<{ datasets: DatasetDescriptor[] }>('/api/datasets'),
      api<StrategyResponse>('/api/strategies'),
      api<SessionState>('/api/paper/state'),
      api<PaperCapabilities>('/api/paper/capabilities'),
    ]);
    setDatasets(datasetResponse.datasets);
    setStrategies(strategyResponse.strategies);
    setPluginErrors(strategyResponse.errors);
    setSession(state);
    setPaperCapabilities(capabilities);
    const resolvedKey = strategyResponse.strategies.some((item) => item.key === selectedKey)
      ? selectedKey
      : strategyResponse.strategies[0]?.key ?? '';
    setSelectedKey(resolvedKey);
    setParameters(defaultsFor(strategyResponse.strategies.find((item) => item.key === resolvedKey)));
  }, [selectedKey]);

  useEffect(() => {
    loadMetadata().catch((cause: unknown) => setError(cause instanceof Error ? cause.message : String(cause)));
  }, []);

  useEffect(() => {
    if (selected) {
      setParameters(defaultsFor(selected));
      if (
        selected.required_timeframe_seconds !== null
        && selected.required_timeframe_seconds !== undefined
      ) {
        setTimeframeSeconds(selected.required_timeframe_seconds);
      }
    }
  }, [selectedKey]);

  useEffect(() => {
    let disposed = false;
    let reconnect = 0;
    const connect = () => {
      if (disposed) return;
      const ws = new WebSocket(websocketUrl('/ws/events'));
      websocket.current = ws;
      ws.onopen = () => ws.send('ready');
      ws.onmessage = (event) => {
        const message = JSON.parse(event.data) as { type: string; payload: any };
        if (message.type === 'bootstrap') {
          setLiveBars(message.payload.bars ?? []);
          setLivePlots(message.payload.plots ?? []);
          setSession(message.payload.session);
          setLiveFills([]);
          setLiveEquity([]);
        } else if (message.type === 'bar_update' || message.type === 'bar_closed') {
          setLiveBars((current) => upsertBar(current, message.payload as Bar));
        } else if (message.type === 'fill') {
          setLiveFills((current) => [...current, message.payload as Fill]);
        } else if (message.type === 'plots') {
          setLivePlots((current) => upsertPlot(current, message.payload as PlotPoint));
        } else if (message.type === 'account') {
          const account = message.payload as Account;
          setSession((current) => ({ ...current, account }));
          setLiveEquity((current) => [...current, { timestamp: account.timestamp, equity: account.equity, drawdown_pct: 0 }].slice(-5_000));
        } else if (message.type === 'session') {
          setSession(message.payload as SessionState);
        }
      };
      ws.onclose = () => {
        websocket.current = null;
        if (!disposed) reconnect = window.setTimeout(connect, 1_500);
      };
    };
    connect();
    return () => {
      disposed = true;
      window.clearTimeout(reconnect);
      websocket.current?.close();
    };
  }, []);

  const perform = async (label: string, operation: () => Promise<void>) => {
    setBusy(label);
    setError('');
    try {
      await operation();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy('');
    }
  };

  const dataset = datasets[0];
  const indicatorSpecs: IndicatorSpec[] = selected?.indicator_specs ?? [];

  const prepareDataset = () => perform('Preparing Parquet dataset…', async () => {
    await api('/api/datasets/prepare', {
      method: 'POST',
      body: JSON.stringify({ dataset_key: dataset?.key ?? 'btcusdt_1s', force: false }),
    });
    await loadMetadata();
  });

  const runBacktest = () => perform('Computing signals and running backtest…', async () => {
    const backtest = await api<BacktestResult>('/api/backtests', {
      method: 'POST',
      body: JSON.stringify({
        dataset_key: dataset?.key ?? 'btcusdt_1s',
        strategy_key: selectedKey,
        parameters,
        timeframe_seconds: timeframeSeconds,
        start_timestamp: asTimestamp(startDate),
        end_timestamp: asTimestamp(endDate),
        initial_equity: initialEquity,
        fee_bps: feeBps,
        slippage_bps: slippageBps,
        spread_bps: spreadBps,
        allow_short: allowShort,
        max_leverage: maxLeverage,
        engine,
      }),
    });
    setResult(backtest);
    const chartPayload = await api<ChartPayload>(`/api/backtests/${backtest.id}/chart?max_points=5000`);
    chartPayload.bars = chartPayload.bars.map((bar) => ({ ...bar, timeframe_seconds: chartPayload.bucket_seconds }));
    setChart(chartPayload);
  });

  const startPaper = () => perform('Starting Kraken paper session…', async () => {
    const state = await api<SessionState>('/api/paper/start', {
      method: 'POST',
      body: JSON.stringify({
        strategy_key: selectedKey,
        parameters,
        timeframe_seconds: timeframeSeconds,
        bootstrap_bars: paperBootstrapBars,
        initial_equity: initialEquity,
        fee_bps: feeBps,
        slippage_bps: slippageBps,
        allow_short: allowShort,
        max_leverage: maxLeverage,
      }),
    });
    setSession(state);
  });

  const stopPaper = () => perform('Stopping paper session…', async () => {
    setSession(await api<SessionState>('/api/paper/stop', { method: 'POST' }));
  });

  const reloadPlugins = () => perform('Reloading Python plugins…', async () => {
    const response = await api<StrategyResponse>('/api/strategies/reload', { method: 'POST' });
    setStrategies(response.strategies);
    setPluginErrors(response.errors);
  });

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">RUST · PYTHON · POLARS · REACT</div>
          <h1>Meteor Quant</h1>
        </div>
        <div className="status-cluster">
          <span className={`status-dot ${dataset?.prepared ? 'good' : 'warn'}`} />
          <span>{dataset?.prepared ? 'Parquet ready' : 'CSV source detected'}</span>
          <span className="divider" />
          <span className={`status-dot ${session.status === 'running' ? 'good' : ''}`} />
          <span>Kraken paper: {session.status}</span>
        </div>
      </header>

      <nav className="tabs">
        <button className={tab === 'backtest' ? 'active' : ''} onClick={() => setTab('backtest')}>Backtest lab</button>
        <button className={tab === 'paper' ? 'active' : ''} onClick={() => setTab('paper')}>Kraken paper</button>
        <button className={tab === 'plugins' ? 'active' : ''} onClick={() => setTab('plugins')}>Python plugins</button>
        <button className={tab === 'marketlm' ? 'active' : ''} onClick={() => setTab('marketlm')}>Train MarketLM</button>
        <button className={tab === 'markethybrid' ? 'active' : ''} onClick={() => setTab('markethybrid')}>Train MarketHybrid</button>
      </nav>

      {error && <div className="alert error"><strong>Action failed</strong><span>{error}</span></div>}
      {busy && <div className="progress"><span className="spinner" />{busy}</div>}

      {tab === 'backtest' && (
        <main className="workspace">
          <aside className="control-panel">
            <section className="panel-section">
              <div className="section-heading"><h2>Dataset</h2><span>BTCUSDT · 1 second</span></div>
              <div className="dataset-card">
                <strong>{dataset?.prepared ? 'Canonical Parquet cache' : 'Two Binance CSV files'}</strong>
                <span>{number(dataset?.row_count, 0)} rows · {formatBytes(dataset?.prepared_bytes || dataset?.source_bytes || 0)}</span>
                <span>{dateLabel(dataset?.first_timestamp)} → {dateLabel(dataset?.last_timestamp)}</span>
                <div className="file-list">{dataset?.raw_files.map((path) => <code key={path}>{path.split(/[\\/]/).pop()}</code>)}</div>
              </div>
              <button className="secondary full" disabled={Boolean(busy)} onClick={prepareDataset}>{dataset?.prepared ? 'Validate / refresh cache' : 'Prepare Parquet cache'}</button>
            </section>

            <section className="panel-section">
              <div className="section-heading"><h2>Strategy</h2><span>{selected?.source === 'builtin' ? 'Built in' : 'Python plugin'}</span></div>
              <Field label="Strategy">
                <select value={selectedKey} onChange={(event) => setSelectedKey(event.target.value)}>
                  {strategies.map((strategy) => <option key={strategy.key} value={strategy.key}>{strategy.name}</option>)}
                </select>
              </Field>
              <p className="muted">{selected?.description}</p>
              <StrategyParameters strategy={selected} values={parameters} onChange={setParameters} />
            </section>

            <section className="panel-section">
              <div className="section-heading"><h2>Range & execution</h2><span>Next-bar open fills</span></div>
              <div className="two-col">
                <Field label="Timeframe">
                  <select value={timeframeSeconds} onChange={(event) => setTimeframeSeconds(Number(event.target.value))}>
                    <option value={1}>1 second</option><option value={5}>5 seconds</option><option value={15}>15 seconds</option>
                    <option value={60}>1 minute</option><option value={300}>5 minutes</option><option value={900}>15 minutes</option>
                    <option value={3600}>1 hour</option>
                  </select>
                </Field>
                <Field label="Engine">
                  <select value={engine} onChange={(event) => setEngine(event.target.value as 'auto' | 'rust' | 'python')}>
                    <option value="auto">Auto (prefer Rust)</option><option value="rust">Rust required</option><option value="python">Python Arrow fallback</option>
                  </select>
                </Field>
              </div>
              <Field label="Start (empty = beginning)"><input type="datetime-local" value={startDate} onChange={(event) => setStartDate(event.target.value)} /></Field>
              <Field label="End (empty = end)"><input type="datetime-local" value={endDate} onChange={(event) => setEndDate(event.target.value)} /></Field>
              <div className="two-col">
                <Field label="Initial equity"><input type="number" value={initialEquity} min={100} onChange={(event) => setInitialEquity(Number(event.target.value))} /></Field>
                <Field label="Max leverage"><input type="number" value={maxLeverage} min={0.01} max={10} step={0.1} onChange={(event) => setMaxLeverage(Number(event.target.value))} /></Field>
                <Field label="Fee (bps)"><input type="number" value={feeBps} min={0} step={0.1} onChange={(event) => setFeeBps(Number(event.target.value))} /></Field>
                <Field label="Slippage (bps)"><input type="number" value={slippageBps} min={0} step={0.1} onChange={(event) => setSlippageBps(Number(event.target.value))} /></Field>
                <Field label="Spread (bps)"><input type="number" value={spreadBps} min={0} step={0.1} onChange={(event) => setSpreadBps(Number(event.target.value))} /></Field>
                <label className="toggle-field compact"><input type="checkbox" checked={allowShort} onChange={(event) => setAllowShort(event.target.checked)} /><span>Allow short</span></label>
              </div>
              <button className="primary full" disabled={Boolean(busy) || !selectedKey} onClick={runBacktest}>Run authoritative backtest</button>
            </section>
          </aside>

          <section className="result-panel">
            <BacktestMetrics result={result} />
            {chart ? (
              <TradingChart bars={chart.bars} fills={result?.fills ?? []} plots={chart.plots} indicatorSpecs={indicatorSpecs} equity={result?.equity ?? []} />
            ) : (
              <div className="hero-empty"><strong>Ready for multi-year 1-second research</strong><span>Prepare the dataset, select a strategy, and run. Chart payloads are downsampled; the engine still evaluates every bar.</span></div>
            )}
            {result && (
              <section className="run-summary">
                <div><span>Evaluated range</span><strong>{dateLabel(result.start_timestamp)} → {dateLabel(result.end_timestamp)}</strong></div>
                <div><span>Signal rows</span><strong>{number(chart?.source_row_count ?? result.metrics.bar_count, 0)}</strong></div>
                <div><span>Chart bucket</span><strong>{chart?.bucket_seconds}s</strong></div>
                <div><span>Result ID</span><code>{result.id}</code></div>
              </section>
            )}
          </section>
        </main>
      )}

      {tab === 'paper' && (
        <main className="workspace">
          <aside className="control-panel">
            <section className="panel-section">
              <div className="section-heading"><h2>Kraken BTC/USD</h2><span>Paper only</span></div>
              <Field label="Strategy"><select value={selectedKey} onChange={(event) => setSelectedKey(event.target.value)}>{strategies.map((strategy) => <option key={strategy.key} value={strategy.key}>{strategy.name}</option>)}</select></Field>
              <StrategyParameters strategy={selected} values={parameters} onChange={setParameters} />
              <Field label="Live timeframe">
                <select value={timeframeSeconds} onChange={(event) => setTimeframeSeconds(Number(event.target.value))}>
                  {paperCapabilities.timeframes_seconds.map((seconds) => (
                    <option key={seconds} value={seconds}>{timeframeLabel(seconds)}</option>
                  ))}
                </select>
              </Field>
              <Field
                label="Bootstrap bars"
                hint={`Sub-minute history is reconstructed from ${paperCapabilities.subminute_source}.`}
              >
                <input
                  type="number"
                  min={2}
                  max={paperCapabilities.maximum_bootstrap_bars}
                  step={1}
                  value={paperBootstrapBars}
                  onChange={(event) => setPaperBootstrapBars(Number(event.target.value))}
                />
              </Field>
              <div className="two-col">
                <Field label="Initial equity"><input type="number" value={initialEquity} min={100} onChange={(event) => setInitialEquity(Number(event.target.value))} /></Field>
                <Field label="Max leverage"><input type="number" value={maxLeverage} min={0.01} max={10} step={0.1} onChange={(event) => setMaxLeverage(Number(event.target.value))} /></Field>
                <Field label="Fee (bps)"><input type="number" value={feeBps} min={0} step={0.1} onChange={(event) => setFeeBps(Number(event.target.value))} /></Field>
                <Field label="Slippage (bps)"><input type="number" value={slippageBps} min={0} step={0.1} onChange={(event) => setSlippageBps(Number(event.target.value))} /></Field>
                <label className="toggle-field compact"><input type="checkbox" checked={allowShort} onChange={(event) => setAllowShort(event.target.checked)} /><span>Allow short</span></label>
              </div>
              <p className="muted">Kraken bid/ask data supplies the live spread. Fee and slippage assumptions are applied to every simulated fill.</p>
              <div className="button-row">
                <button className="primary" disabled={Boolean(busy) || session.status === 'running'} onClick={startPaper}>Start paper</button>
                <button className="danger" disabled={Boolean(busy) || session.status === 'stopped'} onClick={stopPaper}>Stop</button>
              </div>
              <div className="session-state">
                <span>Status</span><strong>{session.status}</strong><small>{session.status_message}</small>
                {session.bootstrap_source && <small>{session.bootstrap_source} · {session.bootstrap_bars ?? 0} bars</small>}
              </div>
            </section>
          </aside>
          <section className="result-panel">
            <AccountMetrics account={session.account} />
            <TradingChart bars={liveBars} fills={liveFills} plots={livePlots} indicatorSpecs={session.indicator_specs ?? indicatorSpecs} equity={liveEquity} />
          </section>
        </main>
      )}

      {tab === 'plugins' && (
        <main className="plugin-workspace">
          <section className="plugin-intro">
            <div><div className="eyebrow">ONE DEFINITION · TWO RUNTIMES</div><h2>Add strategies in Python</h2></div>
            <button className="secondary" onClick={reloadPlugins} disabled={Boolean(busy)}>Reload plugins</button>
          </section>
          <div className="plugin-grid">
            {strategies.map((strategy) => (
              <article className="plugin-card" key={strategy.key}>
                <div><strong>{strategy.name}</strong><code>{strategy.key}</code></div>
                <p>{strategy.description}</p>
                <span>{strategy.source}</span>
              </article>
            ))}
          </div>
          <section className="code-help">
            <h3>Plugin contract</h3>
            <p>Subclass <code>StrategyPlugin</code>, implement <code>build_signals()</code> with lazy Polars expressions, and implement <code>on_live_bar()</code> for Kraken paper execution. Export the class as <code>STRATEGY</code>.</p>
            <code>user_strategies/example_strategy.py</code>
          </section>
          {pluginErrors.length > 0 && <div className="alert error"><strong>Plugin load errors</strong>{pluginErrors.map((item) => <span key={item.file}>{item.file}: {item.error}</span>)}</div>}
        </main>
      )}

      {tab === 'marketlm' && (
        <MarketLMTab dataset={dataset} onModelRegistered={loadMetadata} />
      )}

      {tab === 'markethybrid' && (
        <MarketHybridTab dataset={dataset} onModelRegistered={loadMetadata} />
      )}
    </div>
  );
}

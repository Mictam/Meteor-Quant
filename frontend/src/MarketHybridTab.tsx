import { useCallback, useEffect, useMemo, useState } from 'react';
import { api } from './api';
import type { DatasetDescriptor } from './types';

interface IndicatorParameterDefinition {
  type: 'integer' | 'number';
  default: number;
  minimum?: number;
  maximum?: number;
}

interface IndicatorDefinition {
  key: string;
  name: string;
  description: string;
  parameters: Record<string, IndicatorParameterDefinition>;
  outputs: string[];
}

interface IndicatorSelection {
  key: string;
  parameters: Record<string, number>;
}

interface ModelConfig {
  d_model: number;
  n_layers: number;
  n_heads: number;
  mlp_hidden: number;
  dropout: number;
  rope_base: number;
  activation_checkpointing: boolean;
}

interface PredictorConfig {
  d_model: number;
  n_layers: number;
  n_heads: number;
  mlp_hidden: number;
  dropout: number;
}

interface MarketHybridDefaultConfig {
  name: string;
  data: {
    timeframe_seconds: number;
    horizons_seconds: number[];
    patch_size: number;
    context_patches: number;
    train_fraction: number;
    validation_fraction: number;
    cost_threshold_bps: number;
  };
  model: ModelConfig;
  predictor: PredictorConfig;
  jepa: {
    target_patch_offsets: number[];
    ema_start: number;
    ema_end: number;
    latent_loss_weight: number;
    variance_loss_weight: number;
    covariance_loss_weight: number;
  };
  policy: {
    horizon_weights: Record<string, number>;
    position_scale_bps: number;
    confidence_temperature: number;
    intent_deadband: number;
  };
  training: Record<string, unknown> & {
    max_steps: number;
    batch_size: number;
    gradient_accumulation_steps: number;
    learning_rate: number;
    min_learning_rate: number;
    warmup_steps: number;
    validation_interval: number;
    checkpoint_interval: number;
    num_workers: number;
    amp: 'auto' | 'bf16' | 'fp16' | 'off';
    compile: 'off' | 'on';
    stages: Array<Record<string, unknown> & { name: string; end_step: number }>;
  };
}

interface MarketHybridCapabilities {
  torch_available: boolean;
  torch_version: string | null;
  cuda_available: boolean;
  cuda_version: string | null;
  gpu_name: string | null;
  indicators: IndicatorDefinition[];
  default_indicators: IndicatorSelection[];
  timeframes_seconds: number[];
  model_presets: Record<string, ModelConfig>;
  default_config: MarketHybridDefaultConfig;
}

interface MarketHybridRun {
  run_id: string;
  name: string;
  kind: 'prepare' | 'train';
  state: 'queued' | 'preparing' | 'training' | 'completed' | 'failed' | 'stopping' | 'stopped';
  created_at: string;
  updated_at: string;
  progress: number;
  message: string;
  step: number;
  max_steps: number;
  metrics: Record<string, number | null>;
  prepared_dir?: string | null;
  checkpoint_path?: string | null;
  error?: string | null;
  registered?: boolean;
  metric_history?: Array<Record<string, number>>;
  log_tail?: string;
  stage?: { index: number; name: string; local_step: number; start_step: number; end_step: number; progress: number } | null;
  learning_rates?: Record<string, number>;
}

interface RegisteredModel {
  model_id: string;
  display_name: string;
  description: string;
  timeframe_seconds: number;
  horizons_seconds: number[];
  primary_horizon_seconds: number;
  registered_at: string;
  available: boolean;
}

interface MarketHybridResponse {
  runs: MarketHybridRun[];
  models: RegisteredModel[];
}

interface IndicatorState {
  enabled: boolean;
  parameters: Record<string, number>;
}

function asTimestamp(value: string): number | null {
  if (!value) return null;
  const parsed = new Date(value).getTime();
  return Number.isFinite(parsed) ? Math.floor(parsed / 1000) : null;
}

function formatNumber(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return value.toLocaleString('en-US', { maximumFractionDigits: digits });
}

function formatTimeframe(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  if (seconds % 3600 === 0) return `${seconds / 3600}h`;
  return `${seconds / 60}m`;
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

function initialIndicatorState(capabilities: MarketHybridCapabilities): Record<string, IndicatorState> {
  const defaults = new Map(capabilities.default_indicators.map((item) => [item.key, item.parameters]));
  return Object.fromEntries(capabilities.indicators.map((indicator) => [
    indicator.key,
    {
      enabled: defaults.has(indicator.key),
      parameters: Object.fromEntries(Object.entries(indicator.parameters).map(([key, definition]) => [
        key,
        Number(defaults.get(indicator.key)?.[key] ?? definition.default),
      ])),
    },
  ]));
}

export default function MarketHybridTab({
  dataset,
  onModelRegistered,
}: {
  dataset?: DatasetDescriptor;
  onModelRegistered: () => Promise<void>;
}) {
  const [capabilities, setCapabilities] = useState<MarketHybridCapabilities | null>(null);
  const [runs, setRuns] = useState<MarketHybridRun[]>([]);
  const [models, setModels] = useState<RegisteredModel[]>([]);
  const [selectedRunId, setSelectedRunId] = useState('');
  const [selectedRun, setSelectedRun] = useState<MarketHybridRun | null>(null);
  const [indicatorState, setIndicatorState] = useState<Record<string, IndicatorState>>({});
  const [name, setName] = useState('btc-markethybrid-15s-quality');
  const [timeframeSeconds, setTimeframeSeconds] = useState(15);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [horizons, setHorizons] = useState('30,60,180,300,900');
  const [patchSize, setPatchSize] = useState(8);
  const [contextPatches, setContextPatches] = useState(256);
  const [trainFraction, setTrainFraction] = useState(0.8);
  const [validationFraction, setValidationFraction] = useState(0.1);
  const [costThresholdBps, setCostThresholdBps] = useState(30);
  const [preset, setPreset] = useState('quality_4060ti');
  const [model, setModel] = useState<ModelConfig>({
    d_model: 384,
    n_layers: 12,
    n_heads: 6,
    mlp_hidden: 1152,
    dropout: 0.08,
    rope_base: 10000,
    activation_checkpointing: true,
  });
  const [predictor, setPredictor] = useState<PredictorConfig>({ d_model: 256, n_layers: 4, n_heads: 4, mlp_hidden: 768, dropout: 0.08 });
  const [jepaOffsets, setJepaOffsets] = useState('1,2,4,7');
  const [emaStart, setEmaStart] = useState(0.996);
  const [emaEnd, setEmaEnd] = useState(0.9999);
  const [latentLossWeight, setLatentLossWeight] = useState(1);
  const [varianceLossWeight, setVarianceLossWeight] = useState(0.05);
  const [covarianceLossWeight, setCovarianceLossWeight] = useState(0.005);
  const [jepaLossWeight, setJepaLossWeight] = useState(0.5);
  const [actionableLossWeight, setActionableLossWeight] = useState(1.5);
  const [policyPositionLossWeight, setPolicyPositionLossWeight] = useState(0.75);
  const [policyConfidenceLossWeight, setPolicyConfidenceLossWeight] = useState(0.5);
  const [policyIntentLossWeight, setPolicyIntentLossWeight] = useState(0.75);
  const [positionScaleBps, setPositionScaleBps] = useState(30);
  const [confidenceTemperature, setConfidenceTemperature] = useState(2);
  const [maxSteps, setMaxSteps] = useState(40_000);
  const [batchSize, setBatchSize] = useState(8);
  const [gradientAccumulation, setGradientAccumulation] = useState(4);
  const [learningRate, setLearningRate] = useState(0.00015);
  const [minLearningRate, setMinLearningRate] = useState(0.000015);
  const [warmupSteps, setWarmupSteps] = useState(2_000);
  const [validationInterval, setValidationInterval] = useState(500);
  const [checkpointInterval, setCheckpointInterval] = useState(2_000);
  const [numWorkers, setNumWorkers] = useState(0);
  const [amp, setAmp] = useState<'auto' | 'bf16' | 'fp16' | 'off'>('auto');
  const [compile, setCompile] = useState<'off' | 'on'>('off');
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');

  const applyDefaults = (defaults: MarketHybridDefaultConfig) => {
    setName(defaults.name);
    setTimeframeSeconds(defaults.data.timeframe_seconds);
    setHorizons(defaults.data.horizons_seconds.join(','));
    setPatchSize(defaults.data.patch_size);
    setContextPatches(defaults.data.context_patches);
    setTrainFraction(defaults.data.train_fraction);
    setValidationFraction(defaults.data.validation_fraction);
    setCostThresholdBps(defaults.data.cost_threshold_bps);
    setModel(defaults.model);
    setPredictor(defaults.predictor);
    setJepaOffsets(defaults.jepa.target_patch_offsets.join(','));
    setEmaStart(defaults.jepa.ema_start);
    setEmaEnd(defaults.jepa.ema_end);
    setLatentLossWeight(defaults.jepa.latent_loss_weight);
    setVarianceLossWeight(defaults.jepa.variance_loss_weight);
    setCovarianceLossWeight(defaults.jepa.covariance_loss_weight);
    setPositionScaleBps(defaults.policy.position_scale_bps);
    setConfidenceTemperature(defaults.policy.confidence_temperature);
    setMaxSteps(defaults.training.max_steps);
    setBatchSize(defaults.training.batch_size);
    setGradientAccumulation(defaults.training.gradient_accumulation_steps);
    setLearningRate(defaults.training.learning_rate);
    setMinLearningRate(defaults.training.min_learning_rate);
    setWarmupSteps(defaults.training.warmup_steps);
    setValidationInterval(defaults.training.validation_interval);
    setCheckpointInterval(defaults.training.checkpoint_interval);
    setNumWorkers(defaults.training.num_workers);
    setAmp(defaults.training.amp);
    setCompile(defaults.training.compile);
  };

  const refresh = useCallback(async () => {
    const response = await api<MarketHybridResponse>('/api/markethybrid/runs');
    setRuns(response.runs);
    setModels(response.models);
    if (!selectedRunId && response.runs[0]) setSelectedRunId(response.runs[0].run_id);
  }, [selectedRunId]);

  useEffect(() => {
    Promise.all([
      api<MarketHybridCapabilities>('/api/markethybrid/capabilities'),
      api<MarketHybridResponse>('/api/markethybrid/runs'),
    ]).then(([capabilityResponse, runResponse]) => {
      setCapabilities(capabilityResponse);
      setIndicatorState(initialIndicatorState(capabilityResponse));
      setRuns(runResponse.runs);
      setModels(runResponse.models);
      applyDefaults(capabilityResponse.default_config);
      if (runResponse.runs[0]) setSelectedRunId(runResponse.runs[0].run_id);
    }).catch((cause: unknown) => setError(cause instanceof Error ? cause.message : String(cause)));
  }, []);

  useEffect(() => {
    if (!selectedRunId) {
      setSelectedRun(null);
      return;
    }
    api<MarketHybridRun>(`/api/markethybrid/runs/${selectedRunId}`)
      .then(setSelectedRun)
      .catch((cause: unknown) => setError(cause instanceof Error ? cause.message : String(cause)));
  }, [selectedRunId, runs]);

  useEffect(() => {
    const active = runs.some((run) => ['queued', 'preparing', 'training', 'stopping'].includes(run.state));
    if (!active) return;
    const timer = window.setInterval(() => {
      refresh().catch((cause: unknown) => setError(cause instanceof Error ? cause.message : String(cause)));
    }, 2_000);
    return () => window.clearInterval(timer);
  }, [runs, refresh]);

  const selectedIndicators = useMemo(() => Object.entries(indicatorState)
    .filter(([, state]) => state.enabled)
    .map(([key, state]) => ({ key, parameters: state.parameters })), [indicatorState]);

  const parsedOffsets = useMemo(() => jepaOffsets.split(',').map((value) => Number.parseInt(value.trim(), 10)).filter((value) => Number.isFinite(value) && value > 0).sort((a, b) => a - b), [jepaOffsets]);

  const parsedHorizons = useMemo(() => horizons.split(',')
    .map((value) => Number.parseInt(value.trim(), 10))
    .filter((value) => Number.isFinite(value) && value > 0)
    .sort((a, b) => a - b), [horizons]);

  const runOperation = async (label: string, operation: () => Promise<void>) => {
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

  const requestPayload = (prepareOnly: boolean) => ({
    name,
    prepare_only: prepareOnly,
    data: {
      dataset_key: dataset?.key ?? 'btcusdt_1s',
      timeframe_seconds: timeframeSeconds,
      start_timestamp: asTimestamp(startDate),
      end_timestamp: asTimestamp(endDate),
      indicators: selectedIndicators,
      horizons_seconds: parsedHorizons,
      patch_size: patchSize,
      context_patches: contextPatches,
      train_fraction: trainFraction,
      validation_fraction: validationFraction,
      purge_seconds: parsedHorizons.length > 0 ? Math.max(...parsedHorizons) : timeframeSeconds,
      cost_threshold_bps: costThresholdBps,
    },
    model,
    predictor,
    jepa: {
      target_patch_offsets: parsedOffsets,
      ema_start: emaStart,
      ema_end: emaEnd,
      latent_loss_weight: latentLossWeight,
      variance_loss_weight: varianceLossWeight,
      covariance_loss_weight: covarianceLossWeight,
    },
    policy: {
      horizon_weights: Object.fromEntries(parsedHorizons.map((horizon) => [horizon, capabilities?.default_config.policy.horizon_weights[String(horizon)] ?? 1])),
      position_scale_bps: positionScaleBps,
      confidence_temperature: confidenceTemperature,
      intent_deadband: capabilities?.default_config.policy.intent_deadband ?? 0.25,
    },
    training: {
      ...capabilities?.default_config.training,
      max_steps: maxSteps,
      batch_size: batchSize,
      gradient_accumulation_steps: gradientAccumulation,
      learning_rate: learningRate,
      min_learning_rate: minLearningRate,
      warmup_steps: warmupSteps,
      validation_interval: validationInterval,
      checkpoint_interval: checkpointInterval,
      num_workers: numWorkers,
      stages: (capabilities?.default_config.training.stages ?? []).map((stage, index, stages) => ({
        ...stage,
        end_step: index === stages.length - 1 ? maxSteps : stage.end_step,
      })),
      amp,
      compile,
      compile_mode: 'default',
      resume: 'auto',
      seed: 20260713,
      device: 'auto',
    },
  });

  const startRun = (prepareOnly: boolean) => runOperation(
    prepareOnly ? 'Preparing MarketHybrid tensors…' : 'Starting MarketHybrid training…',
    async () => {
      const run = await api<MarketHybridRun>('/api/markethybrid/runs', {
        method: 'POST',
        body: JSON.stringify(requestPayload(prepareOnly)),
      });
      setSelectedRunId(run.run_id);
      await refresh();
    },
  );

  const stopRun = (runId: string) => runOperation('Stopping MarketHybrid worker…', async () => {
    await api(`/api/markethybrid/runs/${runId}/stop`, { method: 'POST' });
    await refresh();
  });

  const registerRun = (run: MarketHybridRun) => runOperation('Registering model as indicator…', async () => {
    await api(`/api/markethybrid/runs/${run.run_id}/register`, {
      method: 'POST',
      body: JSON.stringify({ display_name: run.name, checkpoint: 'best_hybrid', primary_horizon_seconds: 300 }),
    });
    await Promise.all([refresh(), onModelRegistered()]);
  });

  const unregisterModel = (modelId: string) => runOperation('Unregistering model…', async () => {
    await api(`/api/markethybrid/models/${modelId}`, { method: 'DELETE' });
    await Promise.all([refresh(), onModelRegistered()]);
  });

  const choosePreset = (value: string) => {
    setPreset(value);
    const next = capabilities?.model_presets[value];
    if (next) setModel(next);
  };

  return (
    <main className="marketlm-workspace">
      <section className="marketlm-hero">
        <div>
          <div className="eyebrow">MARKETLM · JEPA · ACTIONABILITY · POLICY LEARNING</div>
          <h2>Train MarketHybrid</h2>
          <p>Train MarketLM forecasting together with JEPA future-latent prediction and deployable position, confidence, actionability, and execution-intent heads. Registered checkpoints work as chart indicators or learned-policy strategies.</p>
        </div>
        <div className="marketlm-runtime">
          <span className={`status-dot ${capabilities?.torch_available ? 'good' : 'warn'}`} />
          <strong>{capabilities?.torch_available ? `PyTorch ${capabilities.torch_version}` : 'PyTorch missing'}</strong>
          <small>{capabilities?.cuda_available ? `${capabilities.gpu_name} · CUDA ${capabilities.cuda_version}` : 'CPU training / preparation only'}</small>
        </div>
      </section>

      {error && <div className="alert error"><strong>MarketHybrid action failed</strong><span>{error}</span></div>}
      {busy && <div className="progress"><span className="spinner" />{busy}</div>}

      <div className="marketlm-grid">
        <section className="marketlm-config">
          <div className="section-heading"><h2>Dataset & labels</h2><span>{dataset?.prepared ? `${dataset.row_count?.toLocaleString() ?? '—'} rows ready` : 'Prepare canonical Parquet first'}</span></div>
          <div className="two-col">
            <Field label="Run name"><input value={name} onChange={(event) => setName(event.target.value)} /></Field>
            <Field label="Interval">
              <select value={timeframeSeconds} onChange={(event) => setTimeframeSeconds(Number(event.target.value))}>
                {(capabilities?.timeframes_seconds ?? [1, 5, 15, 60, 300, 900, 3600]).map((seconds) => <option key={seconds} value={seconds}>{formatTimeframe(seconds)}</option>)}
              </select>
            </Field>
            <Field label="Start (empty = beginning)"><input type="datetime-local" value={startDate} onChange={(event) => setStartDate(event.target.value)} /></Field>
            <Field label="End (empty = end)"><input type="datetime-local" value={endDate} onChange={(event) => setEndDate(event.target.value)} /></Field>
          </div>
          <Field label="Forecast horizons in seconds" hint="Comma-separated, sorted multiples of the selected interval."><input value={horizons} onChange={(event) => setHorizons(event.target.value)} /></Field>
          <div className="parameter-grid">
            <Field label="Patch size"><input type="number" min={1} value={patchSize} onChange={(event) => setPatchSize(Number(event.target.value))} /></Field>
            <Field label="Context patches"><input type="number" min={2} value={contextPatches} onChange={(event) => setContextPatches(Number(event.target.value))} /></Field>
            <Field label="Train fraction"><input type="number" min={0.1} max={0.95} step={0.01} value={trainFraction} onChange={(event) => setTrainFraction(Number(event.target.value))} /></Field>
            <Field label="Validation fraction"><input type="number" min={0.01} max={0.4} step={0.01} value={validationFraction} onChange={(event) => setValidationFraction(Number(event.target.value))} /></Field>
            <Field label="Direction cost threshold (bps)"><input type="number" min={0} step={0.1} value={costThresholdBps} onChange={(event) => setCostThresholdBps(Number(event.target.value))} /></Field>
          </div>
          <div className="context-summary"><span>Context</span><strong>{(patchSize * contextPatches).toLocaleString()} bars · {formatNumber(patchSize * contextPatches * timeframeSeconds / 3600, 2)} hours</strong></div>

          <div className="section-heading nested"><h2>Indicator features</h2><span>{selectedIndicators.length} selected</span></div>
          <div className="indicator-selector">
            {capabilities?.indicators.map((indicator) => {
              const state = indicatorState[indicator.key];
              if (!state) return null;
              return (
                <article className={`indicator-option ${state.enabled ? 'selected' : ''}`} key={indicator.key}>
                  <label className="toggle-field">
                    <input type="checkbox" checked={state.enabled} onChange={(event) => setIndicatorState((current) => ({ ...current, [indicator.key]: { ...state, enabled: event.target.checked } }))} />
                    <span>{indicator.name}</span>
                  </label>
                  <p>{indicator.description}</p>
                  {state.enabled && <div className="indicator-parameters">{Object.entries(indicator.parameters).map(([key, definition]) => (
                    <Field key={key} label={key.replaceAll('_', ' ')}>
                      <input type="number" min={definition.minimum} max={definition.maximum} step={definition.type === 'integer' ? 1 : 'any'} value={state.parameters[key]} onChange={(event) => setIndicatorState((current) => ({
                        ...current,
                        [indicator.key]: {
                          ...state,
                          parameters: { ...state.parameters, [key]: Number(event.target.value) },
                        },
                      }))} />
                    </Field>
                  ))}</div>}
                  <small>{indicator.outputs.join(' · ')}</small>
                </article>
              );
            })}
          </div>

          <div className="section-heading nested"><h2>Transformer</h2><span>Patch-based causal decoder</span></div>
          <Field label="Preset"><select value={preset} onChange={(event) => choosePreset(event.target.value)}>{Object.keys(capabilities?.model_presets ?? {}).map((key) => <option value={key} key={key}>{key}</option>)}</select></Field>
          <div className="parameter-grid">
            <Field label="d_model"><input type="number" value={model.d_model} onChange={(event) => setModel({ ...model, d_model: Number(event.target.value) })} /></Field>
            <Field label="Layers"><input type="number" value={model.n_layers} onChange={(event) => setModel({ ...model, n_layers: Number(event.target.value) })} /></Field>
            <Field label="Heads"><input type="number" value={model.n_heads} onChange={(event) => setModel({ ...model, n_heads: Number(event.target.value) })} /></Field>
            <Field label="MLP hidden"><input type="number" value={model.mlp_hidden} onChange={(event) => setModel({ ...model, mlp_hidden: Number(event.target.value) })} /></Field>
            <Field label="Dropout"><input type="number" min={0} max={0.9} step={0.01} value={model.dropout} onChange={(event) => setModel({ ...model, dropout: Number(event.target.value) })} /></Field>
            <label className="toggle-field compact"><input type="checkbox" checked={model.activation_checkpointing} onChange={(event) => setModel({ ...model, activation_checkpointing: event.target.checked })} /><span>Activation checkpointing</span></label>
          </div>

          <div className="section-heading nested"><h2>JEPA & policy</h2><span>Future latent teacher + deployable heads</span></div>
          <Field label="JEPA target patch offsets" hint="Positive patch offsets. Largest offset × patch size must fit inside the largest forecast horizon."><input value={jepaOffsets} onChange={(event) => setJepaOffsets(event.target.value)} /></Field>
          <div className="parameter-grid">
            <Field label="Predictor d_model"><input type="number" value={predictor.d_model} onChange={(event) => setPredictor({ ...predictor, d_model: Number(event.target.value) })} /></Field>
            <Field label="Predictor layers"><input type="number" value={predictor.n_layers} onChange={(event) => setPredictor({ ...predictor, n_layers: Number(event.target.value) })} /></Field>
            <Field label="Predictor heads"><input type="number" value={predictor.n_heads} onChange={(event) => setPredictor({ ...predictor, n_heads: Number(event.target.value) })} /></Field>
            <Field label="Predictor MLP"><input type="number" value={predictor.mlp_hidden} onChange={(event) => setPredictor({ ...predictor, mlp_hidden: Number(event.target.value) })} /></Field>
            <Field label="EMA start"><input type="number" min={0} max={0.99999} step={0.0001} value={emaStart} onChange={(event) => setEmaStart(Number(event.target.value))} /></Field>
            <Field label="EMA end"><input type="number" min={0} max={0.999999} step={0.0001} value={emaEnd} onChange={(event) => setEmaEnd(Number(event.target.value))} /></Field>
            <Field label="Policy position scale (bps)"><input type="number" min={0.1} step={0.1} value={positionScaleBps} onChange={(event) => setPositionScaleBps(Number(event.target.value))} /></Field>
            <Field label="Confidence temperature"><input type="number" min={0.1} step={0.1} value={confidenceTemperature} onChange={(event) => setConfidenceTemperature(Number(event.target.value))} /></Field>
            <Field label="Latent loss"><input type="number" min={0} step={0.01} value={latentLossWeight} onChange={(event) => setLatentLossWeight(Number(event.target.value))} /></Field>
            <Field label="Variance loss"><input type="number" min={0} step={0.001} value={varianceLossWeight} onChange={(event) => setVarianceLossWeight(Number(event.target.value))} /></Field>
            <Field label="Covariance loss"><input type="number" min={0} step={0.001} value={covarianceLossWeight} onChange={(event) => setCovarianceLossWeight(Number(event.target.value))} /></Field>
            <Field label="JEPA objective weight"><input type="number" min={0} step={0.05} value={jepaLossWeight} onChange={(event) => setJepaLossWeight(Number(event.target.value))} /></Field>
            <Field label="Actionable loss weight"><input type="number" min={0} step={0.05} value={actionableLossWeight} onChange={(event) => setActionableLossWeight(Number(event.target.value))} /></Field>
            <Field label="Policy position weight"><input type="number" min={0} step={0.05} value={policyPositionLossWeight} onChange={(event) => setPolicyPositionLossWeight(Number(event.target.value))} /></Field>
            <Field label="Policy confidence weight"><input type="number" min={0} step={0.05} value={policyConfidenceLossWeight} onChange={(event) => setPolicyConfidenceLossWeight(Number(event.target.value))} /></Field>
            <Field label="Policy intent weight"><input type="number" min={0} step={0.05} value={policyIntentLossWeight} onChange={(event) => setPolicyIntentLossWeight(Number(event.target.value))} /></Field>
          </div>

          <div className="section-heading nested"><h2>Training plan</h2><span>Staged optimization</span></div>
          <div className="indicator-selector">
            <article className="indicator-option selected"><strong>1. Representation pretraining</strong><p>Steps 0–8,000 · policy heads frozen · JEPA and forecasting emphasized.</p></article>
            <article className="indicator-option selected"><strong>2. Joint training</strong><p>Steps 8,001–24,000 · all deployable heads train with the encoder.</p></article>
            <article className="indicator-option selected"><strong>3. Policy fine-tuning</strong><p>Steps 24,001–40,000 · reduced encoder LR · JEPA predictor frozen · hybrid-score early stopping.</p></article>
          </div>

          <div className="section-heading nested"><h2>Training</h2><span>RTX 4060 Ti defaults</span></div>
          <div className="parameter-grid">
            <Field label="Steps"><input type="number" value={maxSteps} onChange={(event) => setMaxSteps(Number(event.target.value))} /></Field>
            <Field label="Batch"><input type="number" value={batchSize} onChange={(event) => setBatchSize(Number(event.target.value))} /></Field>
            <Field label="Gradient accumulation"><input type="number" value={gradientAccumulation} onChange={(event) => setGradientAccumulation(Number(event.target.value))} /></Field>
            <Field label="Learning rate"><input type="number" step="any" value={learningRate} onChange={(event) => setLearningRate(Number(event.target.value))} /></Field>
            <Field label="Minimum LR"><input type="number" step="any" value={minLearningRate} onChange={(event) => setMinLearningRate(Number(event.target.value))} /></Field>
            <Field label="Warmup steps"><input type="number" value={warmupSteps} onChange={(event) => setWarmupSteps(Number(event.target.value))} /></Field>
            <Field label="Validation every"><input type="number" value={validationInterval} onChange={(event) => setValidationInterval(Number(event.target.value))} /></Field>
            <Field label="Checkpoint every"><input type="number" value={checkpointInterval} onChange={(event) => setCheckpointInterval(Number(event.target.value))} /></Field>
            <Field label="Data workers" hint="Use 0 on Windows if multiprocessing is unstable."><input type="number" min={0} max={32} value={numWorkers} onChange={(event) => setNumWorkers(Number(event.target.value))} /></Field>
            <Field label="AMP"><select value={amp} onChange={(event) => setAmp(event.target.value as typeof amp)}><option value="auto">Auto</option><option value="bf16">BF16</option><option value="fp16">FP16</option><option value="off">Off</option></select></Field>
            <Field label="torch.compile"><select value={compile} onChange={(event) => setCompile(event.target.value as typeof compile)}><option value="off">Off</option><option value="on">On</option></select></Field>
          </div>
          <div className="button-row marketlm-actions">
            <button className="secondary" disabled={Boolean(busy) || !capabilities} onClick={() => capabilities && applyDefaults(capabilities.default_config)}>Reset to optimized 15-second defaults</button>
            <button className="secondary" disabled={Boolean(busy) || !dataset?.prepared} onClick={() => startRun(true)}>Prepare tensors only</button>
            <button className="primary" disabled={Boolean(busy) || !dataset?.prepared || !capabilities?.torch_available} onClick={() => startRun(false)}>Start / resume training</button>
          </div>
        </section>

        <section className="marketlm-runs">
          <div className="section-heading"><h2>Runs</h2><button className="secondary compact-button" onClick={() => refresh()} disabled={Boolean(busy)}>Refresh</button></div>
          <div className="run-list">
            {runs.length === 0 && <div className="empty-inline">No MarketHybrid runs yet.</div>}
            {runs.map((run) => (
              <button className={`run-card ${selectedRunId === run.run_id ? 'selected' : ''}`} key={run.run_id} onClick={() => setSelectedRunId(run.run_id)}>
                <div><strong>{run.name}</strong><span className={`run-state ${run.state}`}>{run.state}</span></div>
                <small>{run.kind} · {run.step.toLocaleString()}/{run.max_steps.toLocaleString()}</small>
                <div className="run-progress"><span style={{ width: `${Math.max(0, Math.min(100, run.progress * 100))}%` }} /></div>
                <p>{run.message}</p>
              </button>
            ))}
          </div>

          {selectedRun && <article className="run-detail">
            <div className="section-heading"><h2>{selectedRun.name}</h2><span>{selectedRun.run_id}</span></div>
            <div className="metrics-grid compact-metrics">
              <div className="metric"><span>State</span><strong>{selectedRun.state}</strong></div>
              <div className="metric"><span>Progress</span><strong>{(selectedRun.progress * 100).toFixed(1)}%</strong></div>
              <div className="metric"><span>Stage</span><strong>{selectedRun.stage?.name ?? '—'}</strong></div>
              <div className="metric"><span>Stage progress</span><strong>{selectedRun.stage ? `${(selectedRun.stage.progress * 100).toFixed(1)}%` : '—'}</strong></div>
              <div className="metric"><span>Loss</span><strong>{formatNumber(selectedRun.metrics?.loss ?? selectedRun.metrics?.total)}</strong></div>
              <div className="metric"><span>Direction accuracy</span><strong>{typeof selectedRun.metrics?.direction_accuracy === 'number' ? `${(selectedRun.metrics.direction_accuracy * 100).toFixed(2)}%` : '—'}</strong></div>
              <div className="metric"><span>Patch tokens/s</span><strong>{formatNumber(selectedRun.metrics?.patch_tokens_per_second, 0)}</strong></div>
              <div className="metric"><span>Learning rate</span><strong>{formatNumber(selectedRun.metrics?.learning_rate, 8)}</strong></div>
            </div>
            <p className="muted">{selectedRun.message}</p>
            {selectedRun.error && <div className="alert error"><strong>Worker error</strong><span>{selectedRun.error}</span></div>}
            <div className="button-row">
              {['queued', 'preparing', 'training', 'stopping'].includes(selectedRun.state) && <button className="danger" onClick={() => stopRun(selectedRun.run_id)}>Stop</button>}
              {selectedRun.state === 'completed' && !selectedRun.registered && selectedRun.kind === 'train' && <button className="primary" onClick={() => registerRun(selectedRun)}>Register as indicator</button>}
              {selectedRun.registered && <span className="registered-badge">Registered as strategy indicator</span>}
            </div>
            {selectedRun.log_tail && <pre className="worker-log">{selectedRun.log_tail}</pre>}
          </article>}

          <div className="section-heading nested"><h2>Registered indicators</h2><span>{models.length}</span></div>
          <div className="registered-models">
            {models.length === 0 && <div className="empty-inline">Complete a training run and register its best checkpoint.</div>}
            {models.map((registered) => (
              <article className="registered-model" key={registered.model_id}>
                <div><strong>{registered.display_name}</strong><span>{registered.available ? 'ready' : 'missing checkpoint'}</span></div>
                <p>{registered.description}</p>
                <small>{formatTimeframe(registered.timeframe_seconds)} · horizons {registered.horizons_seconds.join(', ')}s · primary {registered.primary_horizon_seconds}s</small>
                <button className="danger subtle" onClick={() => unregisterModel(registered.model_id)}>Unregister</button>
              </article>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}

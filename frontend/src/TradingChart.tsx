import { useEffect, useMemo, useRef } from 'react';
import {
  CandlestickSeries,
  ColorType,
  createChart,
  createSeriesMarkers,
  HistogramSeries,
  LineSeries,
  type IChartApi,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type LogicalRangeChangeEventHandler,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from 'lightweight-charts';
import type { Bar, EquityPoint, Fill, IndicatorSpec, PlotPoint } from './types';

interface TradingChartProps {
  bars: Bar[];
  fills: Fill[];
  plots: PlotPoint[];
  indicatorSpecs: IndicatorSpec[];
  equity: EquityPoint[];
}

const chartOptions = {
  layout: {
    background: { type: ColorType.Solid, color: '#0b0f18' },
    textColor: '#9ca8bc',
    panes: { separatorColor: '#20293a', separatorHoverColor: '#34445f' },
  },
  grid: {
    vertLines: { color: '#151c29' },
    horzLines: { color: '#151c29' },
  },
  crosshair: {
    vertLine: { color: '#53627a', labelBackgroundColor: '#263348' },
    horzLine: { color: '#53627a', labelBackgroundColor: '#263348' },
  },
  rightPriceScale: { borderColor: '#263044' },
  timeScale: {
    borderColor: '#263044',
    timeVisible: true,
    secondsVisible: false,
    rightOffset: 4,
  },
};

function asTime(timestamp: number): UTCTimestamp {
  return timestamp as UTCTimestamp;
}

function useResize(chartRef: React.RefObject<IChartApi | null>, containerRef: React.RefObject<HTMLDivElement | null>) {
  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver(([entry]) => {
      chartRef.current?.applyOptions({ width: Math.floor(entry.contentRect.width) });
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, [chartRef, containerRef]);
}

export default function TradingChart({ bars, fills, plots, indicatorSpecs, equity }: TradingChartProps) {
  const priceContainer = useRef<HTMLDivElement>(null);
  const indicatorContainer = useRef<HTMLDivElement>(null);
  const equityContainer = useRef<HTMLDivElement>(null);
  const priceChart = useRef<IChartApi | null>(null);
  const indicatorChart = useRef<IChartApi | null>(null);
  const equityChart = useRef<IChartApi | null>(null);
  const candleSeries = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeries = useRef<ISeriesApi<'Histogram'> | null>(null);
  const priceLines = useRef(new Map<string, ISeriesApi<'Line'>>());
  const indicatorLines = useRef(new Map<string, ISeriesApi<'Line'>>());
  const equitySeries = useRef<ISeriesApi<'Line'> | null>(null);
  const markerPlugin = useRef<ISeriesMarkersPluginApi<Time> | null>(null);

  const priceSpecs = useMemo(() => indicatorSpecs.filter((spec) => spec.pane === 'price'), [indicatorSpecs]);
  const lowerSpecs = useMemo(() => indicatorSpecs.filter((spec) => spec.pane === 'indicator'), [indicatorSpecs]);
  const specSignature = useMemo(() => JSON.stringify(indicatorSpecs), [indicatorSpecs]);

  useEffect(() => {
    if (!priceContainer.current) return;
    const chart = createChart(priceContainer.current, { ...chartOptions, height: 500 });
    priceChart.current = chart;
    candleSeries.current = chart.addSeries(CandlestickSeries, {
      priceScaleId: 'right',
      upColor: '#21c98b',
      downColor: '#f15b6c',
      borderVisible: false,
      wickUpColor: '#21c98b',
      wickDownColor: '#f15b6c',
      priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
    });
    volumeSeries.current = chart.addSeries(HistogramSeries, {
      priceScaleId: '',
      priceFormat: { type: 'volume' },
      lastValueVisible: false,
      priceLineVisible: false,
    });
    volumeSeries.current.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
    markerPlugin.current = createSeriesMarkers(candleSeries.current, []);

    for (const spec of priceSpecs) {
      priceLines.current.set(spec.key, chart.addSeries(LineSeries, {
        title: spec.label,
        priceScaleId: 'right',
        priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
        lineWidth: Math.min(4, Math.max(1, spec.line_width)) as 1 | 2 | 3 | 4,
        priceLineVisible: false,
        lastValueVisible: true,
      }));
    }

    return () => {
      markerPlugin.current = null;
      priceLines.current.clear();
      chart.remove();
      priceChart.current = null;
      candleSeries.current = null;
      volumeSeries.current = null;
    };
  }, [specSignature]); // Rebuild only when the strategy's plot contract changes.

  useEffect(() => {
    if (!indicatorContainer.current || lowerSpecs.length === 0) {
      indicatorChart.current?.remove();
      indicatorChart.current = null;
      indicatorLines.current.clear();
      return;
    }
    const chart = createChart(indicatorContainer.current, { ...chartOptions, height: 190 });
    indicatorChart.current = chart;
    for (const spec of lowerSpecs) {
      indicatorLines.current.set(spec.key, chart.addSeries(LineSeries, {
        title: spec.label,
        lineWidth: Math.min(4, Math.max(1, spec.line_width)) as 1 | 2 | 3 | 4,
        priceLineVisible: false,
      }));
    }
    return () => {
      indicatorLines.current.clear();
      chart.remove();
      indicatorChart.current = null;
    };
  }, [specSignature]);

  useEffect(() => {
    if (!equityContainer.current) return;
    const chart = createChart(equityContainer.current, { ...chartOptions, height: 190 });
    equityChart.current = chart;
    equitySeries.current = chart.addSeries(LineSeries, {
      title: 'Equity',
      lineWidth: 2,
      priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
      priceLineVisible: false,
    });
    return () => {
      chart.remove();
      equityChart.current = null;
      equitySeries.current = null;
    };
  }, []);

  useResize(priceChart, priceContainer);
  useResize(indicatorChart, indicatorContainer);
  useResize(equityChart, equityContainer);

  useEffect(() => {
    const candles = [...bars]
      .sort((a, b) => a.timestamp - b.timestamp)
      .map((bar) => ({
        time: asTime(bar.timestamp), open: bar.open, high: bar.high, low: bar.low, close: bar.close,
      }));
    candleSeries.current?.setData(candles);
    volumeSeries.current?.setData([...bars]
      .sort((a, b) => a.timestamp - b.timestamp)
      .map((bar) => ({
        time: asTime(bar.timestamp),
        value: bar.volume,
        color: bar.close >= bar.open ? 'rgba(33, 201, 139, 0.35)' : 'rgba(241, 91, 108, 0.35)',
      })));

    const timeframe = bars[0]?.timeframe_seconds ?? 60;
    const barTimes = new Set(bars.map((bar) => bar.timestamp));
    const markers: SeriesMarker<Time>[] = [...fills]
      .sort((a, b) => a.timestamp - b.timestamp)
      .map((fill) => ({
        time: asTime(barTimes.has(fill.timestamp)
          ? fill.timestamp
          : fill.timestamp - (fill.timestamp % timeframe)),
        position: fill.side === 'buy' ? 'belowBar' : 'aboveBar',
        shape: fill.side === 'buy' ? 'arrowUp' : 'arrowDown',
        color: fill.side === 'buy' ? '#21c98b' : '#f15b6c',
        text: `${fill.side.toUpperCase()} ${fill.quantity.toFixed(5)} · ${fill.reason}`,
      }));
    markerPlugin.current?.setMarkers(markers);
    priceChart.current?.timeScale().applyOptions({ secondsVisible: timeframe < 60 });
  }, [bars, fills, specSignature]);

  useEffect(() => {
    for (const spec of indicatorSpecs) {
      const series = spec.pane === 'price' ? priceLines.current.get(spec.key) : indicatorLines.current.get(spec.key);
      if (!series) continue;
      const offset = spec.time_offset_seconds ?? 0;
      const data = plots
        .map((point) => ({ time: asTime(point.timestamp + offset), value: point.values[spec.key] }))
        .filter((point): point is { time: UTCTimestamp; value: number } => typeof point.value === 'number' && Number.isFinite(point.value));
      series.setData(data);
    }
  }, [plots, indicatorSpecs, specSignature]);

  useEffect(() => {
    equitySeries.current?.setData(equity.map((point) => ({ time: asTime(point.timestamp), value: point.equity })));
  }, [equity]);

  useEffect(() => {
    const source = priceChart.current?.timeScale();
    if (!source) return;
    const sync: LogicalRangeChangeEventHandler = (range) => {
      if (range) {
        indicatorChart.current?.timeScale().setVisibleLogicalRange(range);
        equityChart.current?.timeScale().setVisibleLogicalRange(range);
      }
    };
    source.subscribeVisibleLogicalRangeChange(sync);
    return () => source.unsubscribeVisibleLogicalRangeChange(sync);
  }, [specSignature]);

  return (
    <div className="chart-stack">
      <div className="chart-caption"><span>BTC/USD</span><span>Candles · volume · signals</span></div>
      <div ref={priceContainer} className="chart-surface" />
      {lowerSpecs.length > 0 && (
        <>
          <div className="chart-caption"><span>Indicators</span><span>{lowerSpecs.map((item) => item.label).join(' · ')}</span></div>
          <div ref={indicatorContainer} className="chart-surface compact" />
        </>
      )}
      <div className="chart-caption"><span>Equity</span><span>Mark-to-market portfolio value</span></div>
      <div ref={equityContainer} className="chart-surface compact" />
    </div>
  );
}

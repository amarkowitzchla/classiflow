import { useState, useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Area,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { clsx } from 'clsx';
import type { PlotCurve } from '../../types/plots';
import { CURVE_COLORS, AGGREGATE_COLORS } from '../../types/plots';

interface RocPrChartProps {
  data: PlotCurve;
  height?: number;
  showLegend?: boolean;
  showGrid?: boolean;
  showDiagonal?: boolean;
  className?: string;
}

interface ChartDataPoint {
  x: number;
  [key: string]: number;
}

function getCurveColor(label: string, index: number): string {
  const lowerLabel = label.toLowerCase();
  if (lowerLabel in AGGREGATE_COLORS) {
    return AGGREGATE_COLORS[lowerLabel as keyof typeof AGGREGATE_COLORS];
  }
  return CURVE_COLORS[index % CURVE_COLORS.length];
}

function formatTooltipValue(value: number): string {
  return value.toFixed(4);
}

export function RocPrChart({
  data,
  height = 400,
  showLegend = true,
  showGrid = true,
  showDiagonal = true,
  className,
}: RocPrChartProps) {
  const [hiddenCurves, setHiddenCurves] = useState<Set<string>>(new Set());

  const isRoc = data.plot_type === 'roc';
  const xLabel = isRoc ? 'False Positive Rate' : 'Recall';
  const yLabel = isRoc ? 'True Positive Rate' : 'Precision';

  // Convert curve data to Recharts format
  const chartData = useMemo(() => {
    // Create a unified x-axis by collecting all unique x values
    const allX = new Set<number>();
    data.curves.forEach((curve) => {
      curve.x.forEach((x) => allX.add(x));
    });
    const sortedX = Array.from(allX).sort((a, b) => a - b);

    // Build chart data points
    const points: ChartDataPoint[] = sortedX.map((x) => {
      const point: ChartDataPoint = { x };

      data.curves.forEach((curve) => {
        // Find the closest y value for this x
        const idx = curve.x.findIndex((cx) => cx >= x);
        if (idx >= 0 && idx < curve.y.length) {
          // Linear interpolation for smoother curves
          if (idx > 0 && curve.x[idx] !== x) {
            const x0 = curve.x[idx - 1];
            const x1 = curve.x[idx];
            const y0 = curve.y[idx - 1];
            const y1 = curve.y[idx];
            const t = (x - x0) / (x1 - x0);
            point[curve.label] = y0 + t * (y1 - y0);
          } else {
            point[curve.label] = curve.y[idx];
          }
        }
      });

      // Add std band if available
      if (data.std_band) {
        const bandIdx = data.std_band.x.findIndex((bx) => bx >= x);
        if (bandIdx >= 0) {
          point['_upper'] = data.std_band.y_upper[bandIdx];
          point['_lower'] = data.std_band.y_lower[bandIdx];
        }
      }

      return point;
    });

    return points;
  }, [data]);

  // Get summary metric (AUC for ROC, AP for PR)
  const summaryMetric = isRoc ? data.summary.auc : data.summary.ap;
  const metricLabel = isRoc ? 'AUC' : 'AP';

  const toggleCurve = (label: string) => {
    setHiddenCurves((prev) => {
      const next = new Set(prev);
      if (next.has(label)) {
        next.delete(label);
      } else {
        next.add(label);
      }
      return next;
    });
  };

  const visibleCurves = data.curves.filter((c) => !hiddenCurves.has(c.label));

  return (
    <div className={clsx('w-full', className)}>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart
          data={chartData}
          margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
        >
          {showGrid && <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />}
          <XAxis
            dataKey="x"
            type="number"
            domain={[0, 1]}
            tickFormatter={(v) => v.toFixed(1)}
            label={{ value: xLabel, position: 'bottom', offset: 0 }}
            stroke="#6b7280"
            fontSize={12}
          />
          <YAxis
            type="number"
            domain={[0, 1]}
            tickFormatter={(v) => v.toFixed(1)}
            label={{ value: yLabel, angle: -90, position: 'insideLeft', offset: 10 }}
            stroke="#6b7280"
            fontSize={12}
          />
          <Tooltip
            formatter={(value: number, name: string) => [
              formatTooltipValue(value),
              name === '_upper' || name === '_lower' ? 'Std Band' : name,
            ]}
            labelFormatter={(x: number) => `${xLabel}: ${formatTooltipValue(x)}`}
            contentStyle={{
              backgroundColor: 'white',
              border: '1px solid #e5e7eb',
              borderRadius: '6px',
              padding: '8px 12px',
            }}
          />

          {/* Std deviation band */}
          {data.std_band && (
            <Area
              type="monotone"
              dataKey="_upper"
              stroke="none"
              fill="#3b82f6"
              fillOpacity={0.15}
            />
          )}

          {/* Diagonal reference line for ROC */}
          {showDiagonal && isRoc && (
            <ReferenceLine
              segment={[
                { x: 0, y: 0 },
                { x: 1, y: 1 },
              ]}
              stroke="#9ca3af"
              strokeDasharray="5 5"
              strokeWidth={1}
            />
          )}

          {/* Curve lines */}
          {visibleCurves.map((curve, idx) => {
            const color = getCurveColor(curve.label, idx);
            const isAggregate =
              curve.label.toLowerCase() === 'mean' ||
              curve.label.toLowerCase() === 'micro' ||
              curve.label.toLowerCase() === 'macro';

            return (
              <Line
                key={curve.label}
                type="monotone"
                dataKey={curve.label}
                stroke={color}
                strokeWidth={isAggregate ? 2.5 : 1.5}
                strokeDasharray={isAggregate ? '0' : undefined}
                dot={false}
                activeDot={{ r: 4, fill: color }}
              />
            );
          })}

          {showLegend && (
            <Legend
              onClick={(e) => toggleCurve(e.dataKey as string)}
              wrapperStyle={{ cursor: 'pointer' }}
            />
          )}
        </LineChart>
      </ResponsiveContainer>

      {/* Summary metrics */}
      {summaryMetric && Object.keys(summaryMetric).length > 0 && (
        <div className="mt-4 flex flex-wrap gap-2 justify-center">
          {Object.entries(summaryMetric).map(([key, value]) => (
            <span
              key={key}
              className="px-3 py-1 bg-gray-100 rounded-full text-sm"
            >
              <span className="font-medium">{key}</span>
              <span className="text-gray-600 ml-1">
                {metricLabel}: {typeof value === 'number' ? value.toFixed(3) : '-'}
              </span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export default RocPrChart;

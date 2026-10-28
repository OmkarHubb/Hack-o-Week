import React, { useState } from 'react';

interface MetricPoint {
  epoch: number;
  [key: string]: number;
}

interface SeriesConfig {
  key: string;
  name: string;
  color: string;
  strokeDash?: string;
}

interface TrainingCurvesChartProps {
  title: string;
  data: MetricPoint[];
  series: SeriesConfig[];
  yLabel?: string;
  height?: number;
}

export const TrainingCurvesChart: React.FC<TrainingCurvesChartProps> = ({
  title,
  data,
  series,
  yLabel = 'Loss',
  height = 240,
}) => {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  if (!data || data.length === 0) {
    return (
      <div className="glass-panel" style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)' }}>
        No training history available yet. Run training to generate curves.
      </div>
    );
  }

  const padding = { top: 20, right: 30, bottom: 35, left: 50 };
  const width = 600;

  // Calculate min and max
  let allValues: number[] = [];
  data.forEach((d) => {
    series.forEach((s) => {
      if (typeof d[s.key] === 'number' && !isNaN(d[s.key])) {
        allValues.push(d[s.key]);
      }
    });
  });

  const minY = Math.min(0, ...allValues);
  const maxY = Math.max(1, Math.max(...allValues) * 1.15);
  const minX = data[0].epoch;
  const maxX = data[data.length - 1].epoch || 1;

  const scaleX = (x: number) => {
    if (maxX === minX) return padding.left;
    return padding.left + ((x - minX) / (maxX - minX)) * (width - padding.left - padding.right);
  };

  const scaleY = (y: number) => {
    if (maxY === minY) return height - padding.bottom;
    return height - padding.bottom - ((y - minY) / (maxY - minY)) * (height - padding.top - padding.bottom);
  };

  // Build path strings for each series
  const paths = series.map((s) => {
    const points = data
      .filter((d) => typeof d[s.key] === 'number')
      .map((d) => `${scaleX(d.epoch)},${scaleY(d[s.key])}`)
      .join(' L ');
    return { ...s, pathStr: points ? `M ${points}` : '' };
  });

  return (
    <div className="glass-panel" style={{ padding: '16px', position: 'relative' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <h4 style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)' }}>{title}</h4>
        <div style={{ display: 'flex', gap: '14px', flexWrap: 'wrap' }}>
          {series.map((s) => (
            <div key={s.key} style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.8rem' }}>
              <span
                style={{
                  display: 'inline-block',
                  width: '12px',
                  height: '3px',
                  backgroundColor: s.color,
                  borderRadius: '2px',
                }}
              />
              <span style={{ color: 'var(--text-secondary)' }}>{s.name}</span>
            </div>
          ))}
        </div>
      </div>

      <div style={{ width: '100%', overflowX: 'auto' }}>
        <svg
          viewBox={`0 0 ${width} ${height}`}
          style={{ width: '100%', height: 'auto', maxHeight: `${height}px`, overflow: 'visible' }}
        >
          {/* Y Gridlines */}
          {[0, 0.25, 0.5, 0.75, 1.0].map((ratio) => {
            const val = minY + ratio * (maxY - minY);
            const y = scaleY(val);
            return (
              <g key={ratio}>
                <line
                  x1={padding.left}
                  y1={y}
                  x2={width - padding.right}
                  y2={y}
                  stroke="rgba(255, 255, 255, 0.08)"
                  strokeDasharray="3 3"
                />
                <text
                  x={padding.left - 8}
                  y={y + 4}
                  fill="var(--text-muted)"
                  fontSize="10"
                  textAnchor="end"
                  fontFamily="Fira Code"
                >
                  {val.toFixed(2)}
                </text>
              </g>
            );
          })}

          {/* X Grid & Labels */}
          {data
            .filter((_, idx) => idx % Math.max(1, Math.floor(data.length / 6)) === 0 || idx === data.length - 1)
            .map((d) => {
              const x = scaleX(d.epoch);
              return (
                <g key={d.epoch}>
                  <line
                    x1={x}
                    y1={padding.top}
                    x2={x}
                    y2={height - padding.bottom}
                    stroke="rgba(255, 255, 255, 0.05)"
                  />
                  <text
                    x={x}
                    y={height - padding.bottom + 18}
                    fill="var(--text-muted)"
                    fontSize="10"
                    textAnchor="middle"
                    fontFamily="Fira Code"
                  >
                    Ep {d.epoch}
                  </text>
                </g>
              );
            })}

          {/* Lines */}
          {paths.map((p) => (
            <path
              key={p.key}
              d={p.pathStr}
              fill="none"
              stroke={p.color}
              strokeWidth="2.5"
              strokeDasharray={p.strokeDash}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          ))}

          {/* Data Points */}
          {series.map((s) =>
            data.map((d, idx) => {
              const val = d[s.key];
              if (typeof val !== 'number') return null;
              const cx = scaleX(d.epoch);
              const cy = scaleY(val);
              const isHovered = hoveredIdx === idx;
              return (
                <circle
                  key={`${s.key}-${idx}`}
                  cx={cx}
                  cy={cy}
                  r={isHovered ? 5 : 2.5}
                  fill={s.color}
                  stroke="var(--bg-primary)"
                  strokeWidth="1.5"
                  style={{ cursor: 'pointer', transition: 'r 0.15s ease' }}
                  onMouseEnter={() => setHoveredIdx(idx)}
                  onMouseLeave={() => setHoveredIdx(null)}
                />
              );
            })
          )}
        </svg>
      </div>

      {/* Tooltip */}
      {hoveredIdx !== null && data[hoveredIdx] && (
        <div
          style={{
            marginTop: '8px',
            padding: '8px 12px',
            background: 'rgba(0, 0, 0, 0.75)',
            borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--border-subtle)',
            fontSize: '0.8rem',
            display: 'flex',
            gap: '16px',
            alignItems: 'center',
          }}
        >
          <span style={{ fontWeight: 600, color: 'var(--accent-cyan)' }}>Epoch {data[hoveredIdx].epoch}:</span>
          {series.map((s) => (
            <span key={s.key} style={{ color: s.color }}>
              {s.name}: <strong>{typeof data[hoveredIdx][s.key] === 'number' ? data[hoveredIdx][s.key].toFixed(4) : '-'}</strong>
            </span>
          ))}
        </div>
      )}
    </div>
  );
};

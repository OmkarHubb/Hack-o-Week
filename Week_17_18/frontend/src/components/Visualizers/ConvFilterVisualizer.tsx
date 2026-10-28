import React, { useState, useEffect } from 'react';
import { api } from '../../services/api';
import { Play, Sparkles, Sliders } from 'lucide-react';

interface ConvFilterData {
  filter_name: string;
  explanation: string;
  kernel: number[][];
  input_pattern: string;
  input_shape: number[];
  conv_shape: number[];
  relu_shape: number[];
  pool_shape: number[];
  input_matrix: number[][];
  conv_matrix: number[][];
  relu_matrix: number[][];
  pool_matrix: number[][];
  summary: {
    conv_min: number;
    conv_max: number;
    relu_zeros_pct: number;
    pool_downsample_ratio: string;
  };
}

export const ConvFilterVisualizer: React.FC = () => {
  const [filterType, setFilterType] = useState<string>('sobel_x');
  const [inputPattern, setInputPattern] = useState<string>('digit_seven');
  const [customKernel, setCustomKernel] = useState<number[][]>([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0],
  ]);
  const [data, setData] = useState<ConvFilterData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [hoveredCoord, setHoveredCoord] = useState<{ r: number; c: number } | null>(null);

  const fetchConvFilter = async () => {
    setLoading(true);
    try {
      const res = await api.runConvFilter({
        filter_type: filterType,
        custom_kernel: filterType === 'custom' ? customKernel : undefined,
        input_pattern: inputPattern,
      });
      setData(res);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchConvFilter();
  }, [filterType, inputPattern]);

  // Color generator for heatmaps
  const getCellColor = (val: number, minVal: number = 0, maxVal: number = 1, isConv: boolean = false) => {
    if (isConv) {
      // Diverging blue-white-red/amber for conv (negative vs positive gradients)
      if (val < 0) {
        const norm = Math.min(1, Math.abs(val) / (Math.abs(minVal) || 1));
        return `rgba(59, 130, 246, ${0.15 + norm * 0.8})`; // Blue
      } else {
        const norm = Math.min(1, val / (maxVal || 1));
        return `rgba(244, 63, 94, ${0.15 + norm * 0.8})`; // Rose
      }
    }
    // Standard grayscale to cyan/indigo intensity
    const norm = Math.max(0, Math.min(1, (val - minVal) / ((maxVal - minVal) || 1)));
    return `rgba(6, 182, 212, ${norm})`;
  };

  const handleKernelCellChange = (r: number, c: number, valueStr: string) => {
    const val = parseFloat(valueStr) || 0;
    const nextKernel = customKernel.map((row, ri) =>
      row.map((cell, ci) => (ri === r && ci === c ? val : cell))
    );
    setCustomKernel(nextKernel);
  };

  return (
    <div className="glass-panel" style={{ padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <div>
          <h3 style={{ fontSize: '1.15rem', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Sparkles size={18} color="var(--accent-cyan)" />
            Interactive 2D Convolution & Feature Map Inspector
          </h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
            Inspect how 3x3 kernels slide across the input image, detect features, pass through ReLU non-linearity, and downsample with Max Pooling.
          </p>
        </div>
        <button className="btn-primary" onClick={fetchConvFilter} disabled={loading}>
          <Play size={15} /> {loading ? 'Computing...' : 'Apply Filter'}
        </button>
      </div>

      {/* Controls row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px', marginBottom: '20px' }}>
        <div>
          <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '6px', display: 'block' }}>
            Input Pattern:
          </label>
          <select
            className="select-field"
            value={inputPattern}
            onChange={(e) => setInputPattern(e.target.value)}
          >
            <option value="digit_seven">Handwritten '7' Glyph</option>
            <option value="cross">Cross Pattern (+)</option>
            <option value="checkerboard">Checkerboard Grid</option>
            <option value="diagonal">Diagonal Stripe</option>
          </select>
        </div>

        <div>
          <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '6px', display: 'block' }}>
            Filter Kernel Preset:
          </label>
          <select
            className="select-field"
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
          >
            <option value="sobel_x">Sobel X (Vertical Edges)</option>
            <option value="sobel_y">Sobel Y (Horizontal Edges)</option>
            <option value="sharpen">Sharpen (Contrast Enhancer)</option>
            <option value="gaussian_blur">Gaussian Blur (Smoother)</option>
            <option value="ridge">Ridge / Laplacian (All Edges)</option>
            <option value="custom">Custom 3x3 Matrix</option>
          </select>
        </div>

        {/* 3x3 Kernel Editor */}
        <div style={{ background: 'rgba(0, 0, 0, 0.4)', padding: '10px 14px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Sliders size={13} /> 3x3 Kernel Matrix:
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '4px' }}>
            {(filterType === 'custom' ? customKernel : (data?.kernel || customKernel)).map((row, r) =>
              row.map((cell, c) => (
                <input
                  key={`${r}-${c}`}
                  type="number"
                  step="0.5"
                  disabled={filterType !== 'custom'}
                  value={cell}
                  onChange={(e) => handleKernelCellChange(r, c, e.target.value)}
                  style={{
                    width: '100%',
                    background: 'rgba(255, 255, 255, 0.05)',
                    border: '1px solid var(--border-subtle)',
                    color: 'var(--text-primary)',
                    textAlign: 'center',
                    fontSize: '0.75rem',
                    padding: '2px',
                    borderRadius: '4px',
                    fontFamily: 'Fira Code',
                  }}
                />
              ))
            )}
          </div>
        </div>
      </div>

      {data && (
        <>
          <div style={{ padding: '10px 14px', background: 'rgba(99, 102, 241, 0.08)', borderRadius: 'var(--radius-sm)', borderLeft: '3px solid var(--accent-indigo)', marginBottom: '20px', fontSize: '0.85rem' }}>
            <strong>Kernel Explanation:</strong> {data.explanation}
          </div>

          {/* Pipeline Stages Display */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px', alignItems: 'start' }}>
            {/* Step 1: Input */}
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '6px' }}>
                1. Input Image
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '8px' }}>
                Shape: 28 x 28
              </div>
              <div
                className="pixel-grid"
                style={{ gridTemplateColumns: 'repeat(28, 1fr)', width: '180px', margin: '0 auto' }}
              >
                {data.input_matrix.map((row, r) =>
                  row.map((val, c) => (
                    <div
                      key={`in-${r}-${c}`}
                      className="pixel-cell"
                      style={{ backgroundColor: getCellColor(val, 0, 1) }}
                      onMouseEnter={() => setHoveredCoord({ r, c })}
                    />
                  ))
                )}
              </div>
            </div>

            {/* Step 2: Conv2D */}
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--accent-cyan)', marginBottom: '6px' }}>
                2. Conv2D Feature Map
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '8px' }}>
                Shape: 28 x 28 (Padding=1)
              </div>
              <div
                className="pixel-grid"
                style={{ gridTemplateColumns: 'repeat(28, 1fr)', width: '180px', margin: '0 auto' }}
              >
                {data.conv_matrix.map((row, r) =>
                  row.map((val, c) => (
                    <div
                      key={`conv-${r}-${c}`}
                      className="pixel-cell"
                      style={{ backgroundColor: getCellColor(val, data.summary.conv_min, data.summary.conv_max, true) }}
                      title={`Value: ${val}`}
                    />
                  ))
                )}
              </div>
            </div>

            {/* Step 3: ReLU */}
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--accent-violet)', marginBottom: '6px' }}>
                3. Post-ReLU Activation
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '8px' }}>
                f(x) = max(0, x)
              </div>
              <div
                className="pixel-grid"
                style={{ gridTemplateColumns: 'repeat(28, 1fr)', width: '180px', margin: '0 auto' }}
              >
                {data.relu_matrix.map((row, r) =>
                  row.map((val, c) => (
                    <div
                      key={`relu-${r}-${c}`}
                      className="pixel-cell"
                      style={{ backgroundColor: getCellColor(val, 0, data.summary.conv_max) }}
                    />
                  ))
                )}
              </div>
            </div>

            {/* Step 4: MaxPool */}
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--accent-emerald)', marginBottom: '6px' }}>
                4. MaxPool2D (2x2, stride=2)
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '8px' }}>
                Shape: 14 x 14 (Downsampled)
              </div>
              <div
                className="pixel-grid"
                style={{ gridTemplateColumns: 'repeat(14, 1fr)', width: '140px', margin: '20px auto 0' }}
              >
                {data.pool_matrix.map((row, r) =>
                  row.map((val, c) => (
                    <div
                      key={`pool-${r}-${c}`}
                      className="pixel-cell"
                      style={{ backgroundColor: getCellColor(val, 0, data.summary.conv_max) }}
                    />
                  ))
                )}
              </div>
            </div>
          </div>

          {/* Quick Metrics Bar */}
          <div style={{ marginTop: '20px', display: 'flex', gap: '16px', flexWrap: 'wrap', justifyContent: 'center' }}>
            <span className="badge badge-indigo">Conv Gradient Span: [{data.summary.conv_min}, {data.summary.conv_max}]</span>
            <span className="badge badge-amber">ReLU Inactive Neurons: {data.summary.relu_zeros_pct}% zeroes</span>
            <span className="badge badge-emerald">Spatial Compression: {data.summary.pool_downsample_ratio}</span>
          </div>
        </>
      )}
    </div>
  );
};

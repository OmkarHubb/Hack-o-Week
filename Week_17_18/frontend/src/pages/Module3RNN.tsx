import React, { useState } from 'react';
import { useLab } from '../context/LabContext';
import { api } from '../services/api';
import { RnnGateVisualizer } from '../components/Visualizers/RnnGateVisualizer';
import { TrainingCurvesChart } from '../components/Visualizers/TrainingCurvesChart';
import {
  Repeat,
  Play,
  CheckCircle2,
  Cpu,
  Clock,
  TrendingDown,
  Layers
} from 'lucide-react';

export const Module3RNN: React.FC = () => {
  const { markModuleComplete, recordPracticalRun } = useLab();

  const [epochs, setEpochs] = useState<number>(20);
  const [hiddenSize, setHiddenSize] = useState<number>(32);
  const [seqLen, setSeqLen] = useState<number>(20);
  const [benchmarkResult, setBenchmarkResult] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);

  // Quiz state
  const [quizAnswers, setQuizAnswers] = useState<Record<number, number>>({});
  const [quizSubmitted, setQuizSubmitted] = useState<boolean>(false);

  const handleRunBenchmark = async () => {
    setLoading(true);
    try {
      const res = await api.compareRecurrent({
        task_type: 'sine_forecasting',
        epochs,
        hidden_size: hiddenSize,
        sequence_length: seqLen,
      });
      setBenchmarkResult(res);
      recordPracticalRun(3);
      markModuleComplete(3);
    } catch (err: any) {
      alert(err.message);
    } finally {
      setLoading(false);
    }
  };

  const quizQuestions = [
    {
      q: 'Why do Vanilla RNNs struggle with long sequences (e.g. sequences > 15-20 steps)?',
      options: [
        'They run out of computer RAM immediately',
        'Repeated backpropagation through time (BPTT) matrix multiplications cause exponential vanishing or exploding gradients',
        'They cannot handle floating point numbers',
        'They only allow binary inputs',
      ],
      correct: 1,
      explanation: 'Repeatedly multiplying by the recurrent weight matrix Whh over T steps causes gradients to decay exponentially (if singular values < 1) or explode (if > 1), losing long-range context.',
    },
    {
      q: 'What is the role of the LSTM Cell State (C_t) and Forget Gate (f_t)?',
      options: [
        'To delete all model weights during each forward pass',
        'The cell state acts as an additive memory conveyor belt, while the forget gate selectively scales past memory between 0 (erase) and 1 (keep)',
        'To calculate batch normalization statistics',
        'To compute cross-entropy loss',
      ],
      correct: 1,
      explanation: 'The cell state Ct provides a direct linear highway where updates are additive (f_t * C_{t-1} + i_t * C~_t), eliminating gradient decay across long sequence gaps.',
    },
    {
      q: 'How does a GRU (Gated Recurrent Unit) compare architecturally to an LSTM?',
      options: [
        'GRU has no gates at all',
        'GRU merges cell and hidden states into a single h_t, using only 2 gates (Reset and Update), reducing parameters by ~25% with comparable performance',
        'GRU has 6 gates instead of 3',
        'GRU cannot be trained with backpropagation',
      ],
      correct: 1,
      explanation: 'GRU simplifies the LSTM architecture by removing the separate cell state and coupling input/forget logic into an Update Gate, training faster with ~25% fewer parameters.',
    },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px', maxWidth: '1100px', margin: '0 auto' }}>
      {/* Title */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
          <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: 'rgba(139, 92, 246, 0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Repeat size={20} color="var(--accent-violet)" />
          </div>
          <span className="badge badge-indigo">Module 3</span>
          <span className="badge badge-cyan">Sequence Modeling</span>
        </div>
        <h2 style={{ fontSize: '2rem', fontWeight: 800, color: '#ffffff', marginBottom: '8px' }}>
          Recurrent Networks: RNNs, LSTMs, GRUs & Sequence Forecasting
        </h2>
        <p style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
          Sequence data requires architectures that maintain internal memory across discrete time steps. Compare standard recurrent units with gated architectures (LSTM and GRU) designed to solve the vanishing gradient dilemma.
        </p>
      </div>

      {/* SECTION 1: Gate Visualizer */}
      <RnnGateVisualizer />

      {/* SECTION 2: 3-Way Comparative Benchmark Workbench */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '6px' }}>
          2. Head-to-Head Benchmark: RNN vs LSTM vs GRU
        </h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
          Train real PyTorch <code>nn.RNN</code>, <code>nn.LSTM</code>, and <code>nn.GRU</code> simultaneously on the exact same multi-frequency time-series forecasting signal. Compare convergence speed, training time, and predictive accuracy.
        </p>

        {/* Hyperparameter Controls */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '20px' }}>
          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
              Epochs: {epochs}
            </label>
            <input
              type="range"
              min="10"
              max="40"
              step="5"
              value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value, 10))}
              style={{ width: '100%' }}
            />
          </div>

          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
              Hidden State Size: {hiddenSize}
            </label>
            <select
              className="select-field"
              value={hiddenSize}
              onChange={(e) => setHiddenSize(parseInt(e.target.value, 10))}
            >
              <option value="16">16 Units (Lightweight)</option>
              <option value="32">32 Units (Balanced)</option>
              <option value="64">64 Units (High Capacity)</option>
            </select>
          </div>

          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
              Lookback Sequence Length: {seqLen} steps
            </label>
            <select
              className="select-field"
              value={seqLen}
              onChange={(e) => setSeqLen(parseInt(e.target.value, 10))}
            >
              <option value="15">15 Timesteps</option>
              <option value="20">20 Timesteps</option>
              <option value="30">30 Timesteps</option>
            </select>
          </div>
        </div>

        <button className="btn-primary" onClick={handleRunBenchmark} disabled={loading} style={{ marginBottom: '24px' }}>
          <Play size={16} /> {loading ? 'Benchmarking 3 Models on CPU (~3-4s)...' : 'Run 3-Model Benchmark'}
        </button>

        {benchmarkResult && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
            {/* Model Comparison Cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '16px' }}>
              {Object.entries(benchmarkResult.models).map(([name, mData]: [string, any]) => {
                const color = name === 'RNN' ? 'var(--accent-rose)' : (name === 'LSTM' ? 'var(--accent-cyan)' : 'var(--accent-emerald)');
                return (
                  <div
                    key={name}
                    style={{
                      background: 'rgba(0, 0, 0, 0.35)',
                      padding: '16px',
                      borderRadius: 'var(--radius-md)',
                      border: `1px solid ${color}33`,
                      borderTop: `3px solid ${color}`,
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                      <h4 style={{ fontSize: '1.1rem', fontWeight: 700, color }}>{name}</h4>
                      <span className="badge badge-indigo">{mData.param_count} params</span>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '0.82rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Validation MSE Loss:</span>
                        <strong style={{ color: 'var(--text-primary)', fontFamily: 'Fira Code' }}>{mData.final_val_loss}</strong>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Total Training Time:</span>
                        <span style={{ color: 'var(--text-secondary)' }}>{mData.train_duration_ms} ms</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Train Loss (Final):</span>
                        <span style={{ color: 'var(--text-secondary)' }}>{mData.final_train_loss}</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Side-by-Side Loss Curves */}
            {(() => {
              // Combine histories for the chart
              const combinedHistory = benchmarkResult.models.RNN.history.map((h: any, idx: number) => ({
                epoch: h.epoch,
                rnn_val: benchmarkResult.models.RNN.history[idx].val_loss,
                lstm_val: benchmarkResult.models.LSTM.history[idx].val_loss,
                gru_val: benchmarkResult.models.GRU.history[idx].val_loss,
              }));

              return (
                <TrainingCurvesChart
                  title="Validation Loss Convergence: RNN vs LSTM vs GRU"
                  data={combinedHistory}
                  series={[
                    { key: 'rnn_val', name: 'Vanilla RNN Loss', color: '#f43f5e' },
                    { key: 'lstm_val', name: 'LSTM Loss', color: '#06b6d4' },
                    { key: 'gru_val', name: 'GRU Loss', color: '#10b981' },
                  ]}
                  yLabel="Val MSE"
                />
              );
            })()}

            {/* Time Series Multi-Step Predictions vs Ground Truth Chart */}
            <div style={{ background: 'rgba(0, 0, 0, 0.35)', padding: '16px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
              <h4 style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '10px' }}>
                Forecasted Waveform vs Actual Ground Truth (Next 30 Time Steps):
              </h4>
              <div style={{ display: 'flex', gap: '14px', marginBottom: '12px', fontSize: '0.8rem', flexWrap: 'wrap' }}>
                <span style={{ color: '#ffffff', fontWeight: 600 }}>— Ground Truth</span>
                <span style={{ color: '#f43f5e' }}>— Vanilla RNN</span>
                <span style={{ color: '#06b6d4' }}>— LSTM</span>
                <span style={{ color: '#10b981' }}>— GRU</span>
              </div>

              {/* Time series visual line chart */}
              <div style={{ width: '100%', overflowX: 'auto' }}>
                <svg viewBox="0 0 600 160" style={{ width: '100%', maxHeight: '160px', overflow: 'visible' }}>
                  {/* Zero Line */}
                  <line x1="20" y1="80" x2="580" y2="80" stroke="rgba(255, 255, 255, 0.1)" strokeDasharray="3 3" />

                  {/* Draw each waveform */}
                  {[
                    { vals: benchmarkResult.ground_truth, color: '#ffffff', width: 2.5, dash: undefined },
                    { vals: benchmarkResult.models.RNN.test_predictions, color: '#f43f5e', width: 1.8, dash: '4 4' },
                    { vals: benchmarkResult.models.LSTM.test_predictions, color: '#06b6d4', width: 2, dash: undefined },
                    { vals: benchmarkResult.models.GRU.test_predictions, color: '#10b981', width: 2, dash: undefined },
                  ].map((line, lIdx) => {
                    const points = line.vals
                      .map((v: number, i: number) => {
                        const x = 30 + (i / 29) * 540;
                        const y = 80 - v * 40;
                        return `${x},${y}`;
                      })
                      .join(' L ');

                    return (
                      <path
                        key={lIdx}
                        d={`M ${points}`}
                        fill="none"
                        stroke={line.color}
                        strokeWidth={line.width}
                        strokeDasharray={line.dash}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    );
                  })}
                </svg>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* SECTION 3: Conceptual Quiz */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '6px' }}>
          3. Concept Check & Quiz
        </h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '20px' }}>
          Test your understanding of recurrent memory architectures and gating mechanisms.
        </p>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {quizQuestions.map((q, qIdx) => {
            const selected = quizAnswers[qIdx];
            const isCorrect = selected === q.correct;

            return (
              <div key={qIdx} style={{ padding: '16px', background: 'rgba(0, 0, 0, 0.25)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.92rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '12px' }}>
                  {qIdx + 1}. {q.q}
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {q.options.map((opt, optIdx) => {
                    const isSelected = selected === optIdx;
                    let bgColor = 'rgba(255, 255, 255, 0.03)';
                    let borderColor = 'var(--border-subtle)';

                    if (quizSubmitted) {
                      if (optIdx === q.correct) {
                        bgColor = 'rgba(16, 185, 129, 0.15)';
                        borderColor = 'var(--accent-emerald)';
                      } else if (isSelected && !isCorrect) {
                        bgColor = 'rgba(244, 63, 94, 0.15)';
                        borderColor = 'var(--accent-rose)';
                      }
                    } else if (isSelected) {
                      bgColor = 'rgba(139, 92, 246, 0.15)';
                      borderColor = 'var(--accent-violet)';
                    }

                    return (
                      <button
                        key={optIdx}
                        disabled={quizSubmitted}
                        onClick={() => setQuizAnswers((prev) => ({ ...prev, [qIdx]: optIdx }))}
                        style={{
                          padding: '10px 14px',
                          borderRadius: '8px',
                          background: bgColor,
                          border: `1px solid ${borderColor}`,
                          color: 'var(--text-primary)',
                          textAlign: 'left',
                          fontSize: '0.85rem',
                          cursor: quizSubmitted ? 'default' : 'pointer',
                          transition: 'all 0.15s ease',
                        }}
                      >
                        {opt}
                      </button>
                    );
                  })}
                </div>

                {quizSubmitted && (
                  <div style={{ marginTop: '10px', fontSize: '0.82rem', color: isCorrect ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
                    <strong>{isCorrect ? 'Correct! ' : 'Incorrect. '}</strong>
                    <span style={{ color: 'var(--text-secondary)' }}>{q.explanation}</span>
                  </div>
                )}
              </div>
            );
          })}

          {!quizSubmitted ? (
            <button
              className="btn-primary"
              onClick={() => {
                setQuizSubmitted(true);
                markModuleComplete(3);
              }}
              disabled={Object.keys(quizAnswers).length < quizQuestions.length}
              style={{ alignSelf: 'flex-start' }}
            >
              Submit Concept Quiz
            </button>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--accent-emerald)', fontWeight: 600 }}>
              <CheckCircle2 size={18} /> Module 3 Concept Quiz Completed!
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

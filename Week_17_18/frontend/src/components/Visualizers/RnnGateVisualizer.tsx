import React, { useState } from 'react';
import { GitCommit, Activity, RefreshCw } from 'lucide-react';

export const RnnGateVisualizer: React.FC = () => {
  const [selectedArch, setSelectedArch] = useState<'lstm' | 'gru' | 'rnn'>('lstm');
  const [activeGate, setActiveGate] = useState<string>('forget');

  const gateDetails: Record<string, { title: string; formula: string; role: string; activation: string }> = {
    forget: {
      title: 'Forget Gate (f_t)',
      formula: 'f_t = \\sigma(W_f \\cdot [h_{t-1}, x_t] + b_f)',
      role: 'Decides what percentage of the previous cell memory C_{t-1} should be thrown away or retained (0 = completely erase, 1 = completely keep).',
      activation: 'Sigmoid \\sigma \\in (0, 1)',
    },
    input: {
      title: 'Input Gate (i_t) & Candidate (C~_t)',
      formula: 'i_t = \\sigma(W_i \\cdot [h_{t-1}, x_t] + b_i) \\quad \\tilde{C}_t = \\tanh(W_c \\cdot [h_{t-1}, x_t] + b_c)',
      role: 'Decides which new candidate values will be written into the cell state memory highway.',
      activation: 'Sigmoid \\sigma & tanh \\in (-1, 1)',
    },
    cell: {
      title: 'Cell State Highway (C_t)',
      formula: 'C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t',
      role: 'The linear conveyor belt that runs through the entire sequence chain. Additive gradient updates protect against vanishing gradients.',
      activation: 'Linear accumulator + Elementwise Hadamard product (\\odot)',
    },
    output: {
      title: 'Output Gate (o_t) & Hidden State (h_t)',
      formula: 'o_t = \\sigma(W_o \\cdot [h_{t-1}, x_t] + b_o) \\quad h_t = o_t \\odot \\tanh(C_t)',
      role: 'Filters the updated cell memory to produce the hidden representation h_t for current time step prediction and next cell transmission.',
      activation: 'Sigmoid \\sigma & tanh',
    },
    reset: {
      title: 'Reset Gate (r_t)',
      formula: 'r_t = \\sigma(W_r \\cdot [h_{t-1}, x_t] + b_r)',
      role: 'Determines how much of the past hidden state h_{t-1} to ignore when computing the new candidate hidden state.',
      activation: 'Sigmoid \\sigma \\in (0, 1)',
    },
    update: {
      title: 'Update Gate (z_t)',
      formula: 'z_t = \\sigma(W_z \\cdot [h_{t-1}, x_t] + b_z)',
      role: 'Acts as both forget and input gate simultaneously. Balances between copying past memory vs adopting new candidate.',
      activation: 'Sigmoid \\sigma \\in (0, 1)',
    },
    rnn_core: {
      title: 'Vanilla Recurrence Step',
      formula: 'h_t = \\tanh(W_{hh} \\cdot h_{t-1} + W_{xh} \\cdot x_t + b_h)',
      role: 'Direct non-linear recurrence. Repeated matrix multiplications cause gradients to exponentially decay (vanish) over long sequences.',
      activation: 'tanh \\in (-1, 1)',
    },
  };

  return (
    <div className="glass-panel" style={{ padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '18px' }}>
        <div>
          <h3 style={{ fontSize: '1.15rem', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Activity size={18} color="var(--accent-violet)" />
            Interactive Recurrent Cell Anatomy & Gating Explorer
          </h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
            Click on individual gates to understand mathematical signal routing inside LSTM, GRU, and Vanilla RNN cells.
          </p>
        </div>

        {/* Architecture switcher */}
        <div style={{ display: 'flex', background: 'rgba(0, 0, 0, 0.4)', borderRadius: 'var(--radius-md)', padding: '4px', border: '1px solid var(--border-subtle)' }}>
          <button
            className={`btn-secondary ${selectedArch === 'lstm' ? 'active-tab' : ''}`}
            style={{
              padding: '6px 14px',
              fontSize: '0.8rem',
              background: selectedArch === 'lstm' ? 'var(--accent-indigo)' : 'transparent',
              color: '#fff',
              border: 'none',
            }}
            onClick={() => {
              setSelectedArch('lstm');
              setActiveGate('forget');
            }}
          >
            LSTM (3 Gates)
          </button>
          <button
            className={`btn-secondary ${selectedArch === 'gru' ? 'active-tab' : ''}`}
            style={{
              padding: '6px 14px',
              fontSize: '0.8rem',
              background: selectedArch === 'gru' ? 'var(--accent-indigo)' : 'transparent',
              color: '#fff',
              border: 'none',
            }}
            onClick={() => {
              setSelectedArch('gru');
              setActiveGate('reset');
            }}
          >
            GRU (2 Gates)
          </button>
          <button
            className={`btn-secondary ${selectedArch === 'rnn' ? 'active-tab' : ''}`}
            style={{
              padding: '6px 14px',
              fontSize: '0.8rem',
              background: selectedArch === 'rnn' ? 'var(--accent-indigo)' : 'transparent',
              color: '#fff',
              border: 'none',
            }}
            onClick={() => {
              setSelectedArch('rnn');
              setActiveGate('rnn_core');
            }}
          >
            Vanilla RNN
          </button>
        </div>
      </div>

      {/* Interactive Gate Buttons */}
      <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginBottom: '18px' }}>
        {selectedArch === 'lstm' && (
          <>
            <button
              className="btn-secondary"
              style={{
                borderColor: activeGate === 'forget' ? 'var(--accent-rose)' : 'var(--border-subtle)',
                backgroundColor: activeGate === 'forget' ? 'rgba(244, 63, 94, 0.15)' : 'transparent',
              }}
              onClick={() => setActiveGate('forget')}
            >
              1. Forget Gate (f_t)
            </button>
            <button
              className="btn-secondary"
              style={{
                borderColor: activeGate === 'input' ? 'var(--accent-cyan)' : 'var(--border-subtle)',
                backgroundColor: activeGate === 'input' ? 'rgba(6, 182, 212, 0.15)' : 'transparent',
              }}
              onClick={() => setActiveGate('input')}
            >
              2. Input Gate (i_t)
            </button>
            <button
              className="btn-secondary"
              style={{
                borderColor: activeGate === 'cell' ? 'var(--accent-amber)' : 'var(--border-subtle)',
                backgroundColor: activeGate === 'cell' ? 'rgba(245, 158, 11, 0.15)' : 'transparent',
              }}
              onClick={() => setActiveGate('cell')}
            >
              3. Cell Memory Highway (C_t)
            </button>
            <button
              className="btn-secondary"
              style={{
                borderColor: activeGate === 'output' ? 'var(--accent-emerald)' : 'var(--border-subtle)',
                backgroundColor: activeGate === 'output' ? 'rgba(16, 185, 129, 0.15)' : 'transparent',
              }}
              onClick={() => setActiveGate('output')}
            >
              4. Output Gate (o_t)
            </button>
          </>
        )}

        {selectedArch === 'gru' && (
          <>
            <button
              className="btn-secondary"
              style={{
                borderColor: activeGate === 'reset' ? 'var(--accent-rose)' : 'var(--border-subtle)',
                backgroundColor: activeGate === 'reset' ? 'rgba(244, 63, 94, 0.15)' : 'transparent',
              }}
              onClick={() => setActiveGate('reset')}
            >
              1. Reset Gate (r_t)
            </button>
            <button
              className="btn-secondary"
              style={{
                borderColor: activeGate === 'update' ? 'var(--accent-cyan)' : 'var(--border-subtle)',
                backgroundColor: activeGate === 'update' ? 'rgba(6, 182, 212, 0.15)' : 'transparent',
              }}
              onClick={() => setActiveGate('update')}
            >
              2. Update Gate (z_t)
            </button>
          </>
        )}

        {selectedArch === 'rnn' && (
          <button
            className="btn-secondary"
            style={{
              borderColor: 'var(--accent-indigo)',
              backgroundColor: 'rgba(99, 102, 241, 0.15)',
            }}
            onClick={() => setActiveGate('rnn_core')}
          >
            Direct Recurrent Step
          </button>
        )}
      </div>

      {/* Selected Gate Card */}
      {gateDetails[activeGate] && (
        <div style={{ background: 'rgba(0, 0, 0, 0.35)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', padding: '18px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
            <h4 style={{ color: 'var(--accent-cyan)', fontSize: '1rem', fontWeight: 600 }}>
              {gateDetails[activeGate].title}
            </h4>
            <span className="badge badge-indigo">{gateDetails[activeGate].activation}</span>
          </div>

          <div className="code-box" style={{ marginBottom: '12px', color: '#67e8f9' }}>
            {gateDetails[activeGate].formula}
          </div>

          <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            {gateDetails[activeGate].role}
          </p>
        </div>
      )}

      {/* Visual Cell Diagram Representation */}
      <div style={{ marginTop: '20px', padding: '16px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: 'var(--radius-md)', border: '1px dashed var(--border-subtle)' }}>
        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '8px' }}>
          Cell Information Highway Flow:
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', overflowX: 'auto', padding: '10px 0' }}>
          <div style={{ padding: '8px 14px', background: 'rgba(99, 102, 241, 0.15)', borderRadius: '6px', border: '1px solid var(--accent-indigo)', fontSize: '0.8rem', textAlign: 'center' }}>
            Input x_t + Previous h_{'{t-1}'}
          </div>
          <span style={{ color: 'var(--text-muted)' }}>&rarr;</span>
          <div style={{ padding: '8px 14px', background: 'rgba(6, 182, 212, 0.15)', borderRadius: '6px', border: '1px solid var(--accent-cyan)', fontSize: '0.8rem', textAlign: 'center' }}>
            Linear Projection (W [h, x] + b)
          </div>
          <span style={{ color: 'var(--text-muted)' }}>&rarr;</span>
          <div style={{ padding: '8px 14px', background: 'rgba(139, 92, 246, 0.15)', borderRadius: '6px', border: '1px solid var(--accent-violet)', fontSize: '0.8rem', textAlign: 'center' }}>
            Gating Modulation & Memory Highway
          </div>
          <span style={{ color: 'var(--text-muted)' }}>&rarr;</span>
          <div style={{ padding: '8px 14px', background: 'rgba(16, 185, 129, 0.15)', borderRadius: '6px', border: '1px solid var(--accent-emerald)', fontSize: '0.8rem', textAlign: 'center' }}>
            Next Hidden State h_t
          </div>
        </div>
      </div>
    </div>
  );
};

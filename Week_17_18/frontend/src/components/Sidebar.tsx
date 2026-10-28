import React from 'react';
import { useLab } from '../context/LabContext';
import {
  LayoutDashboard,
  Layers,
  Image as ImageIcon,
  Repeat,
  ShieldAlert,
  Share2,
  Cpu,
  Bot,
  CheckCircle2,
  Circle,
  BookOpen
} from 'lucide-react';

export const modulesList = [
  { id: 1, name: 'PyTorch Fundamentals', icon: Layers, subtitle: 'Tensors, Autograd & Training Loop' },
  { id: 2, name: 'CNNs & Vision', icon: ImageIcon, subtitle: 'Filters, Pooling & MNIST CNN' },
  { id: 3, name: 'RNNs, LSTMs & GRUs', icon: Repeat, subtitle: 'Sequence Modeling & Comparison' },
  { id: 4, name: 'Regularization', icon: ShieldAlert, subtitle: 'Dropout, BatchNorm & Overfitting' },
  { id: 5, name: 'Transfer Learning', icon: Share2, subtitle: 'ResNet18, Frozen Layers & Head' },
  { id: 6, name: 'HF Transformers', icon: Cpu, subtitle: 'Tokenization, Sentiment, QA & Summary' },
  { id: 7, name: 'LLM APIs & Models', icon: Bot, subtitle: 'Open Models, Ollama, OpenAI & Keys' },
];

export const Sidebar: React.FC = () => {
  const { activeModule, setActiveModule, progress } = useLab();

  return (
    <aside
      style={{
        width: '280px',
        borderRight: '1px solid var(--border-subtle)',
        background: 'rgba(9, 13, 22, 0.95)',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        height: 'calc(100vh - 70px)',
        position: 'sticky',
        top: '70px',
        padding: '16px 12px',
      }}
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
        <div style={{ padding: '0 8px 8px 8px', fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
          Lab Curriculum
        </div>

        {/* Dashboard link */}
        <button
          onClick={() => setActiveModule(0)}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            padding: '10px 14px',
            borderRadius: 'var(--radius-md)',
            background: activeModule === 0 ? 'rgba(99, 102, 241, 0.15)' : 'transparent',
            border: activeModule === 0 ? '1px solid var(--border-active)' : '1px solid transparent',
            color: activeModule === 0 ? 'var(--text-primary)' : 'var(--text-secondary)',
            textAlign: 'left',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
            width: '100%',
          }}
        >
          <LayoutDashboard size={18} color={activeModule === 0 ? 'var(--accent-indigo)' : 'var(--text-muted)'} />
          <div>
            <div style={{ fontSize: '0.88rem', fontWeight: 600 }}>Lab Dashboard</div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Overview & System Status</div>
          </div>
        </button>

        <div style={{ height: '1px', background: 'var(--border-subtle)', margin: '8px 0' }} />

        {/* 7 Modules */}
        {modulesList.map((m) => {
          const Icon = m.icon;
          const isCompleted = progress.completedModules.includes(m.id);
          const isActive = activeModule === m.id;
          const runs = progress.practicalRuns[m.id] || 0;

          return (
            <button
              key={m.id}
              onClick={() => setActiveModule(m.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '9px 12px',
                borderRadius: 'var(--radius-md)',
                background: isActive ? 'rgba(99, 102, 241, 0.15)' : 'transparent',
                border: isActive ? '1px solid var(--border-active)' : '1px solid transparent',
                color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                textAlign: 'left',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                width: '100%',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <div
                  style={{
                    width: '30px',
                    height: '30px',
                    borderRadius: '8px',
                    background: isActive ? 'var(--accent-indigo)' : 'rgba(255, 255, 255, 0.05)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    transition: 'all 0.2s ease',
                  }}
                >
                  <Icon size={16} color={isActive ? '#ffffff' : 'var(--text-muted)'} />
                </div>
                <div>
                  <div style={{ fontSize: '0.84rem', fontWeight: 600 }}>
                    {m.id}. {m.name}
                  </div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                    {m.subtitle}
                  </div>
                </div>
              </div>

              <div>
                {isCompleted ? (
                  <CheckCircle2 size={16} color="var(--accent-emerald)" />
                ) : (
                  <Circle size={14} color="var(--border-subtle)" />
                )}
              </div>
            </button>
          );
        })}
      </div>

      {/* Footer Info */}
      <div style={{ padding: '12px', background: 'rgba(15, 23, 42, 0.6)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-secondary)', marginBottom: '4px' }}>
          <BookOpen size={14} color="var(--accent-cyan)" />
          <strong>CPU Accelerated</strong>
        </div>
        Real PyTorch tensors, autograd, and backward passes running locally in real time.
      </div>
    </aside>
  );
};

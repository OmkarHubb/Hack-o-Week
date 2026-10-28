import React from 'react';
import { useLab } from '../context/LabContext';
import { modulesList } from '../components/Sidebar';
import {
  Sparkles,
  ArrowRight,
  CheckCircle2,
  Cpu,
  ShieldCheck,
  TrendingUp,
  Award,
  Zap
} from 'lucide-react';

export const Dashboard: React.FC = () => {
  const { setActiveModule, progress, serverStatus } = useLab();

  const completedCount = progress.completedModules.length;
  const totalRuns = Object.values(progress.practicalRuns).reduce((a, b) => a + b, 0);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px', maxWidth: '1200px', margin: '0 auto' }}>
      {/* Hero Banner */}
      <div
        className="glass-panel"
        style={{
          padding: '36px',
          background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(6, 182, 212, 0.15) 100%)',
          border: '1px solid rgba(99, 102, 241, 0.3)',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        <div style={{ maxWidth: '750px', position: 'relative', zIndex: 2 }}>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', background: 'rgba(255, 255, 255, 0.1)', padding: '5px 12px', borderRadius: '9999px', fontSize: '0.8rem', fontWeight: 600, color: 'var(--accent-cyan)', marginBottom: '16px' }}>
            <Sparkles size={14} /> Interactive Deep Learning Platform
          </div>
          <h2 style={{ fontSize: '2.4rem', fontWeight: 800, lineHeight: 1.15, marginBottom: '14px', color: '#ffffff' }}>
            Master Deep Learning with Real PyTorch Models & Live Experiments
          </h2>
          <p style={{ fontSize: '1rem', color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: '24px' }}>
            Explore 7 interactive modules covering fundamental tensors, convolution filter mathematics, sequence memory gating in RNNs & LSTMs, regularization techniques, transfer learning with ResNet, transformer tokenization & pipelines, and modern LLM APIs.
          </p>

          <div style={{ display: 'flex', gap: '14px', flexWrap: 'wrap' }}>
            <button className="btn-primary" onClick={() => setActiveModule(1)}>
              Start Module 1: PyTorch Basics <ArrowRight size={16} />
            </button>
            <button
              className="btn-secondary"
              onClick={() => {
                const nextUnfinished = [1, 2, 3, 4, 5, 6, 7].find((id) => !progress.completedModules.includes(id)) || 1;
                setActiveModule(nextUnfinished);
              }}
            >
              Resume Next Exercise (Module {([1, 2, 3, 4, 5, 6, 7].find((id) => !progress.completedModules.includes(id)) || 1)})
            </button>
          </div>
        </div>
      </div>

      {/* Metrics Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '18px' }}>
        <div className="glass-panel" style={{ padding: '20px', display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'rgba(16, 185, 129, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <CheckCircle2 size={24} color="var(--accent-emerald)" />
          </div>
          <div>
            <div style={{ fontSize: '1.6rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              {completedCount} / 7
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Modules Completed</div>
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '20px', display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'rgba(99, 102, 241, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Zap size={24} color="var(--accent-indigo)" />
          </div>
          <div>
            <div style={{ fontSize: '1.6rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              {totalRuns}
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Real Experiments Executed</div>
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '20px', display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'rgba(6, 182, 212, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Cpu size={24} color="var(--accent-cyan)" />
          </div>
          <div>
            <div style={{ fontSize: '1.6rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              {serverStatus.device}
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Torch {serverStatus.pytorch_version}
            </div>
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '20px', display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'rgba(245, 158, 11, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Award size={24} color="var(--accent-amber)" />
          </div>
          <div>
            <div style={{ fontSize: '1.6rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              {completedCount === 7 ? 'Deep Learning Pro' : 'Apprentice'}
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Current Status Badge</div>
          </div>
        </div>
      </div>

      {/* 7 Modules Grid */}
      <div>
        <h3 style={{ fontSize: '1.3rem', fontWeight: 700, marginBottom: '16px', color: 'var(--text-primary)' }}>
          Curriculum & Hands-on Modules
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '20px' }}>
          {modulesList.map((m) => {
            const Icon = m.icon;
            const isDone = progress.completedModules.includes(m.id);
            const runs = progress.practicalRuns[m.id] || 0;

            return (
              <div
                key={m.id}
                className="glass-panel"
                style={{
                  padding: '24px',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between',
                  gap: '16px',
                  cursor: 'pointer',
                  borderColor: isDone ? 'rgba(16, 185, 129, 0.3)' : 'var(--border-subtle)',
                }}
                onClick={() => setActiveModule(m.id)}
              >
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                    <div style={{ width: '40px', height: '40px', borderRadius: '10px', background: 'rgba(99, 102, 241, 0.12)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <Icon size={20} color="var(--accent-indigo)" />
                    </div>
                    {isDone ? (
                      <span className="badge badge-emerald">
                        <CheckCircle2 size={12} /> Completed
                      </span>
                    ) : (
                      <span className="badge badge-indigo">Module {m.id}</span>
                    )}
                  </div>

                  <h4 style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '6px' }}>
                    {m.id}. {m.name}
                  </h4>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                    {m.subtitle}
                  </p>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: '14px', borderTop: '1px solid var(--border-subtle)', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  <span>{runs} experiment runs</span>
                  <span style={{ color: 'var(--accent-cyan)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '4px' }}>
                    Open Module &rarr;
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

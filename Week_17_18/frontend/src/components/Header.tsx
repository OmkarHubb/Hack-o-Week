import React from 'react';
import { useLab } from '../context/LabContext';
import { Cpu, RotateCcw, CheckCircle2, ShieldCheck, Flame } from 'lucide-react';

export const Header: React.FC = () => {
  const { progress, serverStatus, resetProgress } = useLab();

  const completedCount = progress.completedModules.length;
  const totalModules = 7;
  const progressPercent = Math.round((completedCount / totalModules) * 100);

  return (
    <header
      style={{
        height: '70px',
        borderBottom: '1px solid var(--border-subtle)',
        background: 'rgba(9, 13, 22, 0.85)',
        backdropFilter: 'blur(16px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 28px',
        position: 'sticky',
        top: 0,
        zIndex: 40,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
        <div
          style={{
            width: '38px',
            height: '38px',
            borderRadius: '10px',
            background: 'var(--gradient-brand)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: 'var(--shadow-glow)',
          }}
        >
          <Flame size={22} color="#ffffff" />
        </div>
        <div>
          <h1 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
            Deep Learning Learning Lab
            <span className="badge badge-indigo" style={{ fontSize: '0.65rem' }}>PyTorch Edition</span>
          </h1>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Week 17-18 Practical Curriculum & Interactive Workbench
          </p>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
        {/* Backend Status Badge */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            background: 'rgba(15, 23, 42, 0.7)',
            padding: '6px 14px',
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border-subtle)',
            fontSize: '0.8rem',
          }}
        >
          <span
            style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: serverStatus.online ? 'var(--accent-emerald)' : 'var(--accent-rose)',
              boxShadow: serverStatus.online ? '0 0 10px var(--accent-emerald)' : 'none',
            }}
          />
          <Cpu size={14} color="var(--text-secondary)" />
          <span style={{ color: 'var(--text-secondary)' }}>
            PyTorch {serverStatus.pytorch_version} ({serverStatus.device})
          </span>
        </div>

        {/* Progress Bar Badge */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            background: 'rgba(15, 23, 42, 0.7)',
            padding: '6px 14px',
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border-subtle)',
          }}
        >
          <CheckCircle2 size={16} color="var(--accent-cyan)" />
          <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', gap: '10px' }}>
              <span style={{ color: 'var(--text-muted)' }}>Progress:</span>
              <strong style={{ color: 'var(--text-primary)' }}>{completedCount} / {totalModules} ({progressPercent}%)</strong>
            </div>
            <div style={{ width: '100px', height: '4px', background: 'rgba(255, 255, 255, 0.1)', borderRadius: '2px', overflow: 'hidden' }}>
              <div
                style={{
                  width: `${progressPercent}%`,
                  height: '100%',
                  background: 'var(--gradient-brand)',
                  transition: 'width 0.4s ease',
                }}
              />
            </div>
          </div>
        </div>

        {/* Reset button */}
        <button
          className="btn-secondary"
          onClick={resetProgress}
          title="Reset Learning Progress"
          style={{ padding: '7px 12px', fontSize: '0.8rem' }}
        >
          <RotateCcw size={14} />
          Reset
        </button>
      </div>
    </header>
  );
};

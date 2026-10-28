import React, { useState } from 'react';
import { useLab } from '../context/LabContext';
import { api } from '../services/api';
import { TrainingCurvesChart } from '../components/Visualizers/TrainingCurvesChart';
import {
  Share2,
  Play,
  CheckCircle2,
  Lock,
  Unlock,
  Cpu,
  Layers,
  Sparkles,
  HelpCircle
} from 'lucide-react';

export const Module5TransferLearning: React.FC = () => {
  const { markModuleComplete, recordPracticalRun } = useLab();

  const [mode, setMode] = useState<string>('feature_extraction');
  const [epochs, setEpochs] = useState<number>(5);
  const [learningRate, setLearningRate] = useState<number>(0.001);
  const [targetClasses, setTargetClasses] = useState<number>(5);
  const [transferResult, setTransferResult] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);

  // Quiz state
  const [quizAnswers, setQuizAnswers] = useState<Record<number, number>>({});
  const [quizSubmitted, setQuizSubmitted] = useState<boolean>(false);

  const handleRunTransfer = async () => {
    setLoading(true);
    try {
      const res = await api.runTransferLearning({
        mode,
        epochs,
        learning_rate: learningRate,
        target_classes: targetClasses,
      });
      setTransferResult(res);
      recordPracticalRun(5);
      markModuleComplete(5);
    } catch (err: any) {
      alert(err.message);
    } finally {
      setLoading(false);
    }
  };

  const quizQuestions = [
    {
      q: 'What is the key mechanism of "Feature Extraction" in transfer learning?',
      options: [
        'Training the entire network from scratch with random Xavier initialization',
        'Freezing the pretrained backbone weights (requires_grad = False) and only training a newly appended classification head',
        'Discarding all weights and using decision trees',
        'Running images through an unweighted autoencoder',
      ],
      correct: 1,
      explanation: 'Feature extraction freezes all pretrained convolutional layers (param.requires_grad = False) to preserve universal ImageNet representations and only trains the final linear classification layer.',
    },
    {
      q: 'When is "Fine-Tuning" preferred over simple Feature Extraction?',
      options: [
        'When you have only 5 training images total',
        'When you have a moderately large target dataset and want to adapt deep high-level representations to specific domain nuances',
        'Only on audio files',
        'When you want training to take 0 seconds',
      ],
      correct: 1,
      explanation: 'Fine-tuning unfreezes deeper layers (like Layer 4 of ResNet) with a low learning rate so high-level semantic filters can specialize to the target domain without destroying low-level edge features.',
    },
    {
      q: 'Why do we typically use a smaller learning rate (e.g. 1e-4) when fine-tuning pretrained layers compared to training from scratch?',
      options: [
        'To prevent catastrophic forgetting and avoid destroying pretrained weight features',
        'Because PyTorch crashes if learning rate > 0.001',
        'To make the GPU fans spin slower',
        'It is required by the ImageNet license',
      ],
      correct: 0,
      explanation: 'A small learning rate ensures we gently nudge pretrained weights rather than making giant gradient steps that obliterate the general visual representations learned during ImageNet pretraining.',
    },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px', maxWidth: '1100px', margin: '0 auto' }}>
      {/* Title */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
          <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: 'rgba(99, 102, 241, 0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Share2 size={20} color="var(--accent-indigo)" />
          </div>
          <span className="badge badge-indigo">Module 5</span>
          <span className="badge badge-cyan">Pretrained ResNet-18</span>
        </div>
        <h2 style={{ fontSize: '2rem', fontWeight: 800, color: '#ffffff', marginBottom: '8px' }}>
          Transfer Learning: Feature Extraction, Frozen Layers & Fine-Tuning
        </h2>
        <p style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
          Rather than training deep vision models from scratch on massive datasets for weeks, Transfer Learning leverages representations learned on ImageNet (1.4 million images). Master layer freezing, head replacement, and differential fine-tuning.
        </p>
      </div>

      {/* SECTION 1: ResNet-18 Layer Architecture & Freeze Visualizer */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '14px' }}>
          1. ResNet-18 Hierarchical Visual Feature Pipeline
        </h3>

        {/* Mode Selector */}
        <div style={{ display: 'flex', gap: '12px', marginBottom: '20px' }}>
          <button
            className={`btn-secondary ${mode === 'feature_extraction' ? 'active-tab' : ''}`}
            style={{
              borderColor: mode === 'feature_extraction' ? 'var(--accent-cyan)' : 'var(--border-subtle)',
              backgroundColor: mode === 'feature_extraction' ? 'rgba(6, 182, 212, 0.15)' : 'transparent',
              padding: '10px 18px',
              fontWeight: 600,
            }}
            onClick={() => setMode('feature_extraction')}
          >
            <Lock size={15} color="var(--accent-cyan)" />
            Mode A: Feature Extraction (Freeze All Backbone)
          </button>

          <button
            className={`btn-secondary ${mode === 'fine_tuning' ? 'active-tab' : ''}`}
            style={{
              borderColor: mode === 'fine_tuning' ? 'var(--accent-indigo)' : 'var(--border-subtle)',
              backgroundColor: mode === 'fine_tuning' ? 'rgba(99, 102, 241, 0.15)' : 'transparent',
              padding: '10px 18px',
              fontWeight: 600,
            }}
            onClick={() => setMode('fine_tuning')}
          >
            <Unlock size={15} color="var(--accent-indigo)" />
            Mode B: Fine-Tuning (Unfreeze Layer 4 + Head)
          </button>
        </div>

        {/* Layer-by-layer visual cards */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px', marginBottom: '20px' }}>
          {[
            { name: 'Conv1 + BN', type: 'Edges & Colors', frozen: true, params: '9.4K' },
            { name: 'Layer 1 (ResBlock)', type: 'Textures & Gradients', frozen: true, params: '147K' },
            { name: 'Layer 2 (ResBlock)', type: 'Complex Motifs', frozen: true, params: '525K' },
            { name: 'Layer 3 (ResBlock)', type: 'Object Parts', frozen: true, params: '2.1M' },
            { name: 'Layer 4 (ResBlock)', type: 'High-Level Semantics', frozen: mode === 'feature_extraction', params: '8.4M' },
            { name: 'FC Head (New)', type: `${targetClasses} Custom Classes`, frozen: false, params: '2.5K' },
          ].map((l, i) => (
            <div
              key={i}
              style={{
                background: 'rgba(0, 0, 0, 0.35)',
                padding: '14px',
                borderRadius: 'var(--radius-md)',
                border: `1px solid ${l.frozen ? 'rgba(255, 255, 255, 0.08)' : 'var(--accent-cyan)'}`,
                borderTop: `3px solid ${l.frozen ? 'var(--text-muted)' : 'var(--accent-cyan)'}`,
                textAlign: 'center',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '6px' }}>
                {l.frozen ? <Lock size={16} color="var(--text-muted)" /> : <Unlock size={16} color="var(--accent-cyan)" />}
              </div>
              <div style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-primary)' }}>{l.name}</div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', margin: '4px 0' }}>{l.type}</div>
              <div style={{ fontSize: '0.75rem', fontWeight: 600, color: l.frozen ? 'var(--text-muted)' : 'var(--accent-emerald)' }}>
                {l.frozen ? 'FROZEN (requires_grad=False)' : 'TRAINABLE (Active Grads)'}
              </div>
            </div>
          ))}
        </div>

        {/* Parameter count comparison */}
        <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', padding: '12px 16px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: 'var(--radius-sm)' }}>
          <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
            Total Parameters: <strong style={{ color: '#fff' }}>11,179,077 (11.2M)</strong>
          </span>
          <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
            Trainable Parameters:{' '}
            <strong style={{ color: mode === 'feature_extraction' ? 'var(--accent-cyan)' : 'var(--accent-indigo)' }}>
              {mode === 'feature_extraction' ? '2,565 (0.02% of model)' : '8,396,293 (75.1% of model)'}
            </strong>
          </span>
        </div>
      </div>

      {/* SECTION 2: Real Training Workbench */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '6px' }}>
          2. Train Transfer Learning Model on Target Domain
        </h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
          Execute the training loop on the custom {targetClasses}-class target task with ResNet-18 extracted representations.
        </p>

        {/* Hyperparameter Controls */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '20px' }}>
          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
              Target Domain Classes: {targetClasses}
            </label>
            <select
              className="select-field"
              value={targetClasses}
              onChange={(e) => setTargetClasses(parseInt(e.target.value, 10))}
            >
              <option value="3">3 Classes (Fastest)</option>
              <option value="5">5 Classes (Standard)</option>
              <option value="10">10 Classes</option>
            </select>
          </div>

          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
              Epochs: {epochs}
            </label>
            <input
              type="range"
              min="2"
              max="10"
              step="1"
              value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value, 10))}
              style={{ width: '100%' }}
            />
          </div>

          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
              Learning Rate: {learningRate}
            </label>
            <select
              className="select-field"
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value))}
            >
              <option value="0.0005">0.0005 (Gentle Fine-Tuning)</option>
              <option value="0.001">0.001 (Recommended)</option>
              <option value="0.005">0.005 (Fast Head)</option>
            </select>
          </div>
        </div>

        <button className="btn-primary" onClick={handleRunTransfer} disabled={loading} style={{ marginBottom: '24px' }}>
          <Play size={16} /> {loading ? `Training ResNet-18 (${mode})...` : `Run ${mode === 'feature_extraction' ? 'Feature Extraction' : 'Fine-Tuning'} Training`}
        </button>

        {transferResult && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {/* Metrics */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
              <div style={{ padding: '14px', background: 'rgba(16, 185, 129, 0.1)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Final Validation Accuracy</div>
                <div style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--accent-emerald)' }}>
                  {transferResult.final_val_acc}%
                </div>
              </div>

              <div style={{ padding: '14px', background: 'rgba(99, 102, 241, 0.1)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(99, 102, 241, 0.3)' }}>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Training Duration</div>
                <div style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--accent-indigo)' }}>
                  {transferResult.training_time_ms} ms
                </div>
              </div>
            </div>

            {/* Curves */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '16px' }}>
              <TrainingCurvesChart
                title="Transfer Learning Loss"
                data={transferResult.history}
                series={[
                  { key: 'train_loss', name: 'Train Loss', color: '#6366f1' },
                  { key: 'val_loss', name: 'Val Loss', color: '#ec4899', strokeDash: '4 4' },
                ]}
                yLabel="Loss"
              />

              <TrainingCurvesChart
                title="Transfer Learning Accuracy (%)"
                data={transferResult.history}
                series={[
                  { key: 'train_acc', name: 'Train Acc %', color: '#10b981' },
                  { key: 'val_acc', name: 'Val Acc %', color: '#06b6d4', strokeDash: '4 4' },
                ]}
                yLabel="Accuracy %"
              />
            </div>

            {/* Code Implementation Snippet */}
            <div style={{ background: 'rgba(0, 0, 0, 0.4)', borderRadius: 'var(--radius-md)', padding: '18px', border: '1px solid var(--border-subtle)' }}>
              <h4 style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '10px' }}>
                PyTorch Code Implementation:
              </h4>
              <div className="code-box" style={{ fontSize: '0.82rem' }}>
{`import torchvision.models as models
import torch.nn as nn

# 1. Load Pretrained ResNet-18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 2. Freeze backbone layers
for param in model.parameters():
    param.requires_grad = False  # Locks weights!

# 3. Replace classification head (only this layer is trainable)
num_ftrs = model.fc.in_features  # 512
model.fc = nn.Linear(num_ftrs, ${targetClasses})

# 4. Train only model.fc parameters
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)`}
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
          Test your understanding of transfer learning and layer freezing strategies.
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
                      bgColor = 'rgba(99, 102, 241, 0.15)';
                      borderColor = 'var(--accent-indigo)';
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
                markModuleComplete(5);
              }}
              disabled={Object.keys(quizAnswers).length < quizQuestions.length}
              style={{ alignSelf: 'flex-start' }}
            >
              Submit Concept Quiz
            </button>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--accent-emerald)', fontWeight: 600 }}>
              <CheckCircle2 size={18} /> Module 5 Concept Quiz Completed!
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

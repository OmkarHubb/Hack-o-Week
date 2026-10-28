import React, { useState } from 'react';
import { useLab } from '../context/LabContext';
import { api } from '../services/api';
import { TrainingCurvesChart } from '../components/Visualizers/TrainingCurvesChart';
import {
  ShieldAlert,
  Play,
  CheckCircle2,
  AlertTriangle,
  Zap,
  TrendingDown,
  Layers,
  StopCircle
} from 'lucide-react';

export const Module4Regularization: React.FC = () => {
  const { markModuleComplete, recordPracticalRun } = useLab();

  const [dropoutRate, setDropoutRate] = useState<number>(0.4);
  const [epochs, setEpochs] = useState<number>(40);
  const [learningRate, setLearningRate] = useState<number>(0.01);
  const [weightDecay, setWeightDecay] = useState<number>(0.0001);
  const [experimentResult, setExperimentResult] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);

  // Quiz state
  const [quizAnswers, setQuizAnswers] = useState<Record<number, number>>({});
  const [quizSubmitted, setQuizSubmitted] = useState<boolean>(false);

  const handleRunExperiment = async () => {
    setLoading(true);
    try {
      const res = await api.runRegularization({
        dropout_rate: dropoutRate,
        epochs,
        learning_rate: learningRate,
        weight_decay: weightDecay,
      });
      setExperimentResult(res);
      recordPracticalRun(4);
      markModuleComplete(4);
    } catch (err: any) {
      alert(err.message);
    } finally {
      setLoading(false);
    }
  };

  const quizQuestions = [
    {
      q: 'Why does Dropout act like an ensemble of multiple sub-networks?',
      options: [
        'It downloads 100 pretrained models from Hugging Face',
        'Dropping units with probability p during each iteration means each mini-batch trains a randomly thinned sub-network from 2^N possible architectures',
        'It converts the neural network into a random forest',
        'It permanently deletes weights during testing',
      ],
      correct: 1,
      explanation: 'In a network with N nodes, dropout samples from 2^N possible thinned sub-networks. At test time, using all weights scaled by (1-p) approximates the geometric mean of this vast ensemble!',
    },
    {
      q: 'How does Batch Normalization behave differently during training vs inference (model.eval())?',
      options: [
        'It is only active during inference',
        'During training, it normalizes using current mini-batch mean and variance; during inference, it uses running exponential moving average statistics',
        'It multiplies weights by 10 during inference',
        'It disables all non-linear activation functions',
      ],
      correct: 1,
      explanation: 'During training, mini-batch statistics (mu_B, sigma_B) normalize each layer. PyTorch accumulates running running_mean and running_var, which model.eval() uses deterministically during test time.',
    },
    {
      q: 'What is the "Generalization Gap" in deep learning?',
      options: [
        'The difference between CPU and GPU clock speeds',
        'The performance disparity between Training Accuracy (or Loss) and Validation/Test Accuracy',
        'The distance between input and output layers',
        'The time taken to save a model checkpoint',
      ],
      correct: 1,
      explanation: 'Generalization gap measures how much better a model performs on training data versus unseen validation data. A massive gap (e.g. 100% train vs 60% test) is the textbook definition of overfitting.',
    },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px', maxWidth: '1100px', margin: '0 auto' }}>
      {/* Title */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
          <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: 'rgba(244, 63, 94, 0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <ShieldAlert size={20} color="var(--accent-rose)" />
          </div>
          <span className="badge badge-rose" style={{ background: 'rgba(244, 63, 94, 0.15)', color: '#fda4af', border: '1px solid rgba(244, 63, 94, 0.3)' }}>
            Module 4
          </span>
          <span className="badge badge-indigo">Optimization & Generalization</span>
        </div>
        <h2 style={{ fontSize: '2rem', fontWeight: 800, color: '#ffffff', marginBottom: '8px' }}>
          Regularization: Combating Overfitting with Dropout, BatchNorm & Early Stopping
        </h2>
        <p style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
          Deep neural networks possess massive parameter capacity and can easily memorize noisy training data rather than learning true underlying relationships. Contrast baseline overfitting against Dropout, Batch Normalization, and Early Stopping.
        </p>
      </div>

      {/* SECTION 1: Theoretical Mechanics */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '14px' }}>
          1. The Three Core Pillars of Neural Regularization
        </h3>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
          <div style={{ background: 'rgba(0, 0, 0, 0.35)', padding: '16px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)', borderTop: '3px solid var(--accent-indigo)' }}>
            <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--accent-indigo)', marginBottom: '6px' }}>
              Dropout (p = 0.2 ~ 0.5)
            </div>
            <div className="code-box" style={{ fontSize: '0.78rem', marginBottom: '10px', color: '#a5b4fc' }}>
              r ~ Bernoulli(1 - p)
              y~ = (r * y) / (1 - p)
            </div>
            <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              Randomly drops units during forward passes, breaking co-adaptation of weights and preventing features from relying on specific companion activations.
            </p>
          </div>

          <div style={{ background: 'rgba(0, 0, 0, 0.35)', padding: '16px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)', borderTop: '3px solid var(--accent-cyan)' }}>
            <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--accent-cyan)', marginBottom: '6px' }}>
              Batch Normalization
            </div>
            <div className="code-box" style={{ fontSize: '0.78rem', marginBottom: '10px', color: '#67e8f9' }}>
              x_hat = (x - mu_B) / sqrt(var_B + eps)
              y = gamma * x_hat + beta
            </div>
            <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              Normalizes layer inputs per mini-batch, dampening internal covariate shift. Acts as mild regularization due to slight stochastic mini-batch noise.
            </p>
          </div>

          <div style={{ background: 'rgba(0, 0, 0, 0.35)', padding: '16px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)', borderTop: '3px solid var(--accent-amber)' }}>
            <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--accent-amber)', marginBottom: '6px' }}>
              Early Stopping
            </div>
            <div className="code-box" style={{ fontSize: '0.78rem', marginBottom: '10px', color: '#fcd34d' }}>
              Halt when Val Loss &gt;= Min(Val Loss)
              for 'patience' epochs
            </div>
            <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              Tracks validation loss at every epoch. When validation error begins climbing while training error continues falling, training stops and restores the best checkpoint.
            </p>
          </div>
        </div>
      </div>

      {/* SECTION 2: Real Overfitting Ablation Workbench */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '6px' }}>
          2. Live Overfitting & Regularization Ablation Experiment
        </h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
          We train an overparameterized 3-layer MLP (96 hidden units) on a noisy dataset with only 90 training samples. Watch how the unregularized Baseline memorizes noise and explodes validation loss, while Dropout, BatchNorm, and Early Stopping stabilize generalization.
        </p>

        {/* Hyperparameter Controls */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '20px' }}>
          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
              Dropout Rate p: {dropoutRate}
            </label>
            <input
              type="range"
              min="0.1"
              max="0.7"
              step="0.05"
              value={dropoutRate}
              onChange={(e) => setDropoutRate(parseFloat(e.target.value))}
              style={{ width: '100%' }}
            />
          </div>

          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
              Max Epochs: {epochs}
            </label>
            <input
              type="range"
              min="20"
              max="60"
              step="5"
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
              <option value="0.005">0.005</option>
              <option value="0.01">0.01 (Standard)</option>
              <option value="0.02">0.02</option>
            </select>
          </div>
        </div>

        <button className="btn-primary" onClick={handleRunExperiment} disabled={loading} style={{ marginBottom: '24px' }}>
          <Play size={16} /> {loading ? 'Running 4 Ablation Configurations on CPU...' : 'Execute Regularization Ablation'}
        </button>

        {experimentResult && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
            {/* Generalization Gap Summary Cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '16px' }}>
              {Object.entries(experimentResult.configurations).map(([name, cfg]: [string, any]) => {
                const isOverfit = cfg.generalization_gap > 25.0;
                return (
                  <div
                    key={name}
                    style={{
                      background: 'rgba(0, 0, 0, 0.35)',
                      padding: '16px',
                      borderRadius: 'var(--radius-md)',
                      border: '1px solid var(--border-subtle)',
                      borderLeft: `4px solid ${isOverfit ? 'var(--accent-rose)' : 'var(--accent-emerald)'}`,
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                      <strong style={{ fontSize: '0.92rem', color: 'var(--text-primary)' }}>{name}</strong>
                      {isOverfit ? (
                        <span className="badge badge-rose" style={{ background: 'rgba(244, 63, 94, 0.15)', color: '#f43f5e', fontSize: '0.65rem' }}>
                          Severe Overfit
                        </span>
                      ) : (
                        <span className="badge badge-emerald" style={{ fontSize: '0.65rem' }}>
                          Regulated
                        </span>
                      )}
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '5px', fontSize: '0.8rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Train vs Val Accuracy:</span>
                        <span>
                          <strong style={{ color: 'var(--accent-indigo)' }}>{cfg.final_train_acc}%</strong> / <strong style={{ color: 'var(--accent-emerald)' }}>{cfg.final_val_acc}%</strong>
                        </span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Generalization Gap (Δ):</span>
                        <strong style={{ color: isOverfit ? 'var(--accent-rose)' : 'var(--accent-cyan)', fontFamily: 'Fira Code' }}>
                          +{cfg.generalization_gap}%
                        </strong>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Halt Epoch:</span>
                        <span style={{ color: 'var(--text-secondary)' }}>Epoch {cfg.stopped_epoch}</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Validation Loss Curves Comparison */}
            {(() => {
              const baselineHistory = experimentResult.configurations['Baseline (No Regularization)'].history;
              const dropoutHistory = experimentResult.configurations[`Dropout (p=${dropoutRate})`].history;
              const bnHistory = experimentResult.configurations['Batch Normalization'].history;
              const esHistory = experimentResult.configurations['Early Stopping (Patience=6)'].history;

              const combinedLoss = baselineHistory.map((h: any, idx: number) => ({
                epoch: h.epoch,
                baseline_val: h.val_loss,
                dropout_val: dropoutHistory[idx]?.val_loss,
                bn_val: bnHistory[idx]?.val_loss,
                es_val: esHistory[idx]?.val_loss,
              }));

              return (
                <TrainingCurvesChart
                  title="Validation Loss: Baseline Overfitting vs Regularization Techniques"
                  data={combinedLoss}
                  series={[
                    { key: 'baseline_val', name: 'Baseline (Diverges Upwards)', color: '#f43f5e' },
                    { key: 'dropout_val', name: 'Dropout (Smooth & Stable)', color: '#6366f1' },
                    { key: 'bn_val', name: 'Batch Normalization', color: '#06b6d4' },
                    { key: 'es_val', name: 'Early Stopping (Halted Early)', color: '#f59e0b', strokeDash: '3 3' },
                  ]}
                  yLabel="Validation Loss"
                />
              );
            })()}

            {/* Analysis Note */}
            <div style={{ padding: '14px 18px', background: 'rgba(244, 63, 94, 0.08)', borderRadius: 'var(--radius-md)', borderLeft: '3px solid var(--accent-rose)', fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              <strong style={{ color: '#ffffff' }}>Observation:</strong> Notice how the Baseline model's validation loss climbs higher and higher after epoch ~15 despite training loss dropping to zero. This divergence is the hallmark of overfitting. Both <strong>Dropout</strong> and <strong>Batch Normalization</strong> prevent this divergence, keeping validation error bounded.
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
          Test your understanding of overfitting and regularization methods.
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
                      bgColor = 'rgba(244, 63, 94, 0.15)';
                      borderColor = 'var(--accent-rose)';
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
                markModuleComplete(4);
              }}
              disabled={Object.keys(quizAnswers).length < quizQuestions.length}
              style={{ alignSelf: 'flex-start' }}
            >
              Submit Concept Quiz
            </button>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--accent-emerald)', fontWeight: 600 }}>
              <CheckCircle2 size={18} /> Module 4 Concept Quiz Completed!
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

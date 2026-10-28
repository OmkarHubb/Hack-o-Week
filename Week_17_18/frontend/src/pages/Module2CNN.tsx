import React, { useState } from 'react';
import { useLab } from '../context/LabContext';
import { api } from '../services/api';
import { ConvFilterVisualizer } from '../components/Visualizers/ConvFilterVisualizer';
import { TrainingCurvesChart } from '../components/Visualizers/TrainingCurvesChart';
import {
  Image as ImageIcon,
  Play,
  CheckCircle2,
  Sliders,
  Layers,
  Sparkles,
  Eye
} from 'lucide-react';

export const Module2CNN: React.FC = () => {
  const { markModuleComplete, recordPracticalRun } = useLab();

  // Training state
  const [epochs, setEpochs] = useState<number>(5);
  const [learningRate, setLearningRate] = useState<number>(0.002);
  const [batchSize, setBatchSize] = useState<number>(32);
  const [trainResult, setTrainResult] = useState<any>(null);
  const [trainingLoading, setTrainingLoading] = useState<boolean>(false);

  // Sample inspection state
  const [sampleIdx, setSampleIdx] = useState<number>(0);
  const [sampleData, setSampleData] = useState<any>(null);
  const [sampleLoading, setSampleLoading] = useState<boolean>(false);

  // Quiz state
  const [quizAnswers, setQuizAnswers] = useState<Record<number, number>>({});
  const [quizSubmitted, setQuizSubmitted] = useState<boolean>(false);

  const handleTrainCNN = async () => {
    setTrainingLoading(true);
    try {
      const res = await api.trainCnn({
        dataset_name: 'mnist',
        epochs,
        learning_rate: learningRate,
        batch_size: batchSize,
      });
      setTrainResult(res);
      recordPracticalRun(2);
      markModuleComplete(2);
      // Auto-load first sample
      handleInspectSample(0);
    } catch (err: any) {
      alert(err.message);
    } finally {
      setTrainingLoading(false);
    }
  };

  const handleInspectSample = async (idx: number) => {
    setSampleLoading(true);
    setSampleIdx(idx);
    try {
      const res = await api.predictCnnSample(idx);
      setSampleData(res);
    } catch (err: any) {
      console.error(err);
    } finally {
      setSampleLoading(false);
    }
  };

  const quizQuestions = [
    {
      q: 'What is the primary advantage of Convolutional layers over standard Fully-Connected (Dense) layers for images?',
      options: [
        'They require 100x more floating point calculations',
        'Parameter sharing and local receptive fields provide translation invariance and reduce parameter count',
        'They eliminate the need for activation functions',
        'They only work on 1D audio sequences',
      ],
      correct: 1,
      explanation: 'CNNs use small sliding kernels with shared weights across the whole image. This exploits spatial locality and translation invariance while drastically reducing weights from millions down to thousands.',
    },
    {
      q: 'What does a Max Pooling layer with kernel_size=2 and stride=2 achieve?',
      options: [
        'Increases image resolution by 2x',
        'Downsamples feature map dimensions by half (e.g. 28x28 -> 14x14), retaining the most prominent features while providing spatial translation tolerance',
        'Converts RGB images into grayscale',
        'Multiplies all feature map values by 2',
      ],
      correct: 1,
      explanation: 'Max pooling takes the maximum value in each 2x2 patch, cutting spatial width and height by 50% and reducing computational complexity for deeper layers.',
    },
    {
      q: 'What hierarchical pattern do deep CNN layers typically learn?',
      options: [
        'High-level objects in early layers and simple edges in deep layers',
        'Early layers learn low-level edges and textures; middle layers learn parts; deep layers learn high-level semantic class concepts',
        'Only random white noise',
        'All layers learn identical representations',
      ],
      correct: 1,
      explanation: 'CNNs naturally form hierarchical feature representations: early convs capture low-level gradients/edges, middle convs combine edges into textures/motifs, and deeper convs capture complete object parts.',
    },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px', maxWidth: '1100px', margin: '0 auto' }}>
      {/* Title */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
          <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: 'rgba(6, 182, 212, 0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <ImageIcon size={20} color="var(--accent-cyan)" />
          </div>
          <span className="badge badge-cyan">Module 2</span>
          <span className="badge badge-indigo">Computer Vision</span>
        </div>
        <h2 style={{ fontSize: '2rem', fontWeight: 800, color: '#ffffff', marginBottom: '8px' }}>
          Convolutional Neural Networks: Filters, Feature Maps & Image Classification
        </h2>
        <p style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
          Learn how 2D convolutions slide parameterized kernels over image matrices, extract translation-invariant edge and texture features, and downsample using pooling. Train a real PyTorch CNN on handwritten digits in seconds and inspect intermediate feature map activations.
        </p>
      </div>

      {/* SECTION 1: Interactive Convolution Filter Playground */}
      <ConvFilterVisualizer />

      {/* SECTION 2: Real MNIST CNN Training Workbench */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '6px' }}>
          2. Train a Real PyTorch CNN on MNIST Digits
        </h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
          Architecture: <code>Conv2D(1-&gt;16) -&gt; BatchNorm -&gt; ReLU -&gt; MaxPool(2x2) -&gt; Conv2D(16-&gt;32) -&gt; BatchNorm -&gt; ReLU -&gt; MaxPool(2x2) -&gt; Dense(64) -&gt; Output(10)</code>.
        </p>

        {/* Hyperparameter Controls */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '20px' }}>
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
              <option value="0.001">0.001 (Conservative)</option>
              <option value="0.002">0.002 (Recommended)</option>
              <option value="0.005">0.005 (Aggressive)</option>
            </select>
          </div>

          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
              Batch Size: {batchSize}
            </label>
            <select
              className="select-field"
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value, 10))}
            >
              <option value="16">16 Samples</option>
              <option value="32">32 Samples</option>
              <option value="64">64 Samples</option>
            </select>
          </div>
        </div>

        <button className="btn-primary" onClick={handleTrainCNN} disabled={trainingLoading} style={{ marginBottom: '20px' }}>
          <Play size={16} /> {trainingLoading ? 'Training CNN on CPU (~2-3s)...' : 'Train CNN Model'}
        </button>

        {trainResult && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {/* Accuracy Banner */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
              <div style={{ padding: '14px', background: 'rgba(16, 185, 129, 0.1)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Validation Accuracy</div>
                <div style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--accent-emerald)' }}>
                  {trainResult.final_val_acc}%
                </div>
              </div>

              <div style={{ padding: '14px', background: 'rgba(99, 102, 241, 0.1)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(99, 102, 241, 0.3)' }}>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Final Training Loss</div>
                <div style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--accent-indigo)' }}>
                  {trainResult.history[trainResult.history.length - 1].train_loss}
                </div>
              </div>
            </div>

            {/* Curves */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '16px' }}>
              <TrainingCurvesChart
                title="Training vs Validation Loss"
                data={trainResult.history}
                series={[
                  { key: 'train_loss', name: 'Train Loss', color: '#6366f1' },
                  { key: 'val_loss', name: 'Val Loss', color: '#ec4899', strokeDash: '4 4' },
                ]}
                yLabel="Loss"
              />

              <TrainingCurvesChart
                title="Training vs Validation Accuracy (%)"
                data={trainResult.history}
                series={[
                  { key: 'train_acc', name: 'Train Acc %', color: '#10b981' },
                  { key: 'val_acc', name: 'Val Acc %', color: '#06b6d4', strokeDash: '4 4' },
                ]}
                yLabel="Accuracy %"
              />
            </div>

            {/* Architecture Table */}
            <div style={{ background: 'rgba(0, 0, 0, 0.35)', borderRadius: 'var(--radius-md)', padding: '16px', border: '1px solid var(--border-subtle)' }}>
              <h4 style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '10px' }}>
                CNN Architecture Layer Specifications:
              </h4>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', fontSize: '0.82rem', borderCollapse: 'collapse', textAlign: 'left' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}>
                      <th style={{ padding: '8px' }}>Layer</th>
                      <th style={{ padding: '8px' }}>Output Dimensions</th>
                      <th style={{ padding: '8px' }}>Parameters</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trainResult.architecture_summary.map((row: any, i: number) => (
                      <tr key={i} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.04)' }}>
                        <td style={{ padding: '8px', color: 'var(--text-primary)', fontWeight: 500 }}>{row.layer}</td>
                        <td style={{ padding: '8px', color: 'var(--accent-cyan)', fontFamily: 'Fira Code' }}>{row.shape}</td>
                        <td style={{ padding: '8px', color: 'var(--text-secondary)' }}>{row.params.toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* SECTION 3: Interactive Sample & Feature Map Inspector */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Eye size={18} color="var(--accent-cyan)" />
          3. Live Inference & Intermediate Feature Map Extractor
        </h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
          Select a test digit sample to see how the trained CNN processes it and examine the actual feature maps produced by Conv1 (14x14) and Conv2 (7x7) channels.
        </p>

        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '20px' }}>
          {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((idx) => (
            <button
              key={idx}
              className="btn-secondary"
              style={{
                borderColor: sampleIdx === idx ? 'var(--accent-cyan)' : 'var(--border-subtle)',
                backgroundColor: sampleIdx === idx ? 'rgba(6, 182, 212, 0.15)' : 'transparent',
                padding: '6px 14px',
                fontSize: '0.82rem',
              }}
              onClick={() => handleInspectSample(idx)}
              disabled={sampleLoading}
            >
              Test Digit #{idx}
            </button>
          ))}
        </div>

        {sampleData && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '20px', alignItems: 'start' }}>
            {/* Input Image + Prediction */}
            <div style={{ background: 'rgba(0, 0, 0, 0.35)', padding: '16px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)', textAlign: 'center' }}>
              <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '8px' }}>
                True Label: <strong>{sampleData.true_label}</strong>
              </div>
              <div
                className="pixel-grid"
                style={{ gridTemplateColumns: 'repeat(28, 1fr)', width: '140px', margin: '0 auto 12px' }}
              >
                {sampleData.input_image.map((row: any, r: number) =>
                  row.map((val: number, c: number) => (
                    <div
                      key={`test-in-${r}-${c}`}
                      className="pixel-cell"
                      style={{ backgroundColor: `rgba(255, 255, 255, ${Math.max(0, val)})` }}
                    />
                  ))
                )}
              </div>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, color: sampleData.predicted_label === sampleData.true_label ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
                Predicted: Digit {sampleData.predicted_label}
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                Confidence: {sampleData.confidence}%
              </div>
            </div>

            {/* Conv1 Channels */}
            <div style={{ background: 'rgba(0, 0, 0, 0.35)', padding: '16px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--accent-cyan)', marginBottom: '8px' }}>
                Conv1 Layer Activations (14x14)
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px' }}>
                {sampleData.conv1_feature_maps.map((f: any) => (
                  <div key={f.channel_id} style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Ch #{f.channel_id}</div>
                    <div
                      className="pixel-grid"
                      style={{ gridTemplateColumns: 'repeat(14, 1fr)', width: '70px', margin: '4px auto' }}
                    >
                      {f.matrix.map((row: any, r: number) =>
                        row.map((val: number, c: number) => (
                          <div
                            key={`f1-${f.channel_id}-${r}-${c}`}
                            className="pixel-cell"
                            style={{ backgroundColor: `rgba(6, 182, 212, ${Math.min(1, val * 0.8)})` }}
                          />
                        ))
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Conv2 Channels */}
            <div style={{ background: 'rgba(0, 0, 0, 0.35)', padding: '16px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--accent-violet)', marginBottom: '8px' }}>
                Conv2 Layer Activations (7x7)
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px' }}>
                {sampleData.conv2_feature_maps.map((f: any) => (
                  <div key={f.channel_id} style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Ch #{f.channel_id}</div>
                    <div
                      className="pixel-grid"
                      style={{ gridTemplateColumns: 'repeat(7, 1fr)', width: '56px', margin: '4px auto' }}
                    >
                      {f.matrix.map((row: any, r: number) =>
                        row.map((val: number, c: number) => (
                          <div
                            key={`f2-${f.channel_id}-${r}-${c}`}
                            className="pixel-cell"
                            style={{ backgroundColor: `rgba(139, 92, 246, ${Math.min(1, val * 0.8)})` }}
                          />
                        ))
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* SECTION 4: Conceptual Quiz */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '6px' }}>
          4. Concept Check & Quiz
        </h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '20px' }}>
          Test your understanding of CNN operations and feature extraction.
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
                      bgColor = 'rgba(6, 182, 212, 0.15)';
                      borderColor = 'var(--accent-cyan)';
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
                markModuleComplete(2);
              }}
              disabled={Object.keys(quizAnswers).length < quizQuestions.length}
              style={{ alignSelf: 'flex-start' }}
            >
              Submit Concept Quiz
            </button>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--accent-emerald)', fontWeight: 600 }}>
              <CheckCircle2 size={18} /> Module 2 Concept Quiz Completed!
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

import React, { useState } from 'react';
import { useLab } from '../context/LabContext';
import { api } from '../services/api';
import { TrainingCurvesChart } from '../components/Visualizers/TrainingCurvesChart';
import {
  Layers,
  Play,
  RotateCw,
  GitCommit,
  CheckCircle2,
  HelpCircle,
  TrendingDown,
  Terminal,
  Activity
} from 'lucide-react';

export const Module1PyTorchBasics: React.FC = () => {
  const { markModuleComplete, recordPracticalRun } = useLab();

  // Part 1: Tensor Ops State
  const [opType, setOpType] = useState<string>('matmul');
  const [activation, setActivation] = useState<string>('relu');
  const [tensorOpResult, setTensorOpResult] = useState<any>(null);
  const [tensorLoading, setTensorLoading] = useState<boolean>(false);

  // Part 2: Autograd State
  const [formula, setFormula] = useState<string>('3*x**2 + 2*x + 1');
  const [xVal, setXVal] = useState<number>(2.0);
  const [autogradResult, setAutogradResult] = useState<any>(null);
  const [autogradLoading, setAutogradLoading] = useState<boolean>(false);

  // Part 3: Real Training Loop State
  const [lr, setLr] = useState<number>(0.05);
  const [optimizerName, setOptimizerName] = useState<string>('Adam');
  const [epochs, setEpochs] = useState<number>(30);
  const [hiddenDim, setHiddenDim] = useState<number>(16);
  const [trainingResult, setTrainingResult] = useState<any>(null);
  const [trainingLoading, setTrainingLoading] = useState<boolean>(false);

  // Part 4: Quiz State
  const [quizAnswers, setQuizAnswers] = useState<Record<number, number>>({});
  const [quizSubmitted, setQuizSubmitted] = useState<boolean>(false);

  const handleRunTensorOp = async () => {
    setTensorLoading(true);
    try {
      const res = await api.runTensorOp({
        op_type: opType,
        shape_a: [2, 3],
        shape_b: [3, 2],
        activation,
      });
      setTensorOpResult(res);
      recordPracticalRun(1);
    } catch (err: any) {
      alert(err.message);
    } finally {
      setTensorLoading(false);
    }
  };

  const handleRunAutograd = async () => {
    setAutogradLoading(true);
    try {
      const res = await api.runAutograd({ formula, x_val: xVal });
      setAutogradResult(res);
      recordPracticalRun(1);
    } catch (err: any) {
      alert(err.message);
    } finally {
      setAutogradLoading(false);
    }
  };

  const handleTrainMLP = async () => {
    setTrainingLoading(true);
    try {
      const res = await api.trainMlp({
        learning_rate: lr,
        optimizer_name: optimizerName,
        epochs,
        hidden_dim: hiddenDim,
      });
      setTrainingResult(res);
      recordPracticalRun(1);
      markModuleComplete(1);
    } catch (err: any) {
      alert(err.message);
    } finally {
      setTrainingLoading(false);
    }
  };

  const quizQuestions = [
    {
      q: 'Why is optimizer.zero_grad() called before loss.backward() in a PyTorch training loop?',
      options: [
        'To reset the model weights back to random initialization',
        'Because PyTorch accumulates gradients in .grad buffers by default across iterations',
        'To speed up CPU multi-threading',
        'To calculate the training loss value',
      ],
      correct: 1,
      explanation: 'In PyTorch, gradients are accumulated by default (convenient for RNNs and multi-task learning). If you do not call zero_grad(), new gradients get added to old gradients!',
    },
    {
      q: 'What does the autograd computational graph do during loss.backward()?',
      options: [
        'Performs a forward pass through all layers again',
        'Traverses the directed graph backwards, applying the chain rule to compute dLoss/dWeight',
        'Deletes the weights from RAM',
        'Exports the network to an ONNX model',
      ],
      correct: 1,
      explanation: 'Autograd dynamically records forward operations into a Directed Acyclic Graph (DAG) and traverses it in reverse to calculate exact gradients via the chain rule.',
    },
    {
      q: 'Which activation function is defined as f(x) = max(0, x)?',
      options: ['Sigmoid', 'Softmax', 'ReLU (Rectified Linear Unit)', 'Hyperbolic Tangent (tanh)'],
      correct: 2,
      explanation: 'ReLU zeroes out negative values and preserves positive values, preventing gradient saturation and training much faster than sigmoid.',
    },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px', maxWidth: '1100px', margin: '0 auto' }}>
      {/* Title & Introduction */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
          <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: 'rgba(99, 102, 241, 0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Layers size={20} color="var(--accent-indigo)" />
          </div>
          <span className="badge badge-indigo">Module 1</span>
          <span className="badge badge-cyan">Core Foundation</span>
        </div>
        <h2 style={{ fontSize: '2rem', fontWeight: 800, color: '#ffffff', marginBottom: '8px' }}>
          PyTorch Fundamentals: Tensors, Autograd & The Training Loop
        </h2>
        <p style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
          PyTorch is built on two core concepts: <strong>n-dimensional Tensors</strong> with GPU/CPU acceleration, and <strong>Autograd</strong> (an automatic differentiation engine). Learn how forward evaluations construct dynamic computation graphs, how the chain rule backpropagates gradients, and how optimizers update weights in a canonical training loop.
        </p>
      </div>

      {/* SECTION 1: Interactive Tensors Lab */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '6px' }}>
          1. Interactive Tensor Operations
        </h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
          Tensors are multi-dimensional arrays similar to NumPy ndarrays, but with seamless computational graph tracking and hardware acceleration.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '16px' }}>
          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
              Operation:
            </label>
            <select className="select-field" value={opType} onChange={(e) => setOpType(e.target.value)}>
              <option value="matmul">Matrix Multiplication (@)</option>
              <option value="add">Element-wise Addition (+)</option>
              <option value="reshape">Reshape / View (Flatten)</option>
              <option value="transpose">Transpose Dimensions</option>
              <option value="activation">Non-linear Activation</option>
            </select>
          </div>

          {opType === 'activation' && (
            <div>
              <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
                Activation Function:
              </label>
              <select className="select-field" value={activation} onChange={(e) => setActivation(e.target.value)}>
                <option value="relu">ReLU: max(0, x)</option>
                <option value="sigmoid">Sigmoid: 1 / (1 + e^-x)</option>
                <option value="softmax">Softmax: exp(x) / sum(exp)</option>
              </select>
            </div>
          )}

          <div style={{ display: 'flex', alignItems: 'flex-end' }}>
            <button className="btn-primary" onClick={handleRunTensorOp} disabled={tensorLoading} style={{ width: '100%' }}>
              <Play size={15} /> {tensorLoading ? 'Running...' : 'Run Tensor Op'}
            </button>
          </div>
        </div>

        {tensorOpResult && (
          <div style={{ background: 'rgba(0, 0, 0, 0.4)', borderRadius: 'var(--radius-md)', padding: '16px', border: '1px solid var(--border-subtle)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
              <strong style={{ color: 'var(--accent-cyan)', fontSize: '0.95rem' }}>{tensorOpResult.operation_name}</strong>
              <span className="badge badge-indigo">Shape: {JSON.stringify(tensorOpResult.result_tensor.shape)}</span>
            </div>
            <div className="code-box" style={{ marginBottom: '12px', color: '#a5b4fc' }}>
              Formula: {tensorOpResult.formula_latex}
            </div>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '12px' }}>
              {tensorOpResult.description}
            </p>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Result Sample Values (first rows):
              <pre style={{ color: '#67e8f9', marginTop: '4px' }}>
                {JSON.stringify(tensorOpResult.result_tensor.values, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </div>

      {/* SECTION 2: Autograd & Computation Graph */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '6px' }}>
          2. Autograd & Automatic Differentiation Engine
        </h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
          Enter any mathematical expression in terms of <code>x</code>. PyTorch creates a leaf variable with <code>requires_grad=True</code>, traces forward evaluation, and computes <code>dy/dx</code> via <code>y.backward()</code>.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr', gap: '16px', marginBottom: '16px' }}>
          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
              Mathematical Formula y = f(x):
            </label>
            <input
              type="text"
              className="input-field"
              value={formula}
              onChange={(e) => setFormula(e.target.value)}
              placeholder="3*x**2 + 2*x + 1"
            />
          </div>

          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
              Input x Value:
            </label>
            <input
              type="number"
              step="0.5"
              className="input-field"
              value={xVal}
              onChange={(e) => setXVal(parseFloat(e.target.value) || 0)}
            />
          </div>

          <div style={{ display: 'flex', alignItems: 'flex-end' }}>
            <button className="btn-primary" onClick={handleRunAutograd} disabled={autogradLoading} style={{ width: '100%' }}>
              <GitCommit size={15} /> {autogradLoading ? 'Evaluating...' : 'Compute Gradient'}
            </button>
          </div>
        </div>

        {autogradResult && (
          <div style={{ background: 'rgba(0, 0, 0, 0.4)', borderRadius: 'var(--radius-md)', padding: '16px', border: '1px solid var(--border-subtle)' }}>
            <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap', marginBottom: '16px' }}>
              <div style={{ padding: '10px 16px', background: 'rgba(99, 102, 241, 0.1)', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Forward Evaluation y</div>
                <div style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--text-primary)' }}>{autogradResult.y_forward}</div>
              </div>

              <div style={{ padding: '10px 16px', background: 'rgba(6, 182, 212, 0.1)', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Autograd dy/dx (x.grad)</div>
                <div style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--accent-cyan)' }}>{autogradResult.gradient_dy_dx}</div>
              </div>

              <div style={{ padding: '10px 16px', background: 'rgba(16, 185, 129, 0.1)', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Analytical Calculus Check</div>
                <div style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--accent-emerald)' }}>{autogradResult.analytical_expected}</div>
              </div>
            </div>

            {/* Step-by-step DAG nodes */}
            <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '8px' }}>
              Computation Graph Traversal:
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {autogradResult.computation_graph_steps.map((st: any) => (
                <div key={st.step} style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '8px 12px', background: 'rgba(255, 255, 255, 0.03)', borderRadius: '6px', fontSize: '0.82rem' }}>
                  <span className="badge badge-indigo">Step {st.step}</span>
                  <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{st.node}:</span>
                  <span style={{ color: 'var(--text-secondary)', fontFamily: 'Fira Code' }}>{st.operation}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* SECTION 3: Real Model Training Loop Workbench */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '6px' }}>
          3. Real PyTorch Training Loop on Non-Linear Circles
        </h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
          Train a 2-layer MLP with PyTorch <code>nn.Module</code> on a non-linear concentric circles classification problem. Observe loss and accuracy convergence in real time.
        </p>

        {/* Hyperparameter Controls */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '16px', marginBottom: '20px' }}>
          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
              Learning Rate: {lr}
            </label>
            <input
              type="range"
              min="0.005"
              max="0.2"
              step="0.005"
              value={lr}
              onChange={(e) => setLr(parseFloat(e.target.value))}
              style={{ width: '100%' }}
            />
          </div>

          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
              Optimizer:
            </label>
            <select className="select-field" value={optimizerName} onChange={(e) => setOptimizerName(e.target.value)}>
              <option value="Adam">Adam (Adaptive Moment Estimation)</option>
              <option value="SGD">SGD (Stochastic Gradient Descent + Momentum)</option>
            </select>
          </div>

          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
              Epochs: {epochs}
            </label>
            <input
              type="range"
              min="10"
              max="60"
              step="5"
              value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value, 10))}
              style={{ width: '100%' }}
            />
          </div>

          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
              Hidden Neurons: {hiddenDim}
            </label>
            <select className="select-field" value={hiddenDim} onChange={(e) => setHiddenDim(parseInt(e.target.value, 10))}>
              <option value="8">8 Neurons</option>
              <option value="16">16 Neurons</option>
              <option value="32">32 Neurons</option>
            </select>
          </div>
        </div>

        <button className="btn-primary" onClick={handleTrainMLP} disabled={trainingLoading} style={{ marginBottom: '20px' }}>
          <Play size={16} /> {trainingLoading ? 'Training PyTorch Model on CPU...' : 'Execute Training Loop'}
        </button>

        {trainingResult && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {/* Final Metrics Banner */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
              <div style={{ padding: '14px', background: 'rgba(16, 185, 129, 0.1)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Final Accuracy</div>
                <div style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--accent-emerald)' }}>
                  {trainingResult.final_accuracy}%
                </div>
              </div>

              <div style={{ padding: '14px', background: 'rgba(99, 102, 241, 0.1)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(99, 102, 241, 0.3)' }}>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Final Cross-Entropy Loss</div>
                <div style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--accent-indigo)' }}>
                  {trainingResult.final_loss}
                </div>
              </div>
            </div>

            {/* Real Training Curves Chart */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '16px' }}>
              <TrainingCurvesChart
                title="Training Loss Convergence"
                data={trainingResult.history}
                series={[{ key: 'loss', name: 'Cross-Entropy Loss', color: '#6366f1' }]}
                yLabel="Loss"
              />

              <TrainingCurvesChart
                title="Classification Accuracy (%)"
                data={trainingResult.history}
                series={[{ key: 'accuracy', name: 'Training Accuracy', color: '#10b981' }]}
                yLabel="Accuracy %"
              />
            </div>

            {/* Step-by-Step Training Loop Walkthrough */}
            <div style={{ background: 'rgba(0, 0, 0, 0.4)', borderRadius: 'var(--radius-md)', padding: '18px', border: '1px solid var(--border-subtle)' }}>
              <h4 style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '10px' }}>
                PyTorch 5-Step Canonical Training Loop In Code:
              </h4>
              <div className="code-box" style={{ color: '#e2e8f0', fontSize: '0.82rem' }}>
{`for epoch in range(epochs):
    # 1. Clear previous accumulated gradients
    optimizer.zero_grad()
    
    # 2. Forward pass through neural network layers
    predictions = model(X_train)
    
    # 3. Compute error using loss function
    loss = criterion(predictions, y_train)
    
    # 4. Backward pass (autograd chain rule: dL/dw)
    loss.backward()
    
    # 5. Optimizer parameter update (w = w - lr * grad)
    optimizer.step()`}
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
          Test your understanding of PyTorch basics to complete this module.
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
                markModuleComplete(1);
              }}
              disabled={Object.keys(quizAnswers).length < quizQuestions.length}
              style={{ alignSelf: 'flex-start' }}
            >
              Submit Concept Quiz
            </button>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--accent-emerald)', fontWeight: 600 }}>
              <CheckCircle2 size={18} /> Module 1 Concept Quiz Completed!
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

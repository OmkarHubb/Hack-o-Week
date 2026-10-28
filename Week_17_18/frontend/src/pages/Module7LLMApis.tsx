import React, { useState } from 'react';
import { useLab } from '../context/LabContext';
import { api } from '../services/api';
import {
  Bot,
  Play,
  CheckCircle2,
  ShieldCheck,
  Lock,
  Eye,
  EyeOff,
  Cpu,
  Zap,
  Sliders,
  Terminal,
  Server
} from 'lucide-react';

export const Module7LLMApis: React.FC = () => {
  const { markModuleComplete, recordPracticalRun } = useLab();

  const [provider, setProvider] = useState<string>('open_model');
  const [apiKey, setApiKey] = useState<string>('');
  const [showKey, setShowKey] = useState<boolean>(false);
  const [ollamaEndpoint, setOllamaEndpoint] = useState<string>('http://localhost:11434');
  const [modelName, setModelName] = useState<string>('gpt-4o-mini');

  const [systemPrompt, setSystemPrompt] = useState<string>(
    'You are a senior deep learning research scientist. Explain concepts clearly, rigorously, and concisely.'
  );
  const [prompt, setPrompt] = useState<string>(
    'Explain how backpropagation works with the calculus chain rule in two concise, rigorous paragraphs.'
  );
  const [temperature, setTemperature] = useState<number>(0.7);
  const [maxTokens, setMaxTokens] = useState<number>(300);

  const [generationResult, setGenerationResult] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);

  // Quiz state
  const [quizAnswers, setQuizAnswers] = useState<Record<number, number>>({});
  const [quizSubmitted, setQuizSubmitted] = useState<boolean>(false);

  const handleGenerate = async () => {
    setLoading(true);
    try {
      const res = await api.generateLlm({
        provider,
        prompt,
        api_key: apiKey.trim() || undefined,
        system_prompt: systemPrompt,
        temperature,
        max_tokens: maxTokens,
        ollama_endpoint: ollamaEndpoint,
        model_name: modelName,
      });

      if (res.status === 'error') {
        alert(res.error || 'Generation failed');
      } else {
        setGenerationResult(res);
        recordPracticalRun(7);
        markModuleComplete(7);
      }
    } catch (err: any) {
      alert(err.message);
    } finally {
      setLoading(false);
    }
  };

  const promptTemplates = [
    {
      title: 'Backprop & Chain Rule',
      p: 'Explain how backpropagation works with the calculus chain rule in two concise, rigorous paragraphs.',
    },
    {
      title: 'CNN vs Transformer',
      p: 'Compare the inductive biases of Convolutional Neural Networks (locality, translation invariance) versus Transformers (global attention). When is each preferred?',
    },
    {
      title: 'Vanishing Gradients Fix',
      p: 'Explain the three most impactful techniques used in modern deep learning to resolve the vanishing gradient problem (Residual connections, Batch/Layer Norm, ReLU).',
    },
  ];

  const quizQuestions = [
    {
      q: 'How does the Temperature (T) parameter affect next-token probability sampling?',
      options: [
        'Higher temperature (T > 1.0) flattens logits, increasing randomness and entropy; lower temperature (T -> 0) sharpens probabilities toward greedy argmax sampling',
        'It increases the GPU hardware temperature to speed up inference',
        'It permanently modifies the neural weights of the foundation model',
        'It translates the generated text into binary',
      ],
      correct: 0,
      explanation: 'Temperature scales the raw logits via z_i / T before the softmax function. T=0 collapses into greedy selection of the highest probability token, while higher T yields diverse, creative text.',
    },
    {
      q: 'Why should API keys NEVER be committed to Git or stored in client-side bundles?',
      options: [
        'Because they expire automatically every 10 seconds',
        'Anyone with access to the repository or web bundle can steal your credentials, incurring unauthorized API billing and data breaches',
        'PyTorch refuses to compile if API keys are present',
        'API keys only work on Linux servers',
      ],
      correct: 1,
      explanation: 'API keys represent your billing and authentication identity. Keys must always remain ephemeral in memory or protected behind backend server proxies, never logged to disk or exposed in frontend code.',
    },
    {
      q: 'What is the main distinction between Open-Weights LLMs (e.g. Llama 3, Mistral) and Proprietary Closed APIs (e.g. GPT-4o)?',
      options: [
        'Open-weights models can be hosted, inspected, and run entirely offline/locally without vendor dependency or data sharing, while closed models require HTTP requests to proprietary cloud servers',
        'Open-weights models only support 5 words per sentence',
        'Closed APIs are always free of charge',
        'There is no difference between them',
      ],
      correct: 0,
      explanation: 'Open models provide accessible weight matrices allowing private local hosting (e.g. via Ollama/vLLM) with total data privacy, whereas closed APIs are accessible solely as cloud web services.',
    },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px', maxWidth: '1100px', margin: '0 auto' }}>
      {/* Title */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
          <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: 'rgba(16, 185, 129, 0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Bot size={20} color="var(--accent-emerald)" />
          </div>
          <span className="badge badge-emerald">Module 7</span>
          <span className="badge badge-indigo">Large Language Models</span>
        </div>
        <h2 style={{ fontSize: '2rem', fontWeight: 800, color: '#ffffff', marginBottom: '8px' }}>
          LLM APIs & Open Models: Generation, Gating & Secure Execution
        </h2>
        <p style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
          Explore modern Large Language Model generation mechanics (sampling temperature, top-p, context windows). Seamlessly toggle between <strong>Zero-Config Local Open Models</strong>, local <strong>Ollama</strong> instances, and commercial <strong>OpenAI / Anthropic</strong> APIs with strict ephemeral key security.
        </p>
      </div>

      {/* SECTION 1: Provider & Security Configuration */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <ShieldCheck size={18} color="var(--accent-emerald)" />
            1. Model Engine & Security Settings
          </h3>
          <span className="badge badge-emerald" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Lock size={12} /> Ephemeral Memory Only
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '16px', marginBottom: '20px' }}>
          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
              Execution Provider:
            </label>
            <select className="select-field" value={provider} onChange={(e) => setProvider(e.target.value)}>
              <option value="open_model">Built-in Local Neural Engine (No Key Needed)</option>
              <option value="ollama">Local Ollama (llama3, mistral, phi3)</option>
              <option value="openai">OpenAI API (GPT-4o-mini, GPT-4o)</option>
              <option value="anthropic">Anthropic API (Claude 3.5 Sonnet)</option>
            </select>
          </div>

          {provider === 'ollama' && (
            <div>
              <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
                Ollama Endpoint & Model:
              </label>
              <div style={{ display: 'flex', gap: '8px' }}>
                <input
                  type="text"
                  className="input-field"
                  value={ollamaEndpoint}
                  onChange={(e) => setOllamaEndpoint(e.target.value)}
                  placeholder="http://localhost:11434"
                />
                <input
                  type="text"
                  className="input-field"
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  placeholder="llama3"
                  style={{ maxWidth: '120px' }}
                />
              </div>
            </div>
          )}

          {(provider === 'openai' || provider === 'anthropic') && (
            <div>
              <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
                {provider === 'openai' ? 'OpenAI Secret Key (sk-...)' : 'Anthropic API Key (sk-ant-...)'}:
              </label>
              <div style={{ position: 'relative' }}>
                <input
                  type={showKey ? 'text' : 'password'}
                  className="input-field"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder={provider === 'openai' ? 'sk-proj-...' : 'sk-ant-...'}
                  style={{ paddingRight: '40px', fontFamily: 'Fira Code' }}
                />
                <button
                  type="button"
                  onClick={() => setShowKey(!showKey)}
                  style={{
                    position: 'absolute',
                    right: '10px',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    background: 'none',
                    border: 'none',
                    color: 'var(--text-muted)',
                    cursor: 'pointer',
                  }}
                >
                  {showKey ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>
          )}
        </div>

        <div style={{ padding: '12px 16px', background: 'rgba(16, 185, 129, 0.08)', borderRadius: 'var(--radius-sm)', borderLeft: '3px solid var(--accent-emerald)', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
          <strong>Zero Disk Footprint Guarantee:</strong> API keys provided in this interface are held strictly in ephemeral RAM for the duration of the HTTP request. Keys are never written to disk, never committed, and never permanently logged.
        </div>
      </div>

      {/* SECTION 2: Interactive Prompt & Generation Workbench */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '6px' }}>
          2. Interactive Generation Workbench
        </h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
          Tune generative sampling parameters and test system-prompt steering on deep learning tasks.
        </p>

        {/* Quick Templates */}
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '16px' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', alignSelf: 'center' }}>
            Quick Prompts:
          </span>
          {promptTemplates.map((t, idx) => (
            <button
              key={idx}
              className="btn-secondary"
              style={{ fontSize: '0.78rem', padding: '5px 12px' }}
              onClick={() => setPrompt(t.p)}
            >
              {t.title}
            </button>
          ))}
        </div>

        {/* System Prompt & User Prompt */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', marginBottom: '20px' }}>
          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
              System Prompt (Directive Conditioning):
            </label>
            <input
              type="text"
              className="input-field"
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
            />
          </div>

          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
              User Prompt:
            </label>
            <textarea
              className="input-field"
              rows={3}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
            />
          </div>
        </div>

        {/* Hyperparameter Controls */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '20px', marginBottom: '20px' }}>
          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
              <span>Sampling Temperature:</span>
              <strong style={{ color: 'var(--accent-cyan)' }}>{temperature}</strong>
            </label>
            <input
              type="range"
              min="0.0"
              max="1.5"
              step="0.05"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              style={{ width: '100%' }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '2px' }}>
              <span>0.0 (Greedy / Deterministic)</span>
              <span>1.5 (High Entropy / Creative)</span>
            </div>
          </div>

          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
              <span>Max Generated Tokens:</span>
              <strong style={{ color: 'var(--accent-indigo)' }}>{maxTokens}</strong>
            </label>
            <input
              type="range"
              min="64"
              max="600"
              step="32"
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value, 10))}
              style={{ width: '100%' }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '2px' }}>
              <span>64 Tokens (~40 words)</span>
              <span>600 Tokens (~400 words)</span>
            </div>
          </div>
        </div>

        <button className="btn-primary" onClick={handleGenerate} disabled={loading} style={{ marginBottom: '20px' }}>
          <Play size={16} /> {loading ? 'Querying Engine & Sampling Tokens...' : 'Generate Text'}
        </button>

        {/* Result Container */}
        {generationResult && (
          <div style={{ background: 'rgba(0, 0, 0, 0.4)', padding: '20px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', flexWrap: 'wrap', gap: '10px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span className="badge badge-emerald">{generationResult.provider}</span>
                <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>Model: {generationResult.model}</span>
              </div>
              <div style={{ display: 'flex', gap: '14px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                <span>Latency: <strong style={{ color: '#fff' }}>{generationResult.duration_ms} ms</strong></span>
                <span>Prompt: <strong style={{ color: 'var(--accent-cyan)' }}>{generationResult.prompt_tokens} tokens</strong></span>
                <span>Output: <strong style={{ color: 'var(--accent-emerald)' }}>{generationResult.completion_tokens} tokens</strong></span>
              </div>
            </div>

            <div style={{ padding: '16px', background: 'rgba(255, 255, 255, 0.03)', borderRadius: '8px', borderLeft: '3px solid var(--accent-emerald)', fontSize: '0.92rem', color: '#f1f5f9', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>
              {generationResult.generated_text}
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
          Test your understanding of LLM sampling, temperature, and key security.
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
                      bgColor = 'rgba(16, 185, 129, 0.15)';
                      borderColor = 'var(--accent-emerald)';
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
                markModuleComplete(7);
              }}
              disabled={Object.keys(quizAnswers).length < quizQuestions.length}
              style={{ alignSelf: 'flex-start' }}
            >
              Submit Concept Quiz
            </button>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--accent-emerald)', fontWeight: 600 }}>
              <CheckCircle2 size={18} /> Module 7 Concept Quiz Completed! You have finished all 7 modules!
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

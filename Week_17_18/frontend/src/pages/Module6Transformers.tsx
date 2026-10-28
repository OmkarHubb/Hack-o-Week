import React, { useState } from 'react';
import { useLab } from '../context/LabContext';
import { api } from '../services/api';
import {
  Cpu,
  Play,
  CheckCircle2,
  Sparkles,
  Search,
  MessageSquare,
  FileText,
  Smile,
  Hash,
  Terminal
} from 'lucide-react';

export const Module6Transformers: React.FC = () => {
  const { markModuleComplete, recordPracticalRun } = useLab();

  // Part 1: Tokenizer State
  const [tokenText, setTokenText] = useState<string>('Hugging Face Transformers make deep learning intuitive and accessible!');
  const [tokenResult, setTokenResult] = useState<any>(null);
  const [tokenLoading, setTokenLoading] = useState<boolean>(false);

  // Part 2: Transformer Task State
  const [activeTask, setActiveTask] = useState<'sentiment' | 'summarization' | 'qa'>('sentiment');
  const [inputText, setInputText] = useState<string>('I absolutely love experimenting with neural networks and PyTorch, the interactive results are incredible!');
  const [contextText, setContextText] = useState<string>(
    'PyTorch was created by Meta AI (formerly Facebook AI Research) in 2016. It pioneered dynamic computational graphs and became the primary framework for contemporary generative AI and foundation models.'
  );
  const [questionText, setQuestionText] = useState<string>('Who created PyTorch and when was it released?');
  const [taskResult, setTaskResult] = useState<any>(null);
  const [taskLoading, setTaskLoading] = useState<boolean>(false);

  // Part 3: Quiz State
  const [quizAnswers, setQuizAnswers] = useState<Record<number, number>>({});
  const [quizSubmitted, setQuizSubmitted] = useState<boolean>(false);

  const handleTokenize = async () => {
    setTokenLoading(true);
    try {
      const res = await api.tokenizeText({ text: tokenText });
      setTokenResult(res);
      recordPracticalRun(6);
    } catch (err: any) {
      alert(err.message);
    } finally {
      setTokenLoading(false);
    }
  };

  const handleRunTask = async () => {
    setTaskLoading(true);
    try {
      const res = await api.runTransformerTask({
        task: activeTask,
        text: inputText,
        context: activeTask === 'qa' ? contextText : undefined,
        question: activeTask === 'qa' ? questionText : undefined,
      });
      setTaskResult(res);
      recordPracticalRun(6);
      markModuleComplete(6);
    } catch (err: any) {
      alert(err.message);
    } finally {
      setTaskLoading(false);
    }
  };

  const quizQuestions = [
    {
      q: 'Why does the Transformer Self-Attention formula scale dot products by 1 / sqrt(d_k)?',
      options: [
        'To reduce RAM usage by half',
        'When d_k is large, dot products grow large in magnitude, pushing softmax into regions with tiny gradients (gradient vanishing)',
        'It converts English words into French',
        'It is an arbitrary constant with no mathematical reason',
      ],
      correct: 1,
      explanation: 'For large vector dimensions d_k, the dot product variance is d_k. Dividing by sqrt(d_k) keeps the variance at 1, preventing the softmax function from saturating and killing backpropagation gradients.',
    },
    {
      q: 'What is the purpose of Subword Tokenization (e.g. WordPiece / BPE)?',
      options: [
        'It translates text into Morse code',
        'It solves the Out-Of-Vocabulary (OOV) problem by decomposing rare or unseen words into frequent subword fragments (e.g. "transformers" -> "transform" + "##ers")',
        'It permanently removes punctuation marks from sentences',
        'It limits vocabularies to only 26 English letters',
      ],
      correct: 1,
      explanation: 'Subword tokenization strikes an optimal balance between word-level (huge vocabulary, fails on typos/rare words) and character-level (very long sequences, weaker semantic density).',
    },
    {
      q: 'In BERT, what is the role of the special [CLS] token placed at index 0?',
      options: [
        'It closes the Python interpreter',
        'Its final hidden state acts as an aggregate pooled representation for sentence-level classification (e.g. sentiment, spam)',
        'It deletes words with negative sentiment',
        'It indicates the end of the text file',
      ],
      correct: 1,
      explanation: 'The [CLS] (Classification) token attends to all other tokens across all self-attention layers. Its final vector is passed into classification heads for sequence-level tasks.',
    },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px', maxWidth: '1100px', margin: '0 auto' }}>
      {/* Title */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
          <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: 'rgba(6, 182, 212, 0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Cpu size={20} color="var(--accent-cyan)" />
          </div>
          <span className="badge badge-cyan">Module 6</span>
          <span className="badge badge-indigo">Attention & Transformers</span>
        </div>
        <h2 style={{ fontSize: '2rem', fontWeight: 800, color: '#ffffff', marginBottom: '8px' }}>
          Hugging Face Transformers: Tokenization & Pretrained NLP Pipelines
        </h2>
        <p style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
          Transformers replaced recurrent networks by processing sequences entirely in parallel using <strong>Self-Attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V</strong>. Explore subword tokenization and run real transformer inference pipelines for sentiment analysis, text summarization, and question answering.
        </p>
      </div>

      {/* SECTION 1: Interactive Tokenizer Explorer */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Hash size={18} color="var(--accent-cyan)" />
          1. Interactive Subword Tokenizer Playground
        </h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
          Type any sentence to see how BERT/RoBERTa tokenizers decompose text into subwords, assign numerical vocabulary IDs, inject <code>[CLS]</code>/<code>[SEP]</code> tokens, and construct attention masks.
        </p>

        <div style={{ display: 'flex', gap: '12px', marginBottom: '16px' }}>
          <input
            type="text"
            className="input-field"
            value={tokenText}
            onChange={(e) => setTokenText(e.target.value)}
            placeholder="Type any sentence to tokenize..."
          />
          <button className="btn-primary" onClick={handleTokenize} disabled={tokenLoading} style={{ flexShrink: 0 }}>
            <Play size={15} /> {tokenLoading ? 'Tokenizing...' : 'Tokenize'}
          </button>
        </div>

        {tokenResult && (
          <div style={{ background: 'rgba(0, 0, 0, 0.35)', padding: '16px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '10px' }}>
              Subword Tokens Breakdown ({tokenResult.token_count} tokens):
            </div>

            {/* Visual Token Pills */}
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '16px' }}>
              {tokenResult.subword_breakdown.map((item: any, i: number) => {
                const isSpecial = item.token.startsWith('[');
                const isSubword = item.token.startsWith('##');
                return (
                  <div
                    key={i}
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      padding: '6px 10px',
                      borderRadius: '8px',
                      background: isSpecial ? 'rgba(99, 102, 241, 0.2)' : (isSubword ? 'rgba(245, 158, 11, 0.15)' : 'rgba(255, 255, 255, 0.05)'),
                      border: `1px solid ${isSpecial ? 'var(--accent-indigo)' : (isSubword ? 'var(--accent-amber)' : 'var(--border-subtle)')}`,
                    }}
                  >
                    <span style={{ fontSize: '0.85rem', fontWeight: 600, color: isSpecial ? 'var(--accent-cyan)' : 'var(--text-primary)', fontFamily: 'Fira Code' }}>
                      {item.token}
                    </span>
                    <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '2px', fontFamily: 'Fira Code' }}>
                      ID: {item.id}
                    </span>
                  </div>
                );
              })}
            </div>

            {/* Token IDs & Mask array */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '12px' }}>
              <div style={{ background: 'rgba(0, 0, 0, 0.4)', padding: '10px 14px', borderRadius: '6px', fontSize: '0.8rem' }}>
                <span style={{ color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Input IDs Vector:</span>
                <span style={{ color: '#a5b4fc', fontFamily: 'Fira Code', wordBreak: 'break-all' }}>
                  {JSON.stringify(tokenResult.token_ids)}
                </span>
              </div>

              <div style={{ background: 'rgba(0, 0, 0, 0.4)', padding: '10px 14px', borderRadius: '6px', fontSize: '0.8rem' }}>
                <span style={{ color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Attention Mask (1=Active, 0=Padding):</span>
                <span style={{ color: '#6ee7b7', fontFamily: 'Fira Code', wordBreak: 'break-all' }}>
                  {JSON.stringify(tokenResult.attention_mask)}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* SECTION 2: Real Pretrained Transformer Task Workbench */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '6px' }}>
          2. Hands-on Transformer Task Pipelines
        </h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '18px' }}>
          Select and test three fundamental transformer architectures: Sentiment Classification, Text Summarization, and Extractive Question Answering.
        </p>

        {/* Task Tabs */}
        <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
          <button
            className={`btn-secondary ${activeTask === 'sentiment' ? 'active-tab' : ''}`}
            style={{
              borderColor: activeTask === 'sentiment' ? 'var(--accent-cyan)' : 'var(--border-subtle)',
              backgroundColor: activeTask === 'sentiment' ? 'rgba(6, 182, 212, 0.15)' : 'transparent',
              padding: '8px 16px',
            }}
            onClick={() => {
              setActiveTask('sentiment');
              setInputText('I absolutely love experimenting with neural networks and PyTorch, the interactive results are incredible!');
            }}
          >
            <Smile size={16} /> Sentiment Analysis
          </button>

          <button
            className={`btn-secondary ${activeTask === 'summarization' ? 'active-tab' : ''}`}
            style={{
              borderColor: activeTask === 'summarization' ? 'var(--accent-indigo)' : 'var(--border-subtle)',
              backgroundColor: activeTask === 'summarization' ? 'rgba(99, 102, 241, 0.15)' : 'transparent',
              padding: '8px 16px',
            }}
            onClick={() => {
              setActiveTask('summarization');
              setInputText(
                'Deep learning has fundamentally transformed computer vision, natural language processing, and robotics over the last decade. Early methods relied heavily on manual feature engineering such as SIFT or edge histograms. With the advent of deep convolutional networks and self-attention transformers, neural models now learn hierarchical representations directly from raw data, drastically improving accuracy and generalization.'
              );
            }}
          >
            <FileText size={16} /> Text Summarization
          </button>

          <button
            className={`btn-secondary ${activeTask === 'qa' ? 'active-tab' : ''}`}
            style={{
              borderColor: activeTask === 'qa' ? 'var(--accent-violet)' : 'var(--border-subtle)',
              backgroundColor: activeTask === 'qa' ? 'rgba(139, 92, 246, 0.15)' : 'transparent',
              padding: '8px 16px',
            }}
            onClick={() => {
              setActiveTask('qa');
            }}
          >
            <Search size={16} /> Question Answering (Span Extraction)
          </button>
        </div>

        {/* Form Inputs based on Task */}
        {activeTask !== 'qa' ? (
          <div style={{ marginBottom: '16px' }}>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
              Input Text:
            </label>
            <textarea
              className="input-field"
              rows={4}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
            />
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', marginBottom: '16px' }}>
            <div>
              <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
                Context Paragraph:
              </label>
              <textarea
                className="input-field"
                rows={3}
                value={contextText}
                onChange={(e) => setContextText(e.target.value)}
              />
            </div>
            <div>
              <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
                Question:
              </label>
              <input
                type="text"
                className="input-field"
                value={questionText}
                onChange={(e) => setQuestionText(e.target.value)}
              />
            </div>
          </div>
        )}

        <button className="btn-primary" onClick={handleRunTask} disabled={taskLoading} style={{ marginBottom: '20px' }}>
          <Play size={15} /> {taskLoading ? 'Executing Transformer Pipeline...' : 'Run Transformer Pipeline'}
        </button>

        {/* Pipeline Output Display */}
        {taskResult && (
          <div style={{ background: 'rgba(0, 0, 0, 0.35)', padding: '18px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <strong style={{ color: 'var(--accent-cyan)', fontSize: '0.95rem' }}>
                Pipeline Output: {taskResult.pipeline_used}
              </strong>
              <span className="badge badge-emerald">Confidence: {taskResult.confidence}%</span>
            </div>

            {taskResult.task === 'sentiment' && (
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '14px' }}>
                  <span style={{ fontSize: '1.2rem', fontWeight: 800, color: taskResult.label === 'POSITIVE' ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
                    {taskResult.label}
                  </span>
                  <div style={{ display: 'flex', gap: '12px', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                    <span>Positive: {taskResult.probabilities.POSITIVE}%</span>
                    <span>Negative: {taskResult.probabilities.NEGATIVE}%</span>
                  </div>
                </div>

                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '8px' }}>
                  Attention Salience Heatmap (Highlighted words driving sentiment):
                </div>
                <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', padding: '10px', background: 'rgba(0, 0, 0, 0.3)', borderRadius: '6px' }}>
                  {taskResult.token_attributions.map((tok: any, i: number) => {
                    const isPositive = tok.attention_weight > 0.1;
                    const isNegative = tok.attention_weight < -0.1;
                    return (
                      <span
                        key={i}
                        style={{
                          padding: '2px 6px',
                          borderRadius: '4px',
                          background: isPositive ? 'rgba(16, 185, 129, 0.2)' : (isNegative ? 'rgba(244, 63, 94, 0.2)' : 'transparent'),
                          color: isPositive ? '#6ee7b7' : (isNegative ? '#fda4af' : 'var(--text-secondary)'),
                          fontWeight: isPositive || isNegative ? 600 : 400,
                          fontSize: '0.85rem',
                        }}
                      >
                        {tok.word}
                      </span>
                    );
                  })}
                </div>
              </div>
            )}

            {taskResult.task === 'summarization' && (
              <div>
                <div style={{ display: 'flex', gap: '16px', marginBottom: '10px', fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                  <span>Original: {taskResult.original_length_words} words</span>
                  <span>Summary: {taskResult.summary_length_words} words</span>
                  <span style={{ color: 'var(--accent-emerald)' }}>Compressed: {taskResult.compression_ratio}</span>
                </div>
                <div style={{ padding: '12px', background: 'rgba(99, 102, 241, 0.08)', borderRadius: '6px', borderLeft: '3px solid var(--accent-indigo)', fontSize: '0.9rem', color: '#ffffff', lineHeight: 1.6 }}>
                  "{taskResult.summary}"
                </div>
              </div>
            )}

            {taskResult.task === 'qa' && (
              <div>
                <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '6px' }}>
                  Extracted Span (Logits argmax):
                </div>
                <div style={{ padding: '12px', background: 'rgba(6, 182, 212, 0.1)', borderRadius: '6px', borderLeft: '3px solid var(--accent-cyan)', fontSize: '1.1rem', fontWeight: 700, color: 'var(--accent-cyan)' }}>
                  "{taskResult.answer}"
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* SECTION 3: Conceptual Quiz */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '6px' }}>
          3. Concept Check & Quiz
        </h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '20px' }}>
          Test your understanding of transformers and self-attention.
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
                markModuleComplete(6);
              }}
              disabled={Object.keys(quizAnswers).length < quizQuestions.length}
              style={{ alignSelf: 'flex-start' }}
            >
              Submit Concept Quiz
            </button>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--accent-emerald)', fontWeight: 600 }}>
              <CheckCircle2 size={18} /> Module 6 Concept Quiz Completed!
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

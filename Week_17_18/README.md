# Deep Learning Learning Lab

A complete, interactive educational web application and workbench covering 7 foundational Deep Learning topics with real PyTorch model training, interactive visualizers, parameter controls, and live metrics.

---

## Architecture Overview

- **Frontend:** React 18, TypeScript, Vite, Vanilla CSS design system with Glassmorphism and Lucide icons.
- **Backend:** FastAPI, Uvicorn, Python 3.11, PyTorch (CPU-accelerated), Torchvision, Hugging Face Transformers.
- **Location:** `Week_17_18/`

---

## Covered Modules & Practicals

1. **PyTorch Fundamentals (`m1_pytorch_basics.py`)**
   - Tensors, shapes, matrix multiplication, transpose, view/reshape, and activations (ReLU, Sigmoid, Softmax).
   - Dynamic Autograd DAG computation graphs and backpropagation gradient calculation.
   - Real 2-layer MLP training loop on non-linear concentric circles with live loss and accuracy curves.

2. **Convolutional Neural Networks (`m2_cnn.py`)**
   - Mathematical 2D convolution, filter kernels (Sobel X/Y, Sharpen, Gaussian Blur, Ridge, Custom 3x3), ReLU, and Max Pooling (2x2).
   - Fast LeNet/Digit CNN training on 28x28 images with training/validation curves.
   - Interactive feature map extractor visualizing intermediate Conv1 (14x14) and Conv2 (7x7) activations for any digit sample.

3. **RNNs, LSTMs, and GRUs (`m3_rnn_lstm_gru.py`)**
   - Sequence modeling theory, hidden states, vanishing gradients, and BPTT.
   - Interactive gate anatomy explorer (Forget, Input, Cell State, Output, Reset, Update).
   - Side-by-side benchmark training Vanilla RNN vs LSTM vs GRU on time-series forecasting with loss curves, runtimes, and multi-step forecasted waveforms vs ground truth.

4. **Regularization & Generalization (`m4_regularization.py`)**
   - Overfitting mechanisms and generalization gap ($Train_{acc} - Val_{acc}$).
   - 4-way ablation on high-capacity MLP: Baseline (overfitting divergence) vs Dropout ($p=0.4$) vs Batch Normalization vs Early Stopping (patience window & checkpoint restoration).

5. **Transfer Learning (`m5_transfer_learning.py`)**
   - Pretrained ResNet-18 visual feature hierarchy (Conv1 edges $\rightarrow$ Layer 4 semantics).
   - Feature Extraction (freezing backbone with $requires\_grad=False$, training only classification head, 2.5K params) vs Fine-Tuning (unfreezing Layer 4 + head, 8.4M params).
   - Training workbench with real metrics and decision guide matrix.

6. **Hugging Face Transformers (`m6_transformers.py`)**
   - Self-Attention mechanism $\text{Attention}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d_k}) V$.
   - Interactive Tokenizer playground: subword decomposition (WordPiece/BPE), token IDs, special tokens (`[CLS]`, `[SEP]`), and attention masks.
   - Real NLP pipelines: Sentiment Analysis (valence + attention attribution), Text Summarization, and Extractive Question Answering (span logits).

7. **LLM APIs & Open Models (`m7_llm_apis.py`)**
   - Generative token sampling, temperature ($T$), top-p, context windows, and prompt steering.
   - Multi-engine support: Built-in Local Open Model (zero configuration), local Ollama daemon (`http://localhost:11434`), OpenAI API, and Anthropic API.
   - Strict ephemeral security: API keys are handled purely in RAM for the request and never saved to disk.

---

## Setup & Running Instructions (Windows)

### Prerequisites
- Python 3.11+
- Node.js (v18+) and npm

### 1. Backend Setup & Startup
Open PowerShell or Command Prompt:

```powershell
cd "C:\Users\omkar\Desktop\SIT\SEM V\Hack-O-Week\Week_17_18\backend"

# Install dependencies (FastAPI, PyTorch, Torchvision, Transformers)
pip install -r requirements.txt

# Run the FastAPI server (Listens on http://127.0.0.1:8000)
python run.py
```

### 2. Frontend Setup & Startup
In a second terminal:

```powershell
cd "C:\Users\omkar\Desktop\SIT\SEM V\Hack-O-Week\Week_17_18\frontend"

# Install npm dependencies
npm install

# Start Vite development server
npm run dev
```

Open your browser and navigate to: **`http://127.0.0.1:5173/`**

---

## Progress Preservation
Progress (completed modules, quiz scores, and experiment runs) is automatically stored in the browser's `localStorage` and persists across page reloads. A **Reset** button in the header allows clearing progress at any time.

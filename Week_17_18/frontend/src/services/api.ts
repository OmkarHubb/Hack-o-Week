// API Client for Deep Learning Lab Backend
const API_BASE_URL = 'http://127.0.0.1:8000';

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errorDetail = res.statusText;
    try {
      const errJson = await res.json();
      errorDetail = errJson.detail || errJson.error || JSON.stringify(errJson);
    } catch {
      // ignore
    }
    throw new Error(`API Error (${res.status}): ${errorDetail}`);
  }
  return res.json();
}

export const api = {
  // Health
  getHealth: async () => {
    const res = await fetch(`${API_BASE_URL}/api/health`);
    return handleResponse<{
      status: string;
      pytorch_version: string;
      cuda_available: boolean;
      device: string;
      lab_modules_count: number;
    }>(res);
  },

  // Module 1: PyTorch Basics
  runTensorOp: async (params: { op_type: string; shape_a: number[]; shape_b: number[]; activation?: string }) => {
    const res = await fetch(`${API_BASE_URL}/api/m1/tensor-op`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return handleResponse<any>(res);
  },

  runAutograd: async (params: { formula: string; x_val: number }) => {
    const res = await fetch(`${API_BASE_URL}/api/m1/autograd`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return handleResponse<any>(res);
  },

  trainMlp: async (params: { learning_rate: number; optimizer_name: string; epochs: number; hidden_dim: number }) => {
    const res = await fetch(`${API_BASE_URL}/api/m1/train-mlp`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return handleResponse<any>(res);
  },

  // Module 2: CNNs
  runConvFilter: async (params: { filter_type: string; custom_kernel?: number[][]; input_pattern: string }) => {
    const res = await fetch(`${API_BASE_URL}/api/m2/conv-filter`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return handleResponse<any>(res);
  },

  trainCnn: async (params: { dataset_name: string; epochs: number; learning_rate: number; batch_size: number }) => {
    const res = await fetch(`${API_BASE_URL}/api/m2/train-cnn`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return handleResponse<any>(res);
  },

  predictCnnSample: async (sampleIndex: number) => {
    const res = await fetch(`${API_BASE_URL}/api/m2/predict-sample?sample_index=${sampleIndex}`);
    return handleResponse<any>(res);
  },

  // Module 3: RNNs
  compareRecurrent: async (params: { task_type?: string; epochs: number; hidden_size: number; sequence_length: number }) => {
    const res = await fetch(`${API_BASE_URL}/api/m3/compare-recurrent`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return handleResponse<any>(res);
  },

  // Module 4: Regularization
  runRegularization: async (params: { dropout_rate: number; epochs: number; learning_rate: number; weight_decay: number }) => {
    const res = await fetch(`${API_BASE_URL}/api/m4/regularization-experiment`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return handleResponse<any>(res);
  },

  // Module 5: Transfer Learning
  runTransferLearning: async (params: { mode: string; epochs: number; learning_rate: number; target_classes: number }) => {
    const res = await fetch(`${API_BASE_URL}/api/m5/transfer-learning`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return handleResponse<any>(res);
  },

  // Module 6: Transformers
  tokenizeText: async (params: { text: string }) => {
    const res = await fetch(`${API_BASE_URL}/api/m6/tokenize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return handleResponse<any>(res);
  },

  runTransformerTask: async (params: { task: string; text: string; context?: string; question?: string }) => {
    const res = await fetch(`${API_BASE_URL}/api/m6/transformer-task`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return handleResponse<any>(res);
  },

  // Module 7: LLM APIs
  generateLlm: async (params: {
    provider: string;
    prompt: string;
    api_key?: string;
    system_prompt?: string;
    temperature?: number;
    max_tokens?: number;
    ollama_endpoint?: string;
    model_name?: string;
  }) => {
    const res = await fetch(`${API_BASE_URL}/api/m7/generate-llm`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return handleResponse<any>(res);
  },
};

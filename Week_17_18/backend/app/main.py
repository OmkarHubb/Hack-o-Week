from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

from app.schemas import (
    TensorOpRequest, AutogradRequest, BasicTrainLoopRequest,
    ConvFilterRequest, CnnTrainRequest, CnnPredictRequest,
    SequenceCompareRequest, RegularizationExperimentRequest,
    TransferLearningRequest, TokenizeRequest, TransformerTaskRequest,
    LLMRequest
)
from app.modules import (
    m1_pytorch_basics,
    m2_cnn,
    m3_rnn_lstm_gru,
    m4_regularization,
    m5_transfer_learning,
    m6_transformers,
    m7_llm_apis
)

app = FastAPI(
    title="Deep Learning Learning Lab API",
    description="Interactive backend covering PyTorch, CNNs, RNNs/LSTMs/GRUs, Regularization, Transfer Learning, Transformers, and LLMs.",
    version="1.0.0"
)

# Enable CORS for React frontend (Vite dev server default: port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health_check():
    import torch
    return {
        "status": "healthy",
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lab_modules_count": 7
    }

# ----------------- MODULE 1: PyTorch Fundamentals -----------------
@app.post("/api/m1/tensor-op")
def tensor_operations(req: TensorOpRequest):
    try:
        return m1_pytorch_basics.run_tensor_operations(
            op_type=req.op_type,
            shape_a=req.shape_a,
            shape_b=req.shape_b,
            activation=req.activation or "relu"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/m1/autograd")
def autograd_demo(req: AutogradRequest):
    try:
        return m1_pytorch_basics.run_autograd_demo(
            formula=req.formula,
            x_val=req.x_val
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/m1/train-mlp")
def train_mlp(req: BasicTrainLoopRequest):
    try:
        return m1_pytorch_basics.train_basic_mlp(
            learning_rate=req.learning_rate,
            optimizer_name=req.optimizer_name,
            epochs=req.epochs,
            hidden_dim=req.hidden_dim
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------- MODULE 2: CNNs -----------------
@app.post("/api/m2/conv-filter")
def conv_filter(req: ConvFilterRequest):
    try:
        return m2_cnn.apply_conv_filter_demo(
            filter_type=req.filter_type,
            custom_kernel=req.custom_kernel,
            input_pattern=req.input_pattern
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/m2/train-cnn")
def train_cnn(req: CnnTrainRequest):
    try:
        return m2_cnn.train_cnn_classifier(
            dataset_name=req.dataset_name,
            epochs=req.epochs,
            learning_rate=req.learning_rate,
            batch_size=req.batch_size
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/m2/predict-sample")
def predict_cnn_sample(sample_index: int = 0):
    try:
        return m2_cnn.predict_sample_inspection(sample_index=sample_index)
    except Exception as e:
        # Fallback to function name in m2_cnn
        return m2_cnn.predict_cnn_sample(sample_index=sample_index)

# ----------------- MODULE 3: RNNs, LSTMs, GRUs -----------------
@app.post("/api/m3/compare-recurrent")
def compare_recurrent(req: SequenceCompareRequest):
    try:
        return m3_rnn_lstm_gru.compare_recurrent_architectures(
            epochs=req.epochs,
            hidden_size=req.hidden_size,
            sequence_length=req.sequence_length
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------- MODULE 4: Regularization -----------------
@app.post("/api/m4/regularization-experiment")
def regularization_experiment(req: RegularizationExperimentRequest):
    try:
        return m4_regularization.run_regularization_experiment(
            dropout_rate=req.dropout_rate,
            epochs=req.epochs,
            learning_rate=req.learning_rate,
            weight_decay=req.weight_decay
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------- MODULE 5: Transfer Learning -----------------
@app.post("/api/m5/transfer-learning")
def transfer_learning(req: TransferLearningRequest):
    try:
        return m5_transfer_learning.run_transfer_learning_experiment(
            mode=req.mode,
            epochs=req.epochs,
            learning_rate=req.learning_rate,
            target_classes=req.target_classes
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------- MODULE 6: Transformers -----------------
@app.post("/api/m6/tokenize")
def tokenize_text(req: TokenizeRequest):
    try:
        return m6_transformers.explore_tokenization(text=req.text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/m6/transformer-task")
def transformer_task(req: TransformerTaskRequest):
    try:
        return m6_transformers.run_transformer_task(
            task=req.task,
            text=req.text,
            context=req.context,
            question=req.question
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------- MODULE 7: LLM APIs -----------------
@app.post("/api/m7/generate-llm")
async def generate_llm(req: LLMRequest):
    try:
        return await m7_llm_apis.execute_llm_generation(
            provider=req.provider,
            prompt=req.prompt,
            api_key=req.api_key,
            system_prompt=req.system_prompt,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            ollama_endpoint=req.ollama_endpoint or "http://localhost:11434",
            model_name=req.model_name or "gpt-4o-mini"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

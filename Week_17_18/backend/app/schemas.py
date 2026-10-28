from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Module 1: PyTorch Basics
class TensorOpRequest(BaseModel):
    op_type: str = Field(default="matmul", description="matmul, add, reshape, transpose, activation")
    shape_a: List[int] = Field(default=[2, 3])
    shape_b: List[int] = Field(default=[3, 2])
    activation: Optional[str] = "relu"

class AutogradRequest(BaseModel):
    formula: str = Field(default="3*x**2 + 2*x + 1")
    x_val: float = Field(default=2.0)

class BasicTrainLoopRequest(BaseModel):
    learning_rate: float = Field(default=0.05, ge=0.0001, le=1.0)
    optimizer_name: str = Field(default="Adam", description="SGD or Adam")
    epochs: int = Field(default=30, ge=5, le=100)
    hidden_dim: int = Field(default=16, ge=4, le=64)

# Module 2: CNNs
class ConvFilterRequest(BaseModel):
    filter_type: str = Field(default="sobel_x", description="sobel_x, sobel_y, sharpen, gaussian_blur, ridge, custom")
    custom_kernel: Optional[List[List[float]]] = None
    input_pattern: str = Field(default="digit_seven", description="digit_seven, cross, checkerboard, diagonal")

class CnnTrainRequest(BaseModel):
    dataset_name: str = Field(default="mnist", description="mnist or fashion_mnist")
    epochs: int = Field(default=5, ge=1, le=15)
    learning_rate: float = Field(default=0.002, ge=0.0001, le=0.05)
    batch_size: int = Field(default=32, ge=8, le=128)

class CnnPredictRequest(BaseModel):
    dataset_name: str = Field(default="mnist")
    sample_index: int = Field(default=0, ge=0, le=49)

# Module 3: RNNs, LSTMs, GRUs
class SequenceCompareRequest(BaseModel):
    task_type: str = Field(default="sine_forecasting", description="sine_forecasting or char_prediction")
    epochs: int = Field(default=20, ge=5, le=60)
    hidden_size: int = Field(default=32, ge=8, le=64)
    sequence_length: int = Field(default=20, ge=5, le=50)

# Module 4: Regularization
class RegularizationExperimentRequest(BaseModel):
    mode: str = Field(default="all", description="baseline, dropout, batchnorm, early_stopping, or all")
    dropout_rate: float = Field(default=0.4, ge=0.0, le=0.9)
    epochs: int = Field(default=40, ge=10, le=100)
    learning_rate: float = Field(default=0.01, ge=0.0001, le=0.1)
    weight_decay: float = Field(default=0.0001, ge=0.0, le=0.05)

# Module 5: Transfer Learning
class TransferLearningRequest(BaseModel):
    mode: str = Field(default="feature_extraction", description="feature_extraction or fine_tuning")
    epochs: int = Field(default=5, ge=1, le=15)
    learning_rate: float = Field(default=0.001, ge=0.00001, le=0.01)
    target_classes: int = Field(default=5, ge=2, le=10)

# Module 6: Transformers
class TokenizeRequest(BaseModel):
    text: str = Field(default="Hugging Face Transformers make deep learning accessible and fun!")

class TransformerTaskRequest(BaseModel):
    task: str = Field(default="sentiment", description="sentiment, summarization, qa")
    text: str = Field(default="I absolutely love learning deep learning with PyTorch, it is fantastic!")
    context: Optional[str] = None
    question: Optional[str] = None

# Module 7: LLM APIs
class LLMRequest(BaseModel):
    provider: str = Field(default="open_model", description="open_model, openai, anthropic, or ollama")
    api_key: Optional[str] = None
    prompt: str = Field(default="Explain how backpropagation works with the chain rule in two concise paragraphs.")
    system_prompt: Optional[str] = "You are an expert deep learning instructor. Explain concepts clearly and concisely."
    temperature: float = Field(default=0.7, ge=0.0, le=1.5)
    max_tokens: int = Field(default=256, ge=16, le=1024)
    ollama_endpoint: Optional[str] = "http://localhost:11434"
    model_name: Optional[str] = "gpt-4o-mini"

# Architecture

## System Overview

This project fine-tunes **Qwen2.5-1.5B-Instruct** for Financial Q&A using **QLoRA** (Quantized Low-Rank Adaptation). The pipeline consists of three sequential stages: data preparation, training, and evaluation.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Preparation│ ──► │   QLoRA Training │ ──► │   Evaluation    │
│ (prepare_data.py)│     │   (train.py)     │     │  (evaluate.py)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
   data/train.jsonl       output/final_adapter    evaluation/
   data/eval.jsonl        (LoRA weights)          results + samples
```

## Component Details

### 1. Data Preparation (`src/prepare_data.py`)

- **Input**: HuggingFace dataset `gbharti/finance-alpaca`
- **Processing**:
  - Filters out samples with short questions/answers (<10 chars) or overly long outputs (>2048 chars)
  - Normalizes whitespace
  - Randomly samples 1000 entries (configurable)
  - Formats into chat template: `system → user → assistant`
  - Splits 90/10 into train/eval
- **Output**: `data/train.jsonl` (900 samples), `data/eval.jsonl` (100 samples)

### 2. QLoRA Training (`src/train.py`)

- **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct` loaded in 4-bit (NF4 quantization via bitsandbytes)
- **LoRA Configuration**:
  - Rank: 16, Alpha: 32, Dropout: 0.05
  - Target modules: all attention projections (`q/k/v/o_proj`) + MLP layers (`gate/up/down_proj`)
  - ~0.5% of parameters trainable
- **Training**: HuggingFace `SFTTrainer` with cosine LR schedule, gradient checkpointing, 3 epochs
- **Output**: LoRA adapter weights saved to `output/financial-qa-qlora/final_adapter/`

### 3. Evaluation (`src/evaluate.py`)

- Loads both the base model and the fine-tuned model (base + LoRA adapter)
- Runs greedy inference on eval samples
- Computes **ROUGE-1, ROUGE-2, ROUGE-L** scores for both models
- Saves quantitative results and qualitative side-by-side comparisons

## Data Flow

```
HuggingFace Dataset
       │
       ▼
┌──────────────┐
│  Clean &     │   Remove short/long samples
│  Filter      │   Normalize whitespace
└──────┬───────┘
       ▼
┌──────────────┐
│  Format to   │   system + user + assistant
│  Chat Template│   messages structure
└──────┬───────┘
       ▼
┌──────────────┐
│  Train/Eval  │   90/10 split
│  Split       │   Saved as JSONL
└──────┬───────┘
       ▼
┌──────────────┐
│  SFTTrainer  │   QLoRA fine-tuning
│  Training    │   on chat-formatted data
└──────┬───────┘
       ▼
┌──────────────┐
│  Evaluation  │   ROUGE metrics +
│  & Compare   │   qualitative samples
└──────────────┘
```

## Tech Stack

| Component       | Technology                          |
|-----------------|-------------------------------------|
| Language        | Python 3.10+                        |
| Base Model      | Qwen/Qwen2.5-1.5B-Instruct         |
| Fine-Tuning     | PEFT (LoRA), TRL (SFTTrainer)       |
| Quantization    | bitsandbytes (4-bit NF4)            |
| Framework       | HuggingFace Transformers            |
| Evaluation      | rouge-score                         |
| Compute         | Google Colab T4 GPU (free tier)     |

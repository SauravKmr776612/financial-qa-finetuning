# Financial Q&A - QLoRA Fine-Tuning on Qwen2.5-1.5B-Instruct

Fine-tune a small open-source language model (Qwen2.5-1.5B-Instruct) for Financial Question Answering using QLoRA, and demonstrate measurable improvement over the base model.

## Model Selection

**Model**: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

**Why this model?**
- 1.5B parameters — small enough to fine-tune on a free Google Colab T4 GPU with QLoRA
- Instruction-tuned variant provides a strong baseline for Q&A tasks
- Qwen2.5 architecture has strong multilingual and reasoning capabilities for its size
- Active community support and well-documented tokenizer/chat template

**Trade-offs (Size vs Performance)**:
- Smaller than Mistral-7B, so faster training and lower memory usage (~6GB VRAM with 4-bit quantization)
- Slightly lower base performance on complex financial reasoning vs 7B models, but the gap narrows significantly after domain-specific fine-tuning
- Ideal for demonstrating measurable improvement with limited compute

## Fine-Tuning Approach

**QLoRA** (Quantized Low-Rank Adaptation):
- Base model loaded in 4-bit (NF4 quantization) to reduce memory
- LoRA adapters (rank=16, alpha=32) applied to all attention + MLP projections
- Only ~0.5% of parameters are trainable, making training fast and efficient

## Dataset

**Source**: [gbharti/finance-alpaca](https://huggingface.co/datasets/gbharti/finance-alpaca) (HuggingFace)
- ~1000 samples selected (900 train / 100 eval)
- Financial Q&A pairs covering investing, banking, taxation, accounting, and personal finance

**Data Cleaning**:
- Removed samples with very short questions/answers (<10 chars)
- Removed overly long outputs (>2048 chars)
- Normalized whitespace

**Formatting**: Instruction-tuning chat format with system prompt, user question, and assistant response.

## Directory Structure

```
repo/
├── src/
│   ├── prepare_data.py    # Step 1: Data loading, cleaning, formatting
│   ├── train.py           # Step 2: QLoRA fine-tuning with SFTTrainer
│   └── evaluate.py        # Step 3: Base vs Fine-tuned comparison
├── data/                  # Generated train.jsonl and eval.jsonl
├── evaluation/            # Evaluation results and qualitative samples
├── output/                # Model checkpoints and LoRA adapters
├── README.md
└── requirements.txt
```

## Setup & Workflow

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
python src/prepare_data.py --sample_size 1000
```

This downloads the finance-alpaca dataset, cleans it, formats it into chat template, and saves train/eval splits to `data/`.

### 3. Fine-Tune with QLoRA

```bash
python src/train.py
```

**Hyperparameters**:
| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch Size | 4 |
| Gradient Accumulation | 4 (effective batch = 16) |
| Learning Rate | 2e-4 |
| LR Scheduler | Cosine |
| Warmup Ratio | 0.1 |
| Max Seq Length | 512 |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Quantization | 4-bit NF4 |

The trained LoRA adapter is saved to `output/financial-qa-qlora/final_adapter/`.

### 4. Evaluate

```bash
python src/evaluate.py --num_samples 50
```

Compares base model vs fine-tuned model on the eval set using:
- **ROUGE-1, ROUGE-2, ROUGE-L** scores
- **Qualitative comparison** of sample outputs (saved to `evaluation/qualitative_samples.json`)

## Evaluation Metrics

| Metric | Base Model | Fine-Tuned | Improvement |
|--------|-----------|------------|-------------|
| ROUGE-1 | 0.3433 | 0.3787 | +0.0354 |
| ROUGE-2 | 0.1350 | 0.1685 | +0.0335 |
| ROUGE-L | 0.2348 | 0.2813 | +0.0465 |

## Tech Stack

- Python 3.10+
- PyTorch
- HuggingFace Transformers, PEFT, TRL
- bitsandbytes (4-bit quantization)
- rouge-score (evaluation)

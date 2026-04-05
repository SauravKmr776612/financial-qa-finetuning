"""
Evaluation Script - Base Model vs Fine-Tuned Model
- Loads the base model and the fine-tuned (LoRA) model
- Runs inference on the eval set
- Computes ROUGE scores and qualitative comparisons
- Saves results to evaluation/
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def generate_response(model, tokenizer, messages: list[dict], max_new_tokens=256) -> str:
    """Generate a response given chat messages."""
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(result[key].fmeasure)

    return {key: sum(vals) / len(vals) for key, vals in scores.items()}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Base vs Fine-Tuned Model")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="output/financial-qa-qlora/final_adapter")
    parser.add_argument("--eval_file", type=str, default="data/eval.jsonl")
    parser.add_argument("--output_dir", type=str, default="evaluation")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of eval samples to test")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load eval data ---
    eval_data = load_jsonl(args.eval_file)[:args.num_samples]
    print(f"Evaluating on {len(eval_data)} samples")

    # --- Quantization config ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load BASE model ---
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # --- Generate base model responses ---
    print("Generating base model responses...")
    base_predictions = []
    references = []
    input_prompts = []

    for sample in tqdm(eval_data, desc="Base model"):
        messages = sample["messages"]
        input_msgs = [m for m in messages if m["role"] != "assistant"]
        reference = [m for m in messages if m["role"] == "assistant"][0]["content"]

        pred = generate_response(base_model, tokenizer, input_msgs)
        base_predictions.append(pred)
        references.append(reference)
        input_prompts.append(input_msgs[-1]["content"])

    # --- Compute base model ROUGE ---
    base_rouge = compute_rouge(base_predictions, references)
    print(f"\nBase Model ROUGE: {base_rouge}")

    # --- Load fine-tuned model (base + LoRA adapter) ---
    print("\nLoading fine-tuned model (LoRA adapter)...")
    ft_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    ft_model.eval()

    # --- Generate fine-tuned model responses ---
    print("Generating fine-tuned model responses...")
    ft_predictions = []

    for sample in tqdm(eval_data, desc="Fine-tuned model"):
        messages = sample["messages"]
        input_msgs = [m for m in messages if m["role"] != "assistant"]
        pred = generate_response(ft_model, tokenizer, input_msgs)
        ft_predictions.append(pred)

    # --- Compute fine-tuned model ROUGE ---
    ft_rouge = compute_rouge(ft_predictions, references)
    print(f"Fine-Tuned Model ROUGE: {ft_rouge}")

    # --- Comparison Summary ---
    summary = {
        "num_samples": len(eval_data),
        "base_model": args.model_name,
        "adapter_path": args.adapter_path,
        "base_rouge": base_rouge,
        "finetuned_rouge": ft_rouge,
        "improvement": {
            key: ft_rouge[key] - base_rouge[key] for key in base_rouge
        },
    }

    summary_path = output_dir / "evaluation_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # --- Qualitative Samples ---
    qualitative = []
    for i in range(min(10, len(eval_data))):
        qualitative.append({
            "question": input_prompts[i],
            "reference": references[i],
            "base_response": base_predictions[i],
            "finetuned_response": ft_predictions[i],
        })

    qual_path = output_dir / "qualitative_samples.json"
    with open(qual_path, "w") as f:
        json.dump(qualitative, f, indent=2, ensure_ascii=False)

    # --- Print Results ---
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'Metric':<15} {'Base':>10} {'Fine-Tuned':>12} {'Improvement':>13}")
    print("-" * 50)
    for key in ["rouge1", "rouge2", "rougeL"]:
        print(f"{key:<15} {base_rouge[key]:>10.4f} {ft_rouge[key]:>12.4f} {summary['improvement'][key]:>+13.4f}")
    print("=" * 60)
    print(f"\nResults saved to: {summary_path}")
    print(f"Qualitative samples saved to: {qual_path}")


if __name__ == "__main__":
    main()

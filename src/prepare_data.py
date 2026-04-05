"""
Data Preparation Script for Financial Q&A Fine-Tuning
- Loads the gbharti/finance-alpaca dataset from HuggingFace
- Cleans and filters the data
- Formats into instruction-tuning chat template
- Splits into train/eval sets
- Saves as JSONL files
"""

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """Remove extra whitespace and normalize text."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_sample(sample: dict) -> bool:
    """Filter out low-quality samples."""
    instruction = clean_text(sample.get("instruction", ""))
    output = clean_text(sample.get("output", ""))

    if len(instruction) < 10 or len(output) < 10:
        return False
    if len(output) > 2048:
        return False
    return True


def format_to_chat(sample: dict) -> dict:
    """Convert a sample into chat-style messages for SFTTrainer."""
    instruction = clean_text(sample["instruction"])
    context = clean_text(sample.get("input", ""))
    output = clean_text(sample["output"])

    user_message = instruction
    if context:
        user_message = f"{instruction}\n\nContext: {context}"

    return {
        "messages": [
            {"role": "system", "content": "You are a helpful financial assistant. Answer financial questions accurately and concisely."},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": output},
        ]
    }


def save_jsonl(data: list[dict], path: Path):
    """Save list of dicts as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare Financial Q&A dataset")
    parser.add_argument("--sample_size", type=int, default=1000, help="Total samples to use")
    parser.add_argument("--eval_ratio", type=float, default=0.1, help="Fraction for evaluation")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading finance-alpaca dataset from HuggingFace...")
    dataset = load_dataset("gbharti/finance-alpaca", split="train")
    print(f"  Raw dataset size: {len(dataset)}")

    print("Filtering and cleaning...")
    valid_samples = [s for s in dataset if is_valid_sample(s)]
    print(f"  Valid samples: {len(valid_samples)}")

    if len(valid_samples) > args.sample_size:
        import random
        random.seed(args.seed)
        valid_samples = random.sample(valid_samples, args.sample_size)
    print(f"  Selected samples: {len(valid_samples)}")

    print("Formatting into chat template...")
    formatted = [format_to_chat(s) for s in valid_samples]

    train_data, eval_data = train_test_split(
        formatted, test_size=args.eval_ratio, random_state=args.seed
    )

    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    save_jsonl(train_data, train_path)
    save_jsonl(eval_data, eval_path)

    print(f"\nDone!")
    print(f"  Train: {len(train_data)} samples -> {train_path}")
    print(f"  Eval:  {len(eval_data)} samples -> {eval_path}")


if __name__ == "__main__":
    main()

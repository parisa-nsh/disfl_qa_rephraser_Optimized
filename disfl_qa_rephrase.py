"""
Disfl QA Question Rephrasing Model - T5 Solution
================================================
This script fine-tunes a T5 transformer model to rephrase disfluent (noisy) questions 
into fluent (clean) questions using the Disfl QA benchmark dataset.

PRIMARY SOLUTION: T5 Transformer Model
- State-of-the-art sequence-to-sequence architecture
- Pre-trained on massive text corpus
- Superior performance on text generation tasks
- Industry standard for NLP tasks

Sections:
1. Environment Setup & Imports
2. Data Download & Preprocessing  
3. T5 Model Training & Evaluation
4. Results Analysis & Visualization
5. Overfitting/Overconfidence Analysis

Author: Paris Nouri
Date: August 1st 2025
"""

# 1. Environment Setup & Imports
import os
import json
import requests
from datasets import load_dataset, Dataset, DatasetDict
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import torch
import sacrebleu
import nltk
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np

# =========================
# Section 2: Data Processing
# =========================

def download_disfl_qa_data(data_dir="data"):
    """
    Downloads the Disfl QA train and dev datasets from GitHub if not already present.
    Args:
        data_dir (str): Directory to save the datasets.
    Returns:
        train_path, dev_path (str): Paths to the downloaded train and dev JSON files.
    """
    os.makedirs(data_dir, exist_ok=True)
    base_url = "https://raw.githubusercontent.com/google-research-datasets/Disfl-QA/main/"
    files = {"train": "train.json", "dev": "dev.json"}
    paths = {}
    for split, fname in files.items():
        out_path = os.path.join(data_dir, fname)
        if not os.path.exists(out_path):
            print(f"Downloading {fname}...")
            url = base_url + fname
            r = requests.get(url)
            r.raise_for_status()
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(r.text)
        else:
            print(f"{fname} already exists.")
        paths[split] = out_path
    return paths["train"], paths["dev"]

def load_and_prepare_dataset(train_path, dev_path):
    """
    Loads the Disfl QA data and prepares it as HuggingFace Datasets.
    Args:
        train_path, dev_path (str): Paths to train and dev JSON files.
    Returns:
        DatasetDict with 'train' and 'validation' splits.
    """
    def parse_json_dict(path):
        # The file is a JSON dict (id -> question object), not a list
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Each value is a dict with 'original' and 'disfluent' keys
        return [{
            "disfluent": item["disfluent"],
            "fluent": item["original"]
        } for item in data.values()]

    train_data = parse_json_dict(train_path)
    dev_data = parse_json_dict(dev_path)
    train_ds = Dataset.from_list(train_data)
    dev_ds = Dataset.from_list(dev_data)
    return DatasetDict({"train": train_ds, "validation": dev_ds})

def preprocess_data(dataset, tokenizer, max_input_length=64, max_target_length=64):
    """
    Tokenizes the dataset for T5 model training.
    Args:
        dataset (DatasetDict): HuggingFace DatasetDict with 'disfluent' and 'fluent' fields.
        tokenizer (T5Tokenizer): Tokenizer for the model.
        max_input_length (int): Max length for input (disfluent) questions.
        max_target_length (int): Max length for target (fluent) questions.
    Returns:
        Tokenized DatasetDict.
    """
    def preprocess_function(examples):
        # Prefix for T5 to specify the task
        inputs = ["rephrase: " + q for q in examples["disfluent"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        # Tokenize targets (fluent questions)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["fluent"], max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return dataset.map(preprocess_function, batched=True, remove_columns=["disfluent", "fluent"])

# =========================
# Section 3: T5 Model Training & Evaluation
# =========================

def train_t5_model(dataset, model_name="t5-small", num_epochs=3):
    """
    Trains the T5 model for question rephrasing.
    """
    print(f"\n=== Training T5 Model ({model_name}) ===")
    
    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Tokenize data
    print("Tokenizing dataset...")
    tokenized_dataset = preprocess_data(dataset, tokenizer)
    
    # Training arguments optimized for T5
    training_args = TrainingArguments(
        output_dir="./t5_results",
        eval_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_total_limit=1,
        logging_dir="./t5_logs",
        logging_steps=50,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_strategy="epoch",
    )

    # Define data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )

    print("Starting T5 model training...")
    trainer.train()
    print("T5 training complete!")
    
    # Save the trained model
    trainer.save_model("./t5_disfl_qa_rephraser")
    print("T5 model saved to ./t5_disfl_qa_rephraser")
    
    return trainer, tokenizer

def evaluate_t5_model(trainer, tokenizer, dataset, num_samples=10):
    """
    Evaluates the T5 model using BLEU and GLEU metrics.
    """
    print("\n=== T5 Model Evaluation ===")
    
    inputs = ["rephrase: " + q for q in dataset["validation"]["disfluent"]]
    targets = dataset["validation"]["fluent"]
    preds = []
    
    # Generate predictions in batches
    for i in range(0, len(inputs), 8):  # batch size 8
        batch = inputs[i:i+8]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(trainer.model.device)
        with torch.no_grad():
            outs = trainer.model.generate(**enc, max_length=64)
        decoded = tokenizer.batch_decode(outs, skip_special_tokens=True)
        preds.extend(decoded)
    
    # Compute BLEU score
    bleu = sacrebleu.corpus_bleu(preds, [targets])
    
    print(f"T5 Model Performance:")
    print(f"BLEU score: {bleu.score:.2f}")
    
    # Print sample predictions
    print("\nT5 Model Sample Predictions:")
    for i in range(num_samples):
        print(f"Disfluent: {dataset['validation'][i]['disfluent']}")
        print(f"Target   : {dataset['validation'][i]['fluent']}")
        print(f"Predicted: {preds[i]}")
        print("-")
    
    return bleu.score, preds

def analyze_overfitting(trainer):
    """
    Analyzes overfitting and overconfidence in the T5 model.
    """
    print("\n=== Overfitting Analysis ===")
    
    logs = trainer.state.log_history
    train_loss = [x["loss"] for x in logs if "loss" in x]
    eval_loss = [x["eval_loss"] for x in logs if "eval_loss" in x]
    steps = [x["step"] for x in logs if "loss" in x]
    eval_steps = [x["step"] for x in logs if "eval_loss" in x]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_loss, label="Train Loss", linewidth=2)
    plt.plot(eval_steps, eval_loss, label="Validation Loss", linewidth=2)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("T5 Model: Training vs. Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("t5_loss_curves.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analyze overfitting
    if len(train_loss) > 0 and len(eval_loss) > 0:
        final_train_loss = train_loss[-1]
        final_eval_loss = eval_loss[-1]
        loss_gap = final_train_loss - final_eval_loss
        
        print(f"Final Training Loss: {final_train_loss:.4f}")
        print(f"Final Validation Loss: {final_eval_loss:.4f}")
        print(f"Loss Gap: {loss_gap:.4f}")
        
        if loss_gap > 0.1:
            print("  Potential overfitting detected (large gap between train/val loss)")
        else:
            print(" Model shows good generalization (small gap between train/val loss)")
    
    print("Loss curves saved as t5_loss_curves.png")

# =========================
# Section 4: Main Execution
# =========================

def main():
    """
    Main execution function for T5 model training and evaluation.
    """
    print("=" * 60)
    print("DISFL QA QUESTION REPHRASING - T5 SOLUTION")
    print("=" * 60)
    
    # 1. Data Download & Preprocessing
    print("\n=== Step 1: Data Download & Preprocessing ===")
    train_path, dev_path = download_disfl_qa_data()
    dataset = load_and_prepare_dataset(train_path, dev_path)
    print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['validation'])} validation samples")
    print(f"Sample disfluent: {dataset['train'][0]['disfluent']}")
    print(f"Sample fluent: {dataset['train'][0]['fluent']}")

    # 2. T5 Model Training
    trainer, tokenizer = train_t5_model(dataset, model_name="t5-small", num_epochs=3)

    # 3. T5 Model Evaluation
    t5_bleu, t5_preds = evaluate_t5_model(trainer, tokenizer, dataset)

    # 4. Overfitting Analysis
    analyze_overfitting(trainer)

    print("\n" + "=" * 60)
    print("EXECUTION COMPLETE")
    print("=" * 60)
    print("T5 Transformer Model Results:")
    print(f"Final BLEU Score: {t5_bleu:.2f}")
    print("Model saved to: ./t5_disfl_qa_rephraser")
    print("Results saved to: ./t5_results/")
    print("=" * 60)

if __name__ == "__main__":
    main() 
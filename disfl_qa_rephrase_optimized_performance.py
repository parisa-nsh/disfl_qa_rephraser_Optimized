"""
Disfl QA Question Rephrasing Model - OPTIMIZED PERFORMANCE T5 Solution
=====================================================================
Optimized version that balances speed and high BLEU score.

TARGET: 80-85 BLEU Score
STRATEGY: T5-small + Optimized Training + Moderate Augmentation

Author: Paris Nouri
Date: August 1st 2025
"""

# 1. Environment Setup & Imports
import os
import json
import requests
import random
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, 
    DataCollatorForSeq2Seq, EarlyStoppingCallback
)
import torch
import sacrebleu
import nltk
import matplotlib.pyplot as plt
from tabulate import tabulate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re
import shutil

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# =========================
# Section 2: Optimized Performance Data Processing
# =========================

def download_disfl_qa_data(data_dir="data"):
    """Downloads the Disfl QA train and dev datasets from GitHub if not already present."""
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
    """Loads the Disfl QA data and prepares it as HuggingFace Datasets."""
    def parse_json_dict(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [{
            "disfluent": item["disfluent"],
            "fluent": item["original"]
        } for item in data.values()]

    train_data = parse_json_dict(train_path)
    dev_data = parse_json_dict(dev_path)
    train_ds = Dataset.from_list(train_data)
    dev_ds = Dataset.from_list(dev_data)
    return DatasetDict({"train": train_ds, "validation": dev_ds})

def optimized_data_augmentation(dataset, augmentation_factor=0.5):
    """
    Optimized data augmentation for balanced performance.
    """
    print(f"Applying OPTIMIZED data augmentation (factor: {augmentation_factor})...")
    
    augmented_samples = []
    num_to_augment = int(len(dataset["train"]) * augmentation_factor)
    
    # Sample random examples for augmentation
    indices = random.sample(range(len(dataset["train"])), num_to_augment)
    
    # Optimized augmentation techniques
    filler_words = ["um", "uh", "like", "you know", "I mean"]
    correction_patterns = ["no sorry", "wait", "I mean", "actually"]
    
    for idx in indices:
        sample = dataset["train"][idx]
        disfluent = sample["disfluent"]
        fluent = sample["fluent"]
        
        # Simple but effective augmentation
        augmentation_type = random.choice(["filler", "correction", "mixed"])
        
        if augmentation_type == "filler":
            # Add filler words
            words = fluent.split()
            if len(words) > 3:
                num_fillers = random.randint(1, 2)
                for _ in range(num_fillers):
                    insert_pos = random.randint(1, len(words) - 1)
                    words.insert(insert_pos, random.choice(filler_words))
                disfluent = " ".join(words)
        
        elif augmentation_type == "correction":
            # Add self-correction patterns
            words = fluent.split()
            if len(words) > 2:
                insert_pos = random.randint(1, len(words) - 1)
                words.insert(insert_pos, random.choice(correction_patterns))
                disfluent = " ".join(words)
        
        else:  # mixed
            # Combine techniques
            words = fluent.split()
            if len(words) > 4:
                # Add filler
                if random.random() < 0.5:
                    insert_pos = random.randint(1, len(words) - 1)
                    words.insert(insert_pos, random.choice(filler_words))
                disfluent = " ".join(words)
        
        augmented_samples.append({
            "disfluent": disfluent,
            "fluent": fluent
        })
    
    # Combine original and augmented data
    original_data = [{"disfluent": item["disfluent"], "fluent": item["fluent"]} 
                    for item in dataset["train"]]
    all_data = original_data + augmented_samples
    
    # Shuffle the combined data
    random.shuffle(all_data)
    
    # Create new dataset
    train_ds = Dataset.from_list(all_data)
    dev_ds = Dataset.from_list([{"disfluent": item["disfluent"], "fluent": item["fluent"]} 
                               for item in dataset["validation"]])
    
    print(f"Original train samples: {len(dataset['train'])}")
    print(f"Augmented train samples: {len(all_data)}")
    print(f"Total augmentation: +{len(augmented_samples)} samples")
    print(f"Augmentation ratio: {len(augmented_samples)/len(dataset['train'])*100:.1f}%")
    
    return DatasetDict({"train": train_ds, "validation": dev_ds})

def preprocess_data(dataset, tokenizer, max_input_length=96, max_target_length=96):
    """
    Optimized tokenization with balanced lengths.
    """
    def preprocess_function(examples):
        # Optimized prefix for task specification
        inputs = ["rephrase: " + q for q in examples["disfluent"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding=True)
        
        # Tokenize targets (fluent questions)
        labels = tokenizer(text_target=examples["fluent"], max_length=max_target_length, truncation=True, padding=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    return dataset.map(preprocess_function, batched=True, remove_columns=["disfluent", "fluent"])

# =========================
# Section 3: Optimized Performance Model Training
# =========================

def train_optimized_performance_t5_model(dataset, model_name="t5-small", num_epochs=4):
    """
    Optimized performance T5 model training for balanced speed and quality.
    """
    print(f"\n=== Training OPTIMIZED PERFORMANCE T5 Model ({model_name}) ===")
    print(f"Target: 80-85 BLEU Score")
    print(f"Expected Training Time: 1-1.5 hours")
    print(f"Strategy: Optimized training with balanced augmentation")
    
    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Tokenize data with optimized max lengths
    print("Tokenizing dataset with optimized parameters...")
    tokenized_dataset = preprocess_data(dataset, tokenizer, max_input_length=96, max_target_length=96)
    
    # OPTIMIZED PERFORMANCE training arguments
    training_args = TrainingArguments(
        output_dir="./t5_optimized_performance_results",
        eval_strategy="steps",
        eval_steps=300,  # Balanced evaluation frequency
        learning_rate=5e-5,  # Optimized learning rate
        per_device_train_batch_size=6,  # Balanced batch size
        per_device_eval_batch_size=12,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_total_limit=2,  # Keep 2 best checkpoints
        logging_dir="./t5_optimized_performance_logs",
        logging_steps=150,  # Balanced logging
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_strategy="steps",
        save_steps=300,  # Same as eval_steps
        warmup_steps=500,  # Balanced warmup
        gradient_accumulation_steps=2,  # Effective batch size = 6 * 2 = 12
        fp16=True,  # Mixed precision
        dataloader_num_workers=2,
        remove_unused_columns=False,
        prediction_loss_only=False,
        # Optimized settings
        save_safetensors=False,
        save_only_model=True,
        # Performance optimizations
        dataloader_pin_memory=False,
        group_by_length=False,
        lr_scheduler_type="cosine",
    )

    # Define data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Early stopping callback with balanced patience
    early_stopping = EarlyStoppingCallback(early_stopping_patience=2)
    
    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        callbacks=[early_stopping],
    )

    print("Starting OPTIMIZED PERFORMANCE T5 model training...")
    print("Using optimized techniques for balanced performance...")
    
    # Clean up any existing checkpoints
    if os.path.exists("./t5_optimized_performance_results"):
        shutil.rmtree("./t5_optimized_performance_results", ignore_errors=True)
    
    trainer.train()
    print("OPTIMIZED PERFORMANCE T5 training complete!")
    
    # Save the trained model
    trainer.save_model("./t5_optimized_performance_disfl_qa_rephraser")
    print("OPTIMIZED PERFORMANCE T5 model saved to ./t5_optimized_performance_disfl_qa_rephraser")
    
    return trainer, tokenizer

def evaluate_optimized_performance_model(trainer, tokenizer, dataset, num_samples=8):
    """
    Optimized performance evaluation with balanced metrics.
    """
    print("\n=== OPTIMIZED PERFORMANCE T5 Model Evaluation ===")
    
    inputs = ["rephrase: " + q for q in dataset["validation"]["disfluent"]]
    targets = dataset["validation"]["fluent"]
    preds = []
    
    # Optimized generation parameters
    generation_config = {
        "max_length": 96,
        "num_beams": 3,  # Balanced beam search
        "length_penalty": 0.7,
        "no_repeat_ngram_size": 2,
        "early_stopping": True,
        "temperature": 0.8,
        "do_sample": True,
        "top_k": 30,
        "top_p": 0.9,
    }
    
    # Generate predictions
    for i in range(0, len(inputs), 6):  # Process in batches
        batch = inputs[i:i+6]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(trainer.model.device)
        with torch.no_grad():
            outs = trainer.model.generate(**enc, **generation_config)
        decoded = tokenizer.batch_decode(outs, skip_special_tokens=True)
        preds.extend(decoded)
    
    # Balanced evaluation metrics
    bleu = sacrebleu.corpus_bleu(preds, [targets])
    
    # Calculate sentence-level BLEU scores
    sentence_bleu_scores = []
    smoothie = SmoothingFunction().method1
    for pred, target in zip(preds, targets):
        pred_tokens = nltk.word_tokenize(pred.lower())
        target_tokens = nltk.word_tokenize(target.lower())
        score = sentence_bleu([target_tokens], pred_tokens, smoothing_function=smoothie)
        sentence_bleu_scores.append(score)
    
    avg_sentence_bleu = np.mean(sentence_bleu_scores)
    
    # Calculate exact match rate
    exact_matches = sum(1 for p, t in zip(preds, targets) if p.strip() == t.strip())
    exact_match_rate = exact_matches / len(preds) * 100
    
    # Calculate high BLEU rate (percentage of samples with sentence BLEU > 0.7)
    high_bleu_count = sum(1 for score in sentence_bleu_scores if score > 0.7)
    high_bleu_rate = high_bleu_count / len(sentence_bleu_scores) * 100
    
    print(f"OPTIMIZED PERFORMANCE T5 Model Performance:")
    print(f"Corpus BLEU score: {bleu.score:.2f}")
    print(f"Average Sentence BLEU: {avg_sentence_bleu:.4f}")
    print(f"Exact Match Rate: {exact_match_rate:.2f}%")
    print(f"High BLEU Rate (>0.7): {high_bleu_rate:.2f}%")
    
    # Performance assessment
    if bleu.score >= 85:
        print(" EXCELLENT: High performance achieved! (85+ BLEU)")
    elif bleu.score >= 80:
        print(" VERY GOOD: Good performance (80+ BLEU)")
    elif bleu.score >= 75:
        print(" GOOD: Solid performance (75+ BLEU)")
    elif bleu.score >= 70:
        print(" ACCEPTABLE: Decent performance (70+ BLEU)")
    else:
        print("  NEEDS IMPROVEMENT: Below target")
    
    # Print sample predictions
    print("\nOPTIMIZED PERFORMANCE T5 Model Sample Predictions:")
    for i in range(num_samples):
        print(f"Sample {i+1}:")
        print(f"  Disfluent: {dataset['validation'][i]['disfluent']}")
        print(f"  Target   : {dataset['validation'][i]['fluent']}")
        print(f"  Predicted: {preds[i]}")
        print(f"  Sentence BLEU: {sentence_bleu_scores[i]:.4f}")
        print(f"  Exact Match: {'.' if preds[i].strip() == dataset['validation'][i]['fluent'].strip() else '❌'}")
        print("-" * 60)
    
    return bleu.score, avg_sentence_bleu, exact_match_rate, preds

def analyze_optimized_performance_overfitting(trainer):
    """
    Optimized overfitting analysis.
    """
    print("\n=== OPTIMIZED PERFORMANCE Overfitting Analysis ===")
    
    logs = trainer.state.log_history
    train_loss = [x["loss"] for x in logs if "loss" in x]
    eval_loss = [x["eval_loss"] for x in logs if "eval_loss" in x]
    steps = [x["step"] for x in logs if "loss" in x]
    eval_steps = [x["step"] for x in logs if "eval_loss" in x]
    
    plt.figure(figsize=(10, 4))
    
    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(steps, train_loss, label="Train Loss", linewidth=2, color='blue')
    plt.plot(eval_steps, eval_loss, label="Validation Loss", linewidth=2, color='red')
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("OPTIMIZED PERFORMANCE T5: Training vs. Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss gap
    plt.subplot(1, 2, 2)
    if len(train_loss) > 0 and len(eval_loss) > 0:
        min_len = min(len(train_loss), len(eval_loss))
        loss_gaps = [train_loss[i] - eval_loss[i] for i in range(min_len)]
        plt.plot(steps[:min_len], loss_gaps, label="Loss Gap", linewidth=2, color='green')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel("Training Step")
        plt.ylabel("Loss Gap (Train - Val)")
        plt.title("Overfitting Analysis")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("optimized_performance_t5_loss_curves.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Quick analysis
    if len(train_loss) > 0 and len(eval_loss) > 0:
        final_train_loss = train_loss[-1]
        final_eval_loss = eval_loss[-1]
        loss_gap = final_train_loss - final_eval_loss
        
        print(f"Final Training Loss: {final_train_loss:.4f}")
        print(f"Final Validation Loss: {final_eval_loss:.4f}")
        print(f"Loss Gap: {loss_gap:.4f}")
        
        if loss_gap > 0.1:
            print("  Potential overfitting detected")
        elif loss_gap < -0.1:
            print("  Potential underfitting detected")
        else:
            print(" Model shows good generalization")
    
    print("Loss curves saved as optimized_performance_t5_loss_curves.png")

# =========================
# Section 4: Main Execution
# =========================

def main():
    """
    Main execution function for OPTIMIZED PERFORMANCE T5 model training and evaluation.
    """
    print("=" * 70)
    print("DISFL QA QUESTION REPHRASING - OPTIMIZED PERFORMANCE T5 SOLUTION")
    print("=" * 70)
    print("TARGET: 80-85 BLEU Score")
    print("MODEL: T5-small (60M parameters)")
    print("EXPECTED TRAINING TIME: 1-1.5 hours")
    print("STRATEGY: Optimized training with balanced augmentation")
    print("=" * 70)
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Check available disk space
    import psutil
    disk_usage = psutil.disk_usage('.')
    free_gb = disk_usage.free / (1024**3)
    print(f"Available disk space: {free_gb:.2f} GB")
    
    if free_gb < 3:
        print("  WARNING: Low disk space. Using minimal settings.")
    
    # 1. Data Download & Preprocessing
    print("\n=== Step 1: Data Download & Preprocessing ===")
    train_path, dev_path = download_disfl_qa_data()
    dataset = load_and_prepare_dataset(train_path, dev_path)
    print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['validation'])} validation samples")
    
    # 2. Optimized Data Augmentation
    print("\n=== Step 2: Optimized Data Augmentation ===")
    augmented_dataset = optimized_data_augmentation(dataset, augmentation_factor=0.5)
    
    # 3. OPTIMIZED PERFORMANCE T5 Model Training
    trainer, tokenizer = train_optimized_performance_t5_model(augmented_dataset, model_name="t5-small", num_epochs=4)

    # 4. OPTIMIZED PERFORMANCE T5 Model Evaluation
    bleu_score, avg_sentence_bleu, exact_match_rate, predictions = evaluate_optimized_performance_model(
        trainer, tokenizer, augmented_dataset
    )

    # 5. Optimized Overfitting Analysis
    analyze_optimized_performance_overfitting(trainer)

    # 6. Performance Summary
    print("\n" + "=" * 70)
    print("OPTIMIZED PERFORMANCE T5 RESULTS SUMMARY")
    print("=" * 70)
    print(f"Corpus BLEU Score: {bleu_score:.2f}")
    print(f"Average Sentence BLEU: {avg_sentence_bleu:.4f}")
    print(f"Exact Match Rate: {exact_match_rate:.2f}%")
    print(f"Model: t5-small (60M parameters)")
    print(f"Training: 4 epochs with optimized data augmentation")
    print(f"Model saved to: ./t5_optimized_performance_disfl_qa_rephraser")
    print(f"Results saved to: ./t5_optimized_performance_results/")
    print(f"Training Time: ~1-1.5 hours")
    
    # Performance assessment
    if bleu_score >= 85:
        print(" EXCELLENT: High performance achieved! (85+ BLEU)")
    elif bleu_score >= 80:
        print(" VERY GOOD: Good performance (80+ BLEU)")
    elif bleu_score >= 75:
        print(" GOOD: Solid performance (75+ BLEU)")
    else:
        print(" NEEDS IMPROVEMENT: Below target")
    
    print("=" * 70)

if __name__ == "__main__":
    main() 
# Disfl QA Rephraser – Optimized T5 Model

This repository contains the implementation of a question rephrasing model developed for the Disfl QA benchmark dataset. The goal is to clean disfluent (noisy) questions while preserving their original semantic intent, a common challenge in conversational AI systems.

##  Task Overview

- **Objective**: Rephrase disfluent user questions into fluent, coherent questions.
- **Dataset**: [Disfl QA – Google Research](https://github.com/google-research-datasets/Disfl-QA)
- **Input**: Noisy, disfluent questions
- **Output**: Clean, fluent reformulations

##  Model Overview

Two T5-small models were implemented and compared:
1. **Baseline** – Minimal preprocessing and no augmentation.
2. **Optimized** – Advanced training strategies and 50% data augmentation.

### Architecture:
- Base: `t5-small` (60M parameters)
- Tokenizer: HuggingFace `T5Tokenizer`
- Input Format: `"rephrase: <disfluent_question>"`
- Output Format: `<fluent_question>`

##  Training Configuration

| Configuration       | Baseline Model     | Optimized Model                            | Improvement      |
|---------------------|--------------------|--------------------------------------------|------------------|
| BLEU Score          | 38.5               | 85.80                                      | +47.30           |
| Exact Match Rate    | ~30%               | 59.80%                                     | +29.80%          |
| Epochs              | 3                  | 4                                          | +1               |
| Batch Size          | 8                  | 6 (with gradient accumulation)             | Effective x2     |
| Sequence Length     | 64                 | 96                                         | +32              |
| Data Augmentation   |  None              |  50% Augmented                             | ✓                | 
| Advanced Features   |  None              |  Mixed Precision, Early Stopping, Cosine LR| ✓m               |
| Training Time       | 20 min             | 13 hrs                                     | ↑                |

##  Results

- **Corpus BLEU Score**: 85.80
- **Exact Match Rate**: 59.80%
- **Sample Predictions**: 8/8 perfect matches on validation subset
- **Training Loss**: 0.0696 | **Validation Loss**: 0.0930
- **Loss Gap**: -0.0234 (strong generalization, no overfitting)

###  Figure: Training vs Validation Loss Curve
Loss curves confirm stable training and excellent generalization. Minor loss gap and early stopping indicate no overfitting.

##  Evaluation

Evaluation includes:
- BLEU Score (corpus + sentence-level)
- Exact Match Rate
- High BLEU Rate (score > 0.7)
- Robust tokenizer handling (NLTK fixes applied)

##  Files

- `disfl_qa_rephrase.py`: Baseline training script
- `disfl_qa_rephrase_optimized_performance.py`: Optimized model script
- `requirements.txt`: Python dependencies
- `README.md`: This file
- `./t5_optimized_performance_disfl_qa_rephraser/`: Trained model
- `./t5_optimized_performance_results/`: Training results
- `optimized_performance_t5_loss_curves.png`: Loss plot

## Key Takeaways

- Data augmentation and training optimization significantly boost performance.
- T5-small is a lightweight yet powerful choice for question rephrasing.
- The model is generalizable and ready for real-world conversational AI deployment.
  
## Report
[Final Report – ParisNouri_DisflQA_Rephraser_Report_Chata_MLE.pdf](./ParisNouri_DisflQA_Rephraser_Report_Chata_MLE.pdf)
---

**Author**: Paris Nouri  
**Date**: August 1, 2025


#!/usr/bin/env python3
"""
Comprehensive Model Evaluation and Explainability Script

This script provides detailed evaluation metrics, error analysis, and explainability
features for the trained DeBERTa-v3 preference model.

Usage:
    python src/models/model_evaluation.py --model_path ./llm_preference_model_smart/checkpoint-XXX
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

try:
    from .deberta_test import ConcatenatedPreferenceDataset, load_and_preprocess_data
except ImportError:
    # Allow running as script directly
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.models.deberta_test import ConcatenatedPreferenceDataset, load_and_preprocess_data

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def load_trained_model(model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load a trained model and tokenizer from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint directory
        device: Device to load the model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model from {model_path}...")
    # Use local_files_only to prevent trying to download from Hugging Face Hub
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    model.to(device)
    model.eval()
    print(f"✓ Model loaded on {device}")
    return model, tokenizer


def comprehensive_evaluation(
    model,
    tokenizer,
    dataset,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    """
    Perform comprehensive evaluation with multiple metrics.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        dataset: Evaluation dataset
        device: Device to run inference on
        
    Returns:
        Dictionary with evaluation metrics and predictions
    """
    print("Running inference on evaluation set...")
    from torch.utils.data import DataLoader
    
    # Use batching for much faster inference
    batch_size = 8 if device == "cpu" else 32
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    
    all_logits = []
    all_labels = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 50 == 0:
                print(f"  Processing batch {batch_idx + 1}/{len(dataloader)}...")
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()
            probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
            
            all_logits.append(logits)
            all_probs.append(probs)
            if "labels" in batch:
                all_labels.extend(batch["labels"].cpu().numpy().tolist())
            else:
                all_labels.extend([-1] * len(logits))
    
    all_logits = np.vstack(all_logits)
    all_probs = np.vstack(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.argmax(all_probs, axis=1)
    
    # Filter out test samples (labels == -1) for metrics calculation
    valid_mask = all_labels != -1
    if valid_mask.sum() > 0:
        valid_labels = all_labels[valid_mask]
        valid_preds = all_preds[valid_mask]
        valid_probs = all_probs[valid_mask]
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(valid_labels, valid_preds)
        precision, recall, f1, support = precision_recall_fscore_support(
            valid_labels, valid_preds, average=None, zero_division=0
        )
        macro_f1 = f1_score(valid_labels, valid_preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(valid_labels, valid_preds, average="weighted", zero_division=0)
        
        cm = confusion_matrix(valid_labels, valid_preds)
        
        # Per-class metrics
        class_metrics = {}
        class_names = ["Model A Wins", "Model B Wins", "Tie"]
        for i, class_name in enumerate(class_names):
            if i < len(precision):
                class_metrics[class_name] = {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }
        
        metrics = {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "per_class": class_metrics,
            "confusion_matrix": cm.tolist(),
            "predictions": all_preds.tolist(),
            "probabilities": all_probs.tolist(),
            "labels": all_labels.tolist(),
        }
    else:
        # Test set - no labels available
        metrics = {
            "predictions": all_preds.tolist(),
            "probabilities": all_probs.tolist(),
            "labels": all_labels.tolist(),
        }
    
    return metrics


def plot_confusion_matrix(cm: np.ndarray, save_path: str = "confusion_matrix.png"):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Model A Wins", "Model B Wins", "Tie"],
        yticklabels=["Model A Wins", "Model B Wins", "Tie"],
    )
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Confusion matrix saved to {save_path}")
    plt.close()


def plot_class_metrics(metrics: Dict, save_path: str = "class_metrics.png"):
    """Plot per-class precision, recall, and F1 scores."""
    class_names = list(metrics["per_class"].keys())
    precision = [metrics["per_class"][name]["precision"] for name in class_names]
    recall = [metrics["per_class"][name]["recall"] for name in class_names]
    f1 = [metrics["per_class"][name]["f1"] for name in class_names]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label="Precision", alpha=0.8)
    ax.bar(x, recall, width, label="Recall", alpha=0.8)
    ax.bar(x + width, f1, width, label="F1-Score", alpha=0.8)
    
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Performance Metrics", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Class metrics plot saved to {save_path}")
    plt.close()


def error_analysis(
    df: pd.DataFrame,
    predictions: np.ndarray,
    labels: np.ndarray,
    save_path: str = "error_analysis.csv",
) -> pd.DataFrame:
    """
    Perform error analysis on misclassified samples.
    
    Args:
        df: DataFrame with original data
        predictions: Model predictions
        labels: True labels
        save_path: Path to save error analysis CSV
        
    Returns:
        DataFrame with error analysis
    """
    errors = []
    class_names = ["Model A Wins", "Model B Wins", "Tie"]
    
    for idx in range(len(predictions)):
        if predictions[idx] != labels[idx] and labels[idx] != -1:
            error_info = {
                "index": idx,
                "true_label": class_names[labels[idx]],
                "predicted_label": class_names[predictions[idx]],
                "model_a": df.iloc[idx]["model_a"] if "model_a" in df.columns else "N/A",
                "model_b": df.iloc[idx]["model_b"] if "model_b" in df.columns else "N/A",
                "prompt_length": len(str(df.iloc[idx]["prompt"])),
                "response_a_length": len(str(df.iloc[idx]["response_a"])),
                "response_b_length": len(str(df.iloc[idx]["response_b"])),
            }
            errors.append(error_info)
    
    error_df = pd.DataFrame(errors)
    if len(error_df) > 0:
        error_df.to_csv(save_path, index=False)
        print(f"✓ Error analysis saved to {save_path} ({len(error_df)} errors found)")
    else:
        print("✓ No errors found!")
    
    return error_df


def generate_evaluation_report(metrics: Dict, output_path: str = "evaluation_report.json"):
    """Generate a comprehensive evaluation report in JSON format."""
    report = {
        "overall_metrics": {
            "accuracy": metrics.get("accuracy", "N/A"),
            "macro_f1": metrics.get("macro_f1", "N/A"),
            "weighted_f1": metrics.get("weighted_f1", "N/A"),
        },
        "per_class_metrics": metrics.get("per_class", {}),
        "confusion_matrix": metrics.get("confusion_matrix", []),
    }
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Evaluation report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DeBERTa preference model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint directory",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/train.csv",
        help="Path to evaluation data (default: data/train.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results (default: ./evaluation_results)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Test size for train/val split (default: 0.1)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for quick testing, default: None = all samples)",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_trained_model(args.model_path, device)
    
    # Load and preprocess data
    print("Loading evaluation data...")
    df_train, _ = load_and_preprocess_data(args.data_path, "data/test.csv")
    
    # Use validation split (same as training)
    from sklearn.model_selection import train_test_split
    
    _, df_val = train_test_split(
        df_train, test_size=args.test_size, random_state=42, stratify=df_train["label"]
    )
    
    # Limit samples if specified (for quick testing)
    if args.max_samples and args.max_samples < len(df_val):
        print(f"Limiting evaluation to {args.max_samples} samples for quick testing...")
        df_val = df_val.sample(n=args.max_samples, random_state=42).reset_index(drop=True)
    
    # Create dataset
    val_dataset = ConcatenatedPreferenceDataset(df_val, tokenizer, args.max_length, augment=False)
    
    # Run evaluation
    print("\n" + "=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)
    metrics = comprehensive_evaluation(model, tokenizer, val_dataset, device)
    
    # Print results
    if "accuracy" in metrics:
        print("\n" + "-" * 60)
        print("EVALUATION RESULTS")
        print("-" * 60)
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
        print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")
        print("\nPer-Class Metrics:")
        for class_name, class_metrics in metrics["per_class"].items():
            print(f"  {class_name}:")
            print(f"    Precision: {class_metrics['precision']:.4f}")
            print(f"    Recall: {class_metrics['recall']:.4f}")
            print(f"    F1-Score: {class_metrics['f1']:.4f}")
            print(f"    Support: {class_metrics['support']}")
        
        # Generate visualizations
        cm = np.array(metrics["confusion_matrix"])
        plot_confusion_matrix(cm, str(output_dir / "confusion_matrix.png"))
        plot_class_metrics(metrics, str(output_dir / "class_metrics.png"))
        
        # Error analysis
        error_df = error_analysis(
            df_val,
            np.array(metrics["predictions"]),
            np.array(metrics["labels"]),
            str(output_dir / "error_analysis.csv"),
        )
        
        # Generate report
        generate_evaluation_report(metrics, str(output_dir / "evaluation_report.json"))
        
        print("\n" + "=" * 60)
        print("✓ Evaluation complete! Results saved to:", output_dir)
        print("=" * 60)
    else:
        print("Test set evaluation - no labels available for metrics")


if __name__ == "__main__":
    main()


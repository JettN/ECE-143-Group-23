"""
DeBERTa-v3 Preference Training Script V2 - With Similarity Features

This is Version 2 of the training script that enhances the base DeBERTa model
with similarity features calculated using sentence transformers.

Key improvements over V1:
- Adds semantic similarity features between prompt and responses
- Custom model architecture that combines DeBERTa embeddings with similarity features
- Uses multi-head attention to integrate features

This script is independent and does not modify the original deberta_test.py.
"""

import random
import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorWithPadding,
)
from torch.utils.tensorboard import SummaryWriter
import os
import subprocess
from pathlib import Path
from typing import Optional

# Import our custom modules
from .similarity_features import SimilarityFeatureCalculator, add_similarity_features_to_dataframe
from .deberta_with_similarity import DeBERTaWithSimilarityForSequenceClassification


# ---------------------------------------------------------------------------
# Environment & configuration
# ---------------------------------------------------------------------------

# Load proxy settings from network configuration (optional)
try:
    result = subprocess.run(
        'bash -c "source /etc/network_turbo && env | grep proxy"',
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout
    for line in output.splitlines():
        if "=" in line:
            var, value = line.split("=", 1)
            os.environ[var] = value
except Exception:
    pass

# Hugging Face model to fine-tune
MODEL_NAME = "microsoft/deberta-v3-base"

# Sequence and training hyperparameters
MAX_LENGTH = 2048
BATCH_SIZE = 3
GRAD_ACCUM_STEPS = 5
LEARNING_RATE = 2e-5  # Slightly higher for V2
EPOCHS = 5  # More epochs for V2
PROTOTYPE_FRAC = 1
TEST_SIZE = 0.1

# Training options
GRADIENT_CHECKPOINTING = False
RESUME_FROM_CHECKPOINT = False

# Dataset directories
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

# Output directories (different from V1 to avoid conflicts)
OUTPUT_DIR = "./llm_preference_model_v2_similarity"
TENSORBOARD_DIR = "./tf-logs-v2"
RUN_NAME = "deberta_v2_similarity_features"


def get_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Find the latest checkpoint directory."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    checkpoints = [
        d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")
    ]
    if not checkpoints:
        return None

    checkpoints_with_steps = []
    for ckpt in checkpoints:
        try:
            step = int(ckpt.name.split("-")[1])
            checkpoints_with_steps.append((step, ckpt))
        except (IndexError, ValueError):
            continue

    if not checkpoints_with_steps:
        return None

    latest_checkpoint = max(checkpoints_with_steps, key=lambda x: x[0])[1]
    return str(latest_checkpoint)


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess_data(train_path: str, test_path: str):
    """
    Load and preprocess training and test data.
    Same as V1 but will be enhanced with similarity features later.
    """
    if not Path(train_path).exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not Path(test_path).exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")
    
    print("Loading data...")
    df_train = pd.read_csv(train_path, engine="python")
    df_test = pd.read_csv(test_path, engine="python")
    
    # Parse JSON strings
    list_cols = ["prompt", "response_a", "response_b"]
    for col in list_cols:
        if col in df_train.columns:
            def parse_json_or_string(x):
                if pd.isna(x):
                    return ""
                if isinstance(x, str):
                    try:
                        parsed = json.loads(x)
                        if isinstance(parsed, list):
                            return " ".join(str(item) for item in parsed)
                        return str(parsed)
                    except (json.JSONDecodeError, ValueError):
                        return str(x)
                return str(x)
            
            df_train[col] = df_train[col].apply(parse_json_or_string)
        
        if col in df_test.columns:
            def parse_json_or_string(x):
                if pd.isna(x):
                    return ""
                if isinstance(x, str):
                    try:
                        parsed = json.loads(x)
                        if isinstance(parsed, list):
                            return " ".join(str(item) for item in parsed)
                        return str(parsed)
                    except (json.JSONDecodeError, ValueError):
                        return str(x)
                return str(x)
            
            df_test[col] = df_test[col].apply(parse_json_or_string)
    
    # Convert one-hot encoded winner columns to single label
    # Label mapping: 0 = model_a wins, 1 = model_b wins, 2 = tie
    df_train["label"] = (
        df_train["winner_model_a"] * 0 + df_train["winner_model_b"] * 1 + df_train["winner_tie"] * 2
    )
    
    # Drop any rows with missing text
    initial_size = len(df_train)
    df_train = df_train.dropna(subset=["prompt", "response_a", "response_b"])
    if len(df_train) < initial_size:
        print(f"Warning: {initial_size - len(df_train)} rows dropped due to missing prompt or responses.")
    
    print(f"Full dataset size: {len(df_train)}")
    print(f"Test dataset size: {len(df_test)}")
    
    return df_train, df_test


# ---------------------------------------------------------------------------
# Dataset with Similarity Features
# ---------------------------------------------------------------------------

class ConcatenatedPreferenceDatasetWithSimilarity(Dataset):
    """
    Cross-encoder dataset with similarity features.
    
    Extends the base dataset to include similarity features that are
    calculated using sentence transformers.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = 1024,
        is_test: bool = False,
        augment: bool = False,
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        self.augment = augment

        self.prompts = df["prompt"].values.astype(str)
        self.response_as = df["response_a"].values.astype(str)
        self.response_bs = df["response_b"].values.astype(str)
        
        # Extract similarity features if they exist in the dataframe
        similarity_feature_cols = [
            "prompt_response_a_sim",
            "prompt_response_b_sim",
            "response_a_response_b_sim",
            "similarity_diff",
            "similarity_ratio",
        ]
        
        if all(col in df.columns for col in similarity_feature_cols):
            self.similarity_features = df[similarity_feature_cols].values.astype(np.float32)
            self.has_similarity_features = True
        else:
            # If features don't exist, create zeros (model will handle this)
            self.similarity_features = np.zeros((len(df), 5), dtype=np.float32)
            self.has_similarity_features = False
            if not is_test:
                print("Warning: Similarity features not found in dataframe. Using zeros.")
        
        if not self.is_test:
            self.labels = df["label"].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """Build a single training or inference example with similarity features."""
        prompt = self.prompts[idx]
        resp_a = self.response_as[idx]
        resp_b = self.response_bs[idx]

        if not self.is_test:
            label = self.labels[idx]

        # Data augmentation: randomly swap responses
        if self.augment and random.random() > 0.5:
            resp_a, resp_b = resp_b, resp_a
            if not self.is_test:
                if label == 0:
                    label = 1
                elif label == 1:
                    label = 0
            # Also swap similarity features
            if self.has_similarity_features:
                sim_features = self.similarity_features[idx].copy()
                # Swap prompt-response similarities
                sim_features[0], sim_features[1] = sim_features[1], sim_features[0]
                # Negate similarity_diff
                sim_features[3] = -sim_features[3]
                # Invert similarity_ratio
                sim_features[4] = 1.0 / (sim_features[4] + 1e-8)
                similarity_features = sim_features
            else:
                similarity_features = self.similarity_features[idx]
        else:
            similarity_features = self.similarity_features[idx]

        # Tokenize (same as V1)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        total_special = 4
        available_for_resps = self.max_length - len(prompt_ids) - total_special

        if available_for_resps < 100:
            prompt_ids = prompt_ids[-512:]
            available_for_resps = self.max_length - len(prompt_ids) - total_special

        max_resp_len = available_for_resps // 2

        resp_a_ids = self.tokenizer.encode(resp_a, add_special_tokens=False)[:max_resp_len]
        resp_b_ids = self.tokenizer.encode(resp_b, add_special_tokens=False)[:max_resp_len]

        input_ids = (
            [self.tokenizer.cls_token_id]
            + prompt_ids
            + [self.tokenizer.sep_token_id]
            + resp_a_ids
            + [self.tokenizer.sep_token_id]
            + resp_b_ids
            + [self.tokenizer.sep_token_id]
        )

        len_p = len(prompt_ids) + 2
        len_a = len(resp_a_ids) + 1
        len_b = len(resp_b_ids) + 1
        token_type_ids = [0] * len_p + [1] * (len_a + len_b)

        attention_mask = [1] * len(input_ids)

        out = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "similarity_features": torch.tensor(similarity_features, dtype=torch.float32),
        }

        if not self.is_test:
            out["labels"] = int(label)

        return out


class CustomDataCollator:
    """
    Custom data collator that handles similarity features.
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.padding_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    def __call__(self, features):
        # Separate similarity features
        similarity_features = [f.pop("similarity_features") for f in features]
        
        # Use standard collator for text features
        batch = self.padding_collator(features)
        
        # Add similarity features back
        batch["similarity_features"] = torch.stack(similarity_features)
        
        return batch


class TensorBoardCallback(TrainerCallback):
    """Custom callback for TensorBoard logging."""
    
    def __init__(self, writer: SummaryWriter):
        self.writer = writer

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % 100 == 0 and model is not None:
            total_norm = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2

            total_norm = total_norm**0.5
            self.writer.add_scalar("gradients/total_norm", total_norm, state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        self.writer.close()


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def compute_metrics(p) -> dict:
    """Compute evaluation metrics."""
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    per_class_acc = {}

    for label in range(3):
        mask = labels == label
        if mask.sum() > 0:
            per_class_acc[f"class_{label}_accuracy"] = accuracy_score(labels[mask], preds[mask])

    overall_acc = accuracy_score(labels, preds)
    return {"accuracy": overall_acc, **per_class_acc}


if __name__ == "__main__":
    # Load and preprocess data
    print("=" * 60)
    print("DeBERTa V2 Training with Similarity Features")
    print("=" * 60)
    print("\nLoading and preprocessing data...")
    df_train, df_test = load_and_preprocess_data(TRAIN_PATH, TEST_PATH)

    if PROTOTYPE_FRAC < 1.0:
        print(f"Creating a {PROTOTYPE_FRAC * 100}% stratified prototype dataset...")
        _, df_train = train_test_split(
            df_train,
            test_size=PROTOTYPE_FRAC,
            random_state=42,
            stratify=df_train["label"],
        )
        print(f"Prototype dataset size: {len(df_train)}")

    # Calculate similarity features
    print("\n" + "=" * 60)
    print("Calculating Similarity Features")
    print("=" * 60)
    similarity_calculator = SimilarityFeatureCalculator()
    df_train, _ = add_similarity_features_to_dataframe(df_train, similarity_calculator)
    df_test, _ = add_similarity_features_to_dataframe(df_test, similarity_calculator)

    # Optional: resume from checkpoint
    checkpoint_to_resume = None
    if RESUME_FROM_CHECKPOINT:
        checkpoint_to_resume = get_latest_checkpoint(OUTPUT_DIR)
        if checkpoint_to_resume:
            print(f"Found checkpoint to resume from: {checkpoint_to_resume}")
        else:
            print("No checkpoint found. Starting training from scratch.")

    # Initialize model and tokenizer
    print("\n" + "=" * 60)
    print("Initializing Model")
    print("=" * 60)
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print("Initializing DeBERTa model with similarity features...")
    model = DeBERTaWithSimilarityForSequenceClassification(
        model_name=MODEL_NAME,
        num_labels=3,
        num_similarity_features=5,
        hidden_dropout_prob=0.1,
    )

    tensorboard_log_dir = os.path.join(TENSORBOARD_DIR, RUN_NAME)
    writer = SummaryWriter(log_dir=tensorboard_log_dir)

    # Split data
    print("\nSplitting data into train/validation...")
    train_df, val_df = train_test_split(
        df_train, test_size=TEST_SIZE, random_state=42, stratify=df_train["label"]
    )

    # Create datasets
    train_dataset = ConcatenatedPreferenceDatasetWithSimilarity(
        train_df, tokenizer, MAX_LENGTH, augment=True
    )
    val_dataset = ConcatenatedPreferenceDatasetWithSimilarity(
        val_df, tokenizer, MAX_LENGTH, augment=False
    )
    test_dataset = ConcatenatedPreferenceDatasetWithSimilarity(
        df_test, tokenizer, MAX_LENGTH, is_test=True
    )

    print(f"Training with {len(train_dataset)} samples, validating with {len(val_dataset)} samples.")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="tensorboard",
        logging_dir=tensorboard_log_dir,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        run_name=RUN_NAME,
    )

    # Custom trainer that handles similarity features
    class CustomTrainer(Trainer):
        """Custom trainer that passes similarity features to the model."""
        
        def compute_loss(self, model, inputs, return_outputs=False):
            similarity_features = inputs.pop("similarity_features")
            labels = inputs.pop("labels") if "labels" in inputs else None
            
            outputs = model(
                **inputs,
                similarity_features=similarity_features,
                labels=labels,
            )
            
            return (outputs.loss, outputs) if return_outputs else outputs.loss

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[TensorBoardCallback(writer)],
        data_collator=CustomDataCollator(tokenizer),
    )

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    print(f"Model: {MODEL_NAME} + Similarity Features")
    print(f"Max sequence length: {MAX_LENGTH}")
    print(f"Batch size: {BATCH_SIZE} (gradient accumulation: {GRAD_ACCUM_STEPS})")
    print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("-" * 60)

    trainer.train(resume_from_checkpoint=checkpoint_to_resume)

    # Evaluation
    print("\n" + "=" * 60)
    print("Validation Performance")
    print("=" * 60)
    val_predictions = trainer.predict(val_dataset)
    val_logits = torch.from_numpy(val_predictions.predictions)
    val_probs = F.softmax(val_logits, dim=1).numpy()
    val_preds = np.argmax(val_probs, axis=1)
    val_labels = val_predictions.label_ids

    cm = confusion_matrix(val_labels, val_preds)
    print("Validation Confusion Matrix:")
    print(cm)

    # Test set inference
    print("\n" + "=" * 60)
    print("Generating Test Predictions")
    print("=" * 60)
    predictions = trainer.predict(test_dataset)
    logits = torch.from_numpy(predictions.predictions)
    probs = F.softmax(logits, dim=1).numpy()

    print("Creating submission file...")
    submission_df = pd.DataFrame({"id": df_test["id"]})
    submission_df["winner_model_a"] = probs[:, 0]
    submission_df["winner_model_b"] = probs[:, 1]
    submission_df["winner_tie"] = probs[:, 2]

    submission_df.to_csv("submission_test_v2.csv", index=False)
    print("'submission_test_v2.csv' created.")
    
    writer.close()
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


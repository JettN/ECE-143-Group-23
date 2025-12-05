"""
DeBERTa-v3 Preference Training Script

This module fine-tunes a Hugging Face `microsoft/deberta-v3-base` model
to predict user preferences between two LLM responses (A vs B vs tie).

Key features:
- Uses preprocessed data from `data_preprocessing.py`
- Cross-encoder input format: [CLS] prompt [SEP] response_a [SEP] response_b [SEP]
- Optional response swapping to reduce positional bias
- Hugging Face Trainer API for training, evaluation, and checkpointing
- TensorBoard logging with gradient-norm monitoring
"""

import random
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorWithPadding,
)
from torch.utils.tensorboard import SummaryWriter
import os
import subprocess
from pathlib import Path

import data_preprocessing as dp


# ---------------------------------------------------------------------------
# Environment & configuration
# ---------------------------------------------------------------------------

# Load proxy settings from network configuration.
# This is specific to the environment where the script is executed and ensures
# that any HTTP(S) requests (e.g., to Hugging Face Hub) respect the proxy.
result = subprocess.run(
    'bash -c "source /etc/network_turbo && env | grep proxy"',
    shell=True,
    capture_output=True,
    text=True,
)
output = result.stdout
for line in output.splitlines():
    if "=" in line:
        var, value = line.split("=", 1)
        os.environ[var] = value

# Hugging Face model to fine-tune
MODEL_NAME = "microsoft/deberta-v3-base"

# Sequence and training hyperparameters
MAX_LENGTH = 2048                 # Max tokens for [prompt + responses + special tokens]
BATCH_SIZE = 3                    # Per-device batch size
GRAD_ACCUM_STEPS = 5              # Gradient accumulation for effective larger batch
LEARNING_RATE = 1e-5
EPOCHS = 3
PROTOTYPE_FRAC = 1                # < 1.0 if you want to train on a smaller subset
TEST_SIZE = 0.1                   # Validation fraction for train/val split

# Training options
GRADIENT_CHECKPOINTING = False    # Set True to save memory at cost of compute
RESUME_FROM_CHECKPOINT = False    # Set True to resume from last saved checkpoint

# Output directories
OUTPUT_DIR = "./llm_preference_model_smart"
TENSORBOARD_DIR = "./tf-logs"
RUN_NAME = "trunc_2048_run"


def get_latest_checkpoint(output_dir: str) -> str | None:
    """
    Find the latest checkpoint directory inside `output_dir`.

    Checkpoints are expected to follow the naming convention: 'checkpoint-<step>'.

    Args:
        output_dir: Path to the directory where Hugging Face Trainer saves checkpoints.

    Returns:
        The path to the checkpoint directory with the highest step number,
        or None if no checkpoints are found.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    checkpoints = [
        d for d in output_path.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint-")
    ]
    if not checkpoints:
        return None

    checkpoints_with_steps = []
    for ckpt in checkpoints:
        try:
            step = int(ckpt.name.split("-")[1])
            checkpoints_with_steps.append((step, ckpt))
        except (IndexError, ValueError):
            # Skip directories that don't follow the expected format
            continue

    if not checkpoints_with_steps:
        return None

    latest_checkpoint = max(checkpoints_with_steps, key=lambda x: x[0])[1]
    return str(latest_checkpoint)


# ---------------------------------------------------------------------------
# Data loading & preprocessing (using external preprocessed module)
# ---------------------------------------------------------------------------

print("Loading preprocessed data from data_preprocessing.py...")

# dp.df: training DataFrame with columns like:
# ['id', 'model_a', 'model_b', 'prompt', 'response_a', 'response_b', 'winner']
df_train = dp.df.copy()

# dp.test_data: test DataFrame where prompt/response_* are lists of strings
df_test = dp.test_data.copy()

# Map `winner` -> `label` to match the classification scheme used by the model.
# Preprocessing script: winner = 0 (tie), 1 (model_a wins), 2 (model_b wins)
# Model labels: 0 (A wins), 1 (B wins), 2 (tie)
winner_to_label = {1: 0, 2: 1, 0: 2}
df_train["label"] = df_train["winner"].map(winner_to_label)

# Drop any rows with missing text/labels (defensive cleaning)
df_train = df_train.dropna(subset=["prompt", "response_a", "response_b", "label"])

print(f"Full preprocessed dataset size: {len(df_train)}")
if PROTOTYPE_FRAC < 1.0:
    # Optionally downsample for quick prototyping while keeping label distribution balanced
    print(f"Creating a {PROTOTYPE_FRAC * 100}% stratified prototype dataset...")
    _, df_train = train_test_split(
        df_train,
        test_size=PROTOTYPE_FRAC,
        random_state=42,
        stratify=df_train["label"],
    )
    print(f"Prototype dataset size: {len(df_train)}")

# Ensure prompts/responses are plain strings in the train set
for col in ["prompt", "response_a", "response_b"]:
    df_train[col] = df_train[col].astype(str)

# For df_test, fields may be lists of strings; we join them into single strings
for col in ["prompt", "response_a", "response_b"]:
    df_test[col] = df_test[col].apply(
        lambda x: " ".join(x) if isinstance(x, list) else str(x)
    )

print("Example preprocessed train row:")
print(df_train.iloc[0][["prompt", "response_a", "response_b", "label"]])

print("Example preprocessed test row:")
print(df_test.iloc[0][["prompt", "response_a", "response_b"]])


# ---------------------------------------------------------------------------
# Dataset & callbacks
# ---------------------------------------------------------------------------

class ConcatenatedPreferenceDataset(Dataset):
    """
    Cross-encoder dataset for preference learning between two LLM responses.

    Each example is encoded as a single sequence:
        [CLS] prompt [SEP] response_a [SEP] response_b [SEP]

    This lets the model compare the two responses in the shared context
    of the prompt.

    Data augmentation:
        With 50% probability (when `augment=True`), response A and B are swapped,
        and the label is updated accordingly to reduce positional bias.

    Args:
        df (pd.DataFrame):
            DataFrame containing columns:
                - 'prompt'
                - 'response_a'
                - 'response_b'
                - 'label' (only required for training/validation)
        tokenizer (transformers.PreTrainedTokenizer):
            Tokenizer used to convert text to input IDs.
        max_length (int):
            Maximum total sequence length (including special tokens).
        is_test (bool):
            If True, labels are not expected and will not be returned.
        augment (bool):
            If True, randomly swaps response_a and response_b for data augmentation.
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
        if not self.is_test:
            self.labels = df["label"].values

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """
        Build a single training or inference example.

        Returns:
            A dictionary containing:
                - input_ids: token IDs for the concatenated sequence
                - token_type_ids: segment IDs (0 for prompt, 1 for responses)
                - attention_mask: 1 for real tokens, 0 for padding
                - labels (int): class label (0, 1, or 2) if not in test mode
        """
        prompt = self.prompts[idx]
        resp_a = self.response_as[idx]
        resp_b = self.response_bs[idx]

        if not self.is_test:
            label = self.labels[idx]

        # ------------------------------------------------------------------
        # Data augmentation: randomly swap responses to reduce positional bias
        # ------------------------------------------------------------------
        if self.augment and random.random() > 0.5:
            resp_a, resp_b = resp_b, resp_a
            if not self.is_test:
                if label == 0:      # A wins -> B wins after swap
                    label = 1
                elif label == 1:    # B wins -> A wins after swap
                    label = 0

        # Tokenize the prompt
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        total_special = 4  # [CLS] + 3 x [SEP]
        available_for_resps = self.max_length - len(prompt_ids) - total_special

        # If there is not enough room left for responses, truncate the prompt
        # from the beginning, keeping the most recent context.
        if available_for_resps < 100:
            prompt_ids = prompt_ids[-512:]
            available_for_resps = self.max_length - len(prompt_ids) - total_special

        # Split remaining token budget evenly between response_a and response_b
        max_resp_len = available_for_resps // 2

        resp_a_ids = self.tokenizer.encode(resp_a, add_special_tokens=False)[:max_resp_len]
        resp_b_ids = self.tokenizer.encode(resp_b, add_special_tokens=False)[:max_resp_len]

        # Build the final concatenated sequence
        input_ids = (
            [self.tokenizer.cls_token_id]
            + prompt_ids
            + [self.tokenizer.sep_token_id]
            + resp_a_ids
            + [self.tokenizer.sep_token_id]
            + resp_b_ids
            + [self.tokenizer.sep_token_id]
        )

        # Segment IDs:
        #   0 -> [CLS] + prompt + first [SEP]
        #   1 -> response_a + [SEP] + response_b + [SEP]
        len_p = len(prompt_ids) + 2  # CLS + prompt + first SEP
        len_a = len(resp_a_ids) + 1  # response_a + SEP
        len_b = len(resp_b_ids) + 1  # response_b + SEP
        token_type_ids = [0] * len_p + [1] * (len_a + len_b)

        attention_mask = [1] * len(input_ids)

        out = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }

        if not self.is_test:
            out["labels"] = int(label)

        return out


class TensorBoardCallback(TrainerCallback):
    """
    Custom Hugging Face Trainer callback for logging gradient norms to TensorBoard.

    This is useful for monitoring training stability and detecting exploding
    or vanishing gradients.
    """

    def __init__(self, writer: SummaryWriter):
        """
        Args:
            writer: A TensorBoard `SummaryWriter` instance.
        """
        self.writer = writer

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """
        At the end of certain training steps, compute and log the total gradient norm.

        Logs every 100 global steps to keep overhead small.
        """
        if state.global_step % 100 == 0 and model is not None:
            total_norm = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2

            total_norm = total_norm**0.5
            self.writer.add_scalar("gradients/total_norm", total_norm, state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        """Close the TensorBoard writer when training is complete."""
        self.writer.close()


# ---------------------------------------------------------------------------
# Model, Trainer, and training loop
# ---------------------------------------------------------------------------

# Optional: resume from the latest checkpoint if enabled.
checkpoint_to_resume = None
if RESUME_FROM_CHECKPOINT:
    checkpoint_to_resume = get_latest_checkpoint(OUTPUT_DIR)
    if checkpoint_to_resume:
        print(f"Found checkpoint to resume from: {checkpoint_to_resume}")
    else:
        print("No checkpoint found. Starting training from scratch.")

print("Initializing model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

tensorboard_log_dir = os.path.join(TENSORBOARD_DIR, RUN_NAME)
writer = SummaryWriter(log_dir=tensorboard_log_dir)

print("Splitting data into train/validation...")
train_df, val_df = train_test_split(
    df_train, test_size=TEST_SIZE, random_state=42, stratify=df_train["label"]
)

train_dataset = ConcatenatedPreferenceDataset(train_df, tokenizer, MAX_LENGTH, augment=True)
val_dataset = ConcatenatedPreferenceDataset(val_df, tokenizer, MAX_LENGTH, augment=False)
test_dataset = ConcatenatedPreferenceDataset(df_test, tokenizer, MAX_LENGTH, is_test=True)

print(f"Training with {len(train_dataset)} samples, validating with {len(val_dataset)} samples.")


def compute_metrics(p) -> dict:
    """
    Compute evaluation metrics for Hugging Face Trainer.

    Args:
        p: An `EvalPrediction` object containing:
            - predictions: raw logits of shape (num_examples, num_classes)
            - label_ids: ground-truth labels of shape (num_examples,)

    Returns:
        A dictionary with:
            - 'accuracy': overall accuracy
            - 'class_0_accuracy': accuracy for class 0 (A wins), if present
            - 'class_1_accuracy': accuracy for class 1 (B wins), if present
            - 'class_2_accuracy': accuracy for class 2 (tie), if present
    """
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    per_class_acc = {}

    for label in range(3):
        mask = labels == label
        if mask.sum() > 0:
            per_class_acc[f"class_{label}_accuracy"] = accuracy_score(labels[mask], preds[mask])

    overall_acc = accuracy_score(labels, preds)
    return {"accuracy": overall_acc, **per_class_acc}


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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[TensorBoardCallback(writer)],
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)

print("Starting training...")
trainer.train(resume_from_checkpoint=checkpoint_to_resume)

# ---------------------------------------------------------------------------
# Evaluation on validation set
# ---------------------------------------------------------------------------

print("Analyzing validation performance...")
val_predictions = trainer.predict(val_dataset)
val_logits = torch.from_numpy(val_predictions.predictions)
val_probs = F.softmax(val_logits, dim=1).numpy()
val_preds = np.argmax(val_probs, axis=1)
val_labels = val_predictions.label_ids

cm = confusion_matrix(val_labels, val_preds)
print("Validation Confusion Matrix:")
print(cm)

# ---------------------------------------------------------------------------
# Inference on test set & submission file
# ---------------------------------------------------------------------------

print("Generating predictions on test set...")
predictions = trainer.predict(test_dataset)
logits = torch.from_numpy(predictions.predictions)
probs = F.softmax(logits, dim=1).numpy()

print("Creating submission file...")
submission_df = pd.DataFrame({"id": df_test["id"]})
submission_df["winner_model_a"] = probs[:, 0]
submission_df["winner_model_b"] = probs[:, 1]
submission_df["winner_model_tie"] = probs[:, 2]

submission_df.to_csv("submission_test.csv", index=False)
print("'submission_test.csv' created.")
writer.close()

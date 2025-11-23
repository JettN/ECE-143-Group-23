"""
DEBUG TRAINING SCRIPT for DeBERTa-v3 on Kaggle LLM Preference Task

- Uses a smaller random subset of the preprocessed training set (dp.df)
- Uses microsoft/deberta-v3-base as a 3-class classifier:
    0: tie
    1: model_a
    2: model_b
- Runs for 1 epoch (same as full script)
- Generates submission_debug.csv
- Saves the trained debug model + tokenizer to ./saved_deberta_model_debug

Pipeline and logic are the same as the full script; only the number
of training samples is reduced for faster iteration.
"""

import sys
# Force tqdm to use plain text progress bars (not HTML) in IPython-like envs
sys.modules["IPython"] = None

import os
os.environ["DISABLE_IPYTHON"] = "1"

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)
import transformers

transformers.utils.logging.set_verbosity_info()

# Preprocessing code
import data_preprocessing as dp


# ============================================================
# DATASET CLASS
# ============================================================
class PreferenceDataset(Dataset):
    """
    Dataset for the preference classification task.

    Each item returns:
        - combined text of prompt + response_a + response_b
        - label (0=tie, 1=model_a, 2=model_b) for training
    """

    def __init__(self, df, tokenizer, max_length=256, is_train=True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def _build_text(self, row):
        return (
            f"[PROMPT] {row['prompt']}\n"
            f"[RESPONSE_A] {row['response_a']}\n"
            f"[RESPONSE_B] {row['response_b']}"
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = self._build_text(row)

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)

        if self.is_train:
            item["labels"] = torch.tensor(int(row["winner"]), dtype=torch.long)

        return item


# ============================================================
# METRICS + CALLBACK
# ============================================================
def compute_metrics(eval_pred):
    """
    Compute simple accuracy over the validation set.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean().item()
    return {"accuracy": acc}


class EpochPrinter(TrainerCallback):
    """
    Simple callback to print when each epoch starts/ends.
    """

    def on_epoch_begin(self, args, state, control, **kwargs):
        if state.epoch is not None:
            print(f"\n🔵 Starting Epoch {int(state.epoch) + 1}")

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch is not None:
            print(f"🟢 Finished Epoch {int(state.epoch)}\n")


# ============================================================
# MAIN PIPELINE (DEBUG VERSION)
# ============================================================
def main():
    """
    Main entry point: train model on a smaller subset, evaluate, save model,
    and create a debug submission file.
    """

    print("Reloaded preprocessing modules.")

    train_df = dp.df.copy()
    test_df = dp.test_data.copy()

    print("Sample of test_df:")
    print(test_df.head(3))

    # Convert list-of-strings to a single string for test set
    for col in ["prompt", "response_a", "response_b"]:
        test_df[col] = test_df[col].apply(
            lambda x: " ".join(x) if isinstance(x, list) else str(x)
        ).fillna("")

    # --------------------------------------------------------
    # DEBUG: Use fewer samples for faster training
    # --------------------------------------------------------
    SEED = 42
    DEBUG_N = 2000  # number of training samples to use for debug

    if len(train_df) > DEBUG_N:
        train_df = train_df.sample(n=DEBUG_N, random_state=SEED)
        print(f"\nDEBUG MODE: Using a random subset of {DEBUG_N} training samples.\n")
    else:
        print(f"\nDEBUG MODE: Train dataset smaller than DEBUG_N, using all {len(train_df)} rows.\n")

    # --------------------------------------------------------
    # Config (same as full script except for smaller dataset)
    # --------------------------------------------------------
    MODEL_NAME = "microsoft/deberta-v3-base"

    MAX_LENGTH = 384        # keep same as full script for consistency
    BATCH_SIZE = 6          # same as full script
    NUM_EPOCHS = 1          # same as full script
    LR = 2e-5

    set_seed(SEED)

    print(f"DEBUG TRAINING CONFIG:")
    print(f"  MAX_LENGTH = {MAX_LENGTH}")
    print(f"  BATCH_SIZE = {BATCH_SIZE}")
    print(f"  NUM_EPOCHS = {NUM_EPOCHS}")
    print(f"  Train rows (debug subset) = {len(train_df)}\n")

    # --------------------------------------------------------
    # Train/Val split
    # --------------------------------------------------------
    train_split, val_split = train_test_split(
        train_df,
        test_size=0.1,
        random_state=SEED,
        stratify=train_df["winner"],
    )

    print("Train samples:", len(train_split))
    print("Val samples:", len(val_split))
    print("Test samples:", len(test_df))

    # --------------------------------------------------------
    # Load tokenizer + model
    # --------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    label2id = {"tie": 0, "model_a": 1, "model_b": 2}
    id2label = {v: k for k, v in label2id.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        label2id=label2id,
        id2label=id2label,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"\nCUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Model running on device: {model.device}\n")

    # --------------------------------------------------------
    # Build datasets
    # --------------------------------------------------------
    train_dataset = PreferenceDataset(train_split, tokenizer, MAX_LENGTH, True)
    val_dataset = PreferenceDataset(val_split, tokenizer, MAX_LENGTH, True)
    test_dataset = PreferenceDataset(test_df, tokenizer, MAX_LENGTH, False)

    # --------------------------------------------------------
    # Training arguments (same as full, different output dir)
    # --------------------------------------------------------
    training_args = TrainingArguments(
        output_dir="./deberta_output_debug",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        logging_steps=50,
        logging_strategy="steps",
        disable_tqdm=False,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,  # Windows-safe
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EpochPrinter()],
    )

    # --------------------------------------------------------
    # Train & evaluate
    # --------------------------------------------------------
    print("\n🚀 Starting DEBUG training...\n")
    trainer.train()
    print("\n🎉 DEBUG training finished!\n")

    print("Evaluating on validation set...")
    val_results = trainer.evaluate()
    print("Validation results:", val_results)

    # --------------------------------------------------------
    # SAVE TRAINED DEBUG MODEL + TOKENIZER
    # --------------------------------------------------------
    save_dir = "./saved_deberta_model_debug"
    os.makedirs(save_dir, exist_ok=True)

    trainer.save_model(save_dir)          # saves model + config
    tokenizer.save_pretrained(save_dir)   # saves tokenizer files

    print(f"\nSaved debug trained model + tokenizer to {save_dir}\n")

    # --------------------------------------------------------
    # Predict test set → debug submission
    # --------------------------------------------------------
    print("Running prediction on test set...")
    preds = trainer.predict(test_dataset).predictions
    probs = torch.softmax(torch.tensor(preds), dim=-1).numpy()

    submission = pd.DataFrame({
        "id": test_df["id"].values,
        "winner_model_a": probs[:, label2id["model_a"]],
        "winner_model_b": probs[:, label2id["model_b"]],
        "winner_tie":     probs[:, label2id["tie"]],
    })

    out_name = "submission_debug.csv"
    submission.to_csv(out_name, index=False)
    print(f"Saved {out_name}!")


if __name__ == "__main__":
    main()

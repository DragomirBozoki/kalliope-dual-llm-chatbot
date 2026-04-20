"""
Intent Classification Training Script
--------------------------------------
Fine-tunes a multilingual DistilBERT model for sequence classification
using a JSONL dataset. Supports multi-language preprocessing and
mixed-precision training via HuggingFace Transformers Trainer API.
"""

import os
import traceback

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from config.labels import id2label, label2id
from config.preprocessing import TextPreprocessor

# ---------------------------------------------------------------------------
# Device Setup
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA not available, running on CPU.")

# ---------------------------------------------------------------------------
# Preprocessor & Tokenizer
# ---------------------------------------------------------------------------

MODEL_NAME = "distilbert-base-multilingual-cased"
MAX_LENGTH = 64
DATASET_PATH = "QuestionChatBot/dataset/*.jsonl"
OUTPUT_DIR = "models/intent-multi-model_save_epoch"
FINAL_MODEL_PATH = "models/intent-medium-model-multilang-v2.0"

preprocessor = TextPreprocessor(language="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

try:
    dataset = load_dataset("json", data_files=DATASET_PATH)
    print(f"Loaded dataset: {len(dataset['train'])} samples.")
except Exception:
    traceback.print_exc()
    raise RuntimeError(f"Failed to load dataset from '{DATASET_PATH}'. Verify path and file integrity.")

# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_fn(examples: dict) -> dict:
    """
    Preprocesses and tokenizes instruction+input pairs.
    Maps output label strings to numeric class indices.
    """
    texts = [
        f"{preprocessor.preprocess(instr)} {preprocessor.preprocess(inp)}"
        for instr, inp in zip(examples["instruction"], examples["input"])
    ]

    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

    tokenized["labels"] = [label2id[label] for label in examples["output"]]

    # Cast to int32 for NumPy/PyTorch compatibility
    tokenized = {key: np.asarray(val, dtype=np.int32) for key, val in tokenized.items()}

    return tokenized


print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_fn, batched=True)
print("Tokenization complete.")

# ---------------------------------------------------------------------------
# Model Initialization
# ---------------------------------------------------------------------------

print(f"Loading model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
).to(device)
print("Model loaded.")

# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=20,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    logging_steps=50,
    logging_dir="./logs",
    report_to="tensorboard",
    fp16=torch.cuda.is_available(),   # Mixed precision only when GPU is available
    dataloader_num_workers=4,
    gradient_accumulation_steps=1,
    disable_tqdm=False,
)

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
    # Uncomment to enable checkpoint saving and prediction logging:
    # callbacks=[
    #     SaveEveryNEpochsCallback(),
    #     PrintPredictionCallback(tokenizer, model, tokenized_dataset["train"], print_every_n_epoch=5),
    # ],
)

# ---------------------------------------------------------------------------
# Training & Model Export
# ---------------------------------------------------------------------------

print("Starting training...")
trainer.train()
print("Training complete.")

trainer.save_model(FINAL_MODEL_PATH)
print(f"Model saved to '{FINAL_MODEL_PATH}'.")

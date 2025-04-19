# ========== Imports ==========
from datasets import load_dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback
)

from config.labels import label2id, id2label  # Output labels mapping

import torch
import os
import random
from collections import defaultdict
from callbacks import SaveEveryNEpochsCallback


# ========== Tokenizer & Model Name ==========
model_name = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ========== Load Dataset ==========
dataset = load_dataset("json", data_files="./Chatgpt_testground/QuestionChatBot/dataset/*.jsonl")

# ========== Reverse Mapping for Model Output ==========
id2label = {v: k for k, v in label2id.items()}

# ========== Tokenization Function ==========
def tokenize_fn(examples):
    texts = [f"{instr} {inp}" for instr, inp in zip(examples["instruction"], examples["input"])]
    tokenized = tokenizer(texts, truncation=True, padding="max_length", max_length=16)
    tokenized["labels"] = [label2id[label] for label in examples["output"]]
    return tokenized

# Apply tokenization
tokenized_dataset = dataset.map(tokenize_fn, batched=True)

# ========== Create a Small Balanced Subset for Testing ==========
n_per_class = 5
class_to_examples = defaultdict(list)

# Group samples by class
for i, example in enumerate(tokenized_dataset["train"]):
    label = example["labels"]
    class_to_examples[label].append(i)

# Sample up to N examples per class
selected_indices = []
for indices in class_to_examples.values():
    selected_indices.extend(random.sample(indices, min(len(indices), n_per_class)))

# Create small subset (for quick testing)
small_dataset = tokenized_dataset["train"].select(selected_indices)

# ========== Initialize Model ==========
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# ========== Training Arguments ==========
training_args = TrainingArguments(
    output_dir="models/intent-multi-model_save_epoch",
    num_train_epochs=50,
    per_device_train_batch_size=2,
    learning_rate=2e-5,
    logging_steps=10,
    logging_dir="./logs",  # For TensorBoard
    save_strategy="no",
    evaluation_strategy="no",
    report_to="tensorboard",  # Enable TensorBoard logging
)

# ========== Trainer ==========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
    callbacks=[SaveEveryNEpochsCallback(save_every_n_epochs=10, output_dir="models/intent-multi-model_save_epoch")]
)

# ========== Train ==========
trainer.train()

# ========== Final Save ==========
trainer.save_model("models/intent-medium-model-multilang-v2.0")
print("✅ Training completed and model saved!")

# ========== Imports ==========
from datasets import load_dataset
import traceback
from config.preprocessing import TextPreprocessor
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
from config.callbacks import PrintPredictionCallback, SaveEveryNEpochsCallback

# ========== Preprocessor & Tokenizer Initialization ==========
preprocessor = TextPreprocessor(language="auto")
model_name = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ========== Load Dataset ==========
try:
    dataset = load_dataset("json", data_files="dataset/*.jsonl")
except Exception as e:
    traceback.print_exc()
    raise e  # Important to stop training if dataset loading fails

# ========== Tokenization Function ==========
def tokenize_fn(examples):
    # Preprocess and concatenate instruction + input fields
    texts = [
        f"{preprocessor.preprocess(instr)} {preprocessor.preprocess(inp)}"
        for instr, inp in zip(examples['instruction'], examples['input'])
    ]

    # Tokenize using pre-trained BERT tokenizer
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=16  # Adjust if needed based on your data
    )

    # Convert textual output labels to numeric class indices
    tokenized["labels"] = [label2id[label] for label in examples["output"]]
    return tokenized

# ========== Tokenize Dataset ==========
tokenized_dataset = dataset.map(tokenize_fn, batched=True)

# ========== Model Initialization ==========
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# ========== Training Arguments ==========
training_args = TrainingArguments(
    output_dir="models/intent-multi-model_save_epoch",  # Where to save checkpoints
    num_train_epochs=50,
    per_device_train_batch_size=2,
    learning_rate=2e-5,
    logging_steps=10,
    logging_dir="./logs",  # TensorBoard logging directory
    save_strategy="no",  # We use custom callback instead
    evaluation_strategy="no",  # Turn off evaluation if not needed
    report_to="tensorboard"  # Enable TensorBoard integration
)

# ========== Trainer Initialization ==========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
    callbacks=[
        SaveEveryNEpochsCallback(
            save_every_n_epochs=10,
            output_dir="models/intent-multi-model_save_epoch"
        ),
        PrintPredictionCallback(tokenizer, model, dataset["train"], print_every_n_epoch=5)
    ]
)

# ========== Train the Model ==========
trainer.train()

# ========== Save Final Model ==========
trainer.save_model("models/intent-medium-model-multilang-v2.0")
print("✅ Training completed and model saved!")
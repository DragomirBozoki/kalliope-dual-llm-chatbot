# ========== Imports ==========
import os
import random
import numpy as np
import torch
import traceback
from collections import defaultdict
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from config.preprocessing import TextPreprocessor
from config.labels import label2id, id2label  # Output labels mapping
from config.callbacks import PrintPredictionCallback, SaveEveryNEpochsCallback

# ========== CUDA Check ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ CUDA Available: {torch.cuda.is_available()}")
print(f"🖥️ Current Device: {torch.cuda.current_device()}")
print(f"🚀 GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# ========== Preprocessor & Tokenizer Initialization ==========
preprocessor = TextPreprocessor(language="auto")
model_name = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ========== Load Dataset ==========
try:
    dataset = load_dataset("json", data_files="QuestionChatBot/dataset/*.jsonl")
    print(f"📊 Loaded dataset with {len(dataset['train'])} samples.")
except Exception as e:
    print("❌ Failed to load the dataset. Check if the path is correct and the files are not empty.")
    traceback.print_exc()
    raise e  # Stop execution if dataset loading fails

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
        max_length=64
    )

    # Convert textual output labels to numeric class indices
    tokenized["labels"] = [label2id[label] for label in examples["output"]]

    # Ensure NumPy compatibility
    for key in tokenized.keys():
        tokenized[key] = np.asarray(tokenized[key], dtype=np.int32)

    return tokenized

# ========== Tokenize Dataset ==========
print("🔄 Tokenizing the dataset...")
tokenized_dataset = dataset.map(tokenize_fn, batched=True)
print("✅ Tokenization completed.")

# ========== Model Initialization ==========
print("🧠 Initializing the model...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
).to(device)
print("✅ Model initialized.")

# ========== Training Arguments ==========
training_args = TrainingArguments(
    output_dir="models/intent-multi-model_save_epoch",
    num_train_epochs=20,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    logging_steps=50,
    logging_dir="./logs",
    report_to="tensorboard",
    fp16=True,  # Enable mixed precision for faster training on GPU
    dataloader_num_workers=4,
    disable_tqdm=False,
    gradient_accumulation_steps=1
)

# ========== Trainer Initialization ==========
print("🛠️ Initializing the Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
    #callbacks=[
    #    SaveEveryNEpochsCallback(),
    #    PrintPredictionCallback(tokenizer, model, tokenized_dataset["train"], print_every_n_epoch=5)
    #]
)
print("✅ Trainer initialized.")

# ========== Train the Model ==========
print("🚀 Starting training...")
trainer.train()
print("✅ Training completed.")

# ========== Save Final Model ==========
model_save_path = "models/intent-medium-model-multilang-v2.0"
trainer.save_model(model_save_path)
print(f"✅ Model saved at '{model_save_path}'")

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

from config.callbacks import PrintPredictionCallback, SaveEveryNEpochsCallback
from config.labels import id2label, label2id
from config.preprocessing import TextPreprocessor

DATASET_PATH = "QuestionChatBot/dataset/*.jsonl"
MODEL_NAME = "distilbert-base-multilingual-cased"
OUTPUT_DIR = "models/intent-multi-model_save_epoch"
FINAL_MODEL_PATH = "models/intent-medium-model-multilang-v2.0"
MAX_SEQ_LENGTH = 64
NUM_EPOCHS = 25
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
LOGGING_STEPS = 50


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("Using CPU")
    return device


def load_training_data(path: str):
    try:
        dataset = load_dataset("json", data_files=path)
        print(f"Loaded {len(dataset['train'])} samples")
        return dataset
    except Exception as e:
        print(f"Failed to load dataset from '{path}'")
        traceback.print_exc()
        raise


def build_tokenize_fn(tokenizer, preprocessor):
    def tokenize(examples):
        texts = [
            f"{preprocessor.preprocess(instr)} {preprocessor.preprocess(inp)}"
            for instr, inp in zip(examples["instruction"], examples["input"])
        ]
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
        )
        tokenized["labels"] = [label2id[label] for label in examples["output"]]
        for key in tokenized:
            tokenized[key] = np.asarray(tokenized[key], dtype=np.int32)
        return tokenized

    return tokenize


def train():
    device = get_device()

    preprocessor = TextPreprocessor(language="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dataset = load_training_data(DATASET_PATH)
    tokenize_fn = build_tokenize_fn(tokenizer, preprocessor)
    tokenized_dataset = dataset.map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_steps=LOGGING_STEPS,
        logging_dir="./logs",
        report_to="tensorboard",
        fp16=True,
        dataloader_num_workers=4,
        disable_tqdm=False,
        gradient_accumulation_steps=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(FINAL_MODEL_PATH)
    print(f"Model saved to '{FINAL_MODEL_PATH}'")


if __name__ == "__main__":
    train()

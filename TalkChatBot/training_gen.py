import os
import warnings

import torch
from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
    logging,
)

warnings.filterwarnings("ignore", category=UserWarning)
logging.set_verbosity_error()

MODEL_NAME = "gpt2"
DATA_PATH = "TalkChatBot/dataset/train_dataset/esdalab_english.jsonl"
OUTPUT_DIR = "./models/gpt2-kalliopev2.0"
MAX_SEQ_LENGTH = 128
NUM_EPOCHS = 150
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
WARMUP_STEPS = 30
WEIGHT_DECAY = 0.01
COSINE_CYCLES = 2
MAX_GRAD_NORM = 1.0
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 2
LOGGING_STEPS = 10


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    return device


def load_tokenizer_and_model(model_name: str, device: torch.device):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    return tokenizer, model


def build_tokenize_fn(tokenizer):
    def tokenize(examples):
        merged = [
            f"{inst.strip()} {inp.strip()} {out.strip()}"
            for inst, inp, out in zip(
                examples["instruction"], examples["input"], examples["output"]
            )
        ]
        tokenized = tokenizer(
            merged,
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return tokenize


def create_optimizer_and_scheduler(model, dataset_size: int):
    steps_per_epoch = dataset_size // BATCH_SIZE
    total_steps = steps_per_epoch * NUM_EPOCHS

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps,
        num_cycles=COSINE_CYCLES,
    )
    return optimizer, scheduler


def train():
    device = get_device()
    tokenizer, model = load_tokenizer_and_model(MODEL_NAME, device)

    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    print(f"Loaded {len(dataset)} samples")

    tokenize_fn = build_tokenize_fn(tokenizer)
    tokenized_dataset = dataset.map(
        tokenize_fn, batched=True, remove_columns=dataset.column_names
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    optimizer, scheduler = create_optimizer_and_scheduler(model, len(tokenized_dataset))

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        report_to="tensorboard",
        fp16=False,
        save_total_limit=SAVE_TOTAL_LIMIT,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        remove_unused_columns=True,
        gradient_accumulation_steps=1,
        disable_tqdm=False,
        max_grad_norm=MAX_GRAD_NORM,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),
    )

    print("Starting training...")
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to '{OUTPUT_DIR}'")


if __name__ == "__main__":
    train()

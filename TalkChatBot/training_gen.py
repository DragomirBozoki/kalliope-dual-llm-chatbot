print("🚀 Starting GPT-2 training...\n")
import os
import warnings
import torch
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
    DataCollatorForLanguageModeling,
    logging,
)

warnings.filterwarnings("ignore", category=UserWarning)
logging.set_verbosity_error()

# ========== Configuration ==========
MODEL_NAME = "gpt2"
DATA_PATH = "TalkChatBot/dataset/train_dataset/esdalab_english.jsonl"
OUTPUT_DIR = "./models/gpt2-kalliopev2.0"
MAX_LEN = 128
EPOCHS = 150  # 70 epoha za mali dataset (~200 pitanja)
BATCH_SIZE = 4
LEARNING_RATE = 1e-5 
WARMUP_STEPS = 30
print("epochs: 30")
# ========== Load Model & Tokenizer ==========
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).cuda()
tokenizer.pad_token = tokenizer.eos_token  # važno za padding

# ========== Load Dataset ==========
print("📦 Loading dataset...")
dataset = load_dataset(
    "json",
    data_files=DATA_PATH,
    split="train"
)

print(f"✅ Loaded {len(dataset)} samples.")
print("🧾 Example:")
print(dataset[0])

# ========== Tokenization ==========
def tokenize_fn(examples):
    # Spajanje instruction + input + output
    merged = [f"{inst.strip()} {inp.strip()} {output.strip()}" for inst, inp, output in zip(examples["instruction"], examples["input"], examples["output"])]
    tokenized = tokenizer(
        merged,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

# ========== Collator ==========
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ========== Optimizer & Scheduler ==========
steps_per_epoch = len(tokenized_dataset) // BATCH_SIZE
total_steps = steps_per_epoch * EPOCHS

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_steps,
    num_cycles=2
)

# ========== Training Arguments ==========
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    report_to="tensorboard",
    fp16=False,
    save_total_limit=2,
    logging_steps=10,
    save_steps=500,
    remove_unused_columns=True,
    gradient_accumulation_steps=1,
    disable_tqdm=False,
    max_grad_norm=1.0,
)

# ========== Trainer ==========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    optimizers=(optimizer, scheduler),
)

print("\n🧠 All systems ready. Starting training...\n")
trainer.train()

# ========== Save ==========
print("\n💾 Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ Training complete! Model saved.")

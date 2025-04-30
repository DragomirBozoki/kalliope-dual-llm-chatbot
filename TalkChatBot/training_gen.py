# ========== Imports ==========
import os

from datasets import load_dataset
from transformers import (
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from datetime import datetime
from config.callbacks import SaveEveryNEpochsCallback
from config.preprocessing import TextPreprocessor


# ========== Load Model & Tokenizer ==========
preprocessor = TextPreprocessor(language="auto")
model_name = "google/mt5-small"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

# ========== Load Dataset ==========
dataset = load_dataset("json", data_files="./dataset/questions.jsonl")

def tokenize_fn(examples):
    # Preprocess and concatenate instruction + input fields
    inputs = [
        f"{preprocessor.preprocess(instr)} {preprocessor.preprocess(inp)}"
        for instr, inp in zip(examples['instruction'], examples['input'])
    ]

    # Tokenize using pre-trained BERT tokenizer
    model_inputs = tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=64  # Adjust if needed based on your data
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["output"],
            max_length = 64,
            padding = "max_length",
            truncation = True
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(tokenize_fn, batched=True)

# ========== TrainingArguments ==========
training_args = TrainingArguments(
    output_dir="./models/mt5-generative-save-epochs",
    num_train_epochs=50,
    per_device_train_batch_size=2,
    learning_rate=3e-4,
    logging_dir="./logs/tb_mt5_" + datetime.now().strftime("%Y%m%d-%H%M%S"),
    logging_steps=10,
    save_strategy="no",  # saving handled manually via callback
    evaluation_strategy="no",
    report_to="tensorboard"
)

# ========== Trainer ==========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
    callbacks=[
        SaveEveryNEpochsCallback(save_every_n_epochs=10, output_dir="./models/mt5-generative-save-epochs")
    ]
)

# ========== Train ==========
trainer.train()

# ========== Save Final Model ==========
trainer.save_model("./models/mt5-generative-final")
tokenizer.save_pretrained("./models/mt5-generative-final")
print("🎉 Training completed and model saved!")

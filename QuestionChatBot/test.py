import torch
import subprocess
import pandas as pd

from transformers import AutoModelForSequenceClassification, AutoTokenizer, MT5Tokenizer, MT5ForConditionalGeneration
from config.logfile import init_log_file
from config.logger import log_entry
from Chatgpt_testground.QuestionChatBot.config.labels import label2id, id2label, reminder_keywords

# ---------- Init ----------
init_log_file()

# Typo dictionary
typo_df = pd.read_csv("typos_multilang_extended.csv")
typo_dict = dict(zip(typo_df["typo"], typo_df["correct"]))

def correct_typos(text):
    words = text.split()
    corrected = [typo_dict.get(w.lower(), w) for w in words]
    return " ".join(corrected)

# Load intent classification model
model_path = "./intent-medium-model-multilang-v2.0"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Load generative model (MT5)
gen_model_path = "./output_mt5"
gen_tokenizer = MT5Tokenizer.from_pretrained(gen_model_path)
gen_model = MT5ForConditionalGeneration.from_pretrained(gen_model_path)

# Instruction for classification
instruction = "Classify the intent of the following sentence."

# Start Kalliope in background
subprocess.Popen(["kalliope", "start"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Main loop
while True:
    print("Write the command (or type 'exit' to quit):")
    text = input("> ").strip()

    if text.lower() == "exit":
        print("Quitting...")
        break

    corrected_text = correct_typos(text.lower())
    input_text = f'{instruction} {corrected_text}'
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=32)

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)
    predicted_class_id = torch.argmax(probabilities, dim=1).item()
    predicted_label = id2label[predicted_class_id]
    prob = probabilities.max().item()

    if any(kw in corrected_text for kw in reminder_keywords) and predicted_label not in ["remember-synapse", "remember-todo"]:
        predicted_label = "remember-synapse"

    print(f"Input: '{text}' → Corrected: '{corrected_text}'")
    print(f"→ Predicted instruction: '{predicted_label}' (Confidence: {prob:.4f})")

    gen_response = ""

    if prob < 0.5:
        print("⚠️ Not confident. Using MT5...")
        gen_input = gen_tokenizer(corrected_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        gen_output = gen_model.generate(**gen_input, max_length=128, num_beams=4, early_stopping=True)
        gen_response = gen_tokenizer.decode(gen_output[0], skip_special_tokens=True)
        print("🤖 MT5 says:", gen_response)
    else:
        print(f"✅ Confident. Kalliope will run synapse: {predicted_label}")
        subprocess.run(["kalliope", "start", "--run-synapse", predicted_label])

    print("-" * 60)

    # Log response
    log_entry(text, corrected_text, predicted_label, f"{prob:.4f}", gen_response)

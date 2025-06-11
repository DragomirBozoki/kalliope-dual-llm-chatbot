import os
import json
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from QuestionChatBot.config.labels import id2label, reminder_keywords

# Paths
BASE_PATH = "/mnt/c/users/bozok/onedrive/desktop/kalliope/Chatgpt_testground"
INTENT_MODEL_PATH = os.path.join(BASE_PATH, "models", "intent-medium-model-multilang-v2.0")
FAISS_INDEX_PATH = os.path.join(BASE_PATH, "faiss/esdalab_index.bin")
FAISS_TEXTS_PATH = os.path.join(BASE_PATH, "faiss/esdalab_texts.json")
DATASET_PATH = os.path.join(BASE_PATH, "TalkChatBot/dataset/train_dataset/esdalab_english.jsonl")

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT classifier
tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL_PATH).to(DEVICE)
model.eval()

# Load FAISS index
index = faiss.read_index(FAISS_INDEX_PATH)
with open(FAISS_TEXTS_PATH, "r") as f:
    texts = json.load(f)
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Load dataset entries for final answers
entries = []
with open(DATASET_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            entries.append(json.loads(line))

def classify_text(text):
    input_text = f"Classify the intent of the following sentence. {text}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        predicted_class_id = torch.argmax(probs, dim=1).item()
        predicted_label = id2label[predicted_class_id]
        confidence = probs.max().item()
    return predicted_label, confidence

def faiss_search(query):
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec), 1)
    closest_q = texts[I[0][0]]
    distance = D[0][0]
    closest_answer = next(
        (entry["output"] for entry in entries if entry["input"].lower() == closest_q.lower()),
        f"I found something similar: {closest_q}"
    )
    return closest_answer

def process_audio_input(text):
    """
    Main function to process the recognized audio text.
    Returns:
        - predicted_label (e.g. "kitchen-on") if high confidence
        - or "generate" as order + FAISS answer as message (for generate.yml)
    """
    predicted_label, confidence = classify_text(text)

    # Check for reminder words and force reminder-synapse if needed
    if any(kw in text for kw in reminder_keywords) and predicted_label not in ["remember-synapse", "remember-todo"]:
        predicted_label = "remember-synapse"

    # If high confidence and not generative, return predicted label
    if confidence >= 0.9 and predicted_label != "generative":
        return predicted_label
    else:
        response = faiss_search(text)
        fallback_data = {
            "order": "say-dynamic-message",
            "text": response
        }
        
        return fallback_data



# Example usage
import sys

if __name__ == "__main__":
    result = process_audio_input(text)
    if isinstance(result, dict):
        print(json.dumps(result))  # for CLI
    else:
        print(result)



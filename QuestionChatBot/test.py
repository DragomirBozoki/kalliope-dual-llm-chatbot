# -*- coding: utf-8 -*-

import os
import sys
import torch
import json
import faiss
import numpy as np
import subprocess
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
)
from config.labels import id2label, reminder_keywords

# Paths
BASE_PATH = os.path.expanduser("~/kalliope_starter_gr/Chatgpt_testground")
INTENT_MODEL_PATH = os.path.join(BASE_PATH, "models", "intent-medium-model-multilang-v2.0")
FAISS_INDEX_PATH = os.path.join(BASE_PATH, "faiss/esdalab_index.bin")
FAISS_TEXTS_PATH = os.path.join(BASE_PATH, "faiss/esdalab_texts.json")
DATASET_PATH = os.path.join(BASE_PATH, "TalkChatBot/dataset/train_dataset/esdalab_english.jsonl")

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_intent_model():
    tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL_PATH).to(DEVICE)
    return tokenizer, model

def load_faiss():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_TEXTS_PATH, "r") as f:
        texts = json.load(f)
    embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return index, texts, embedder

def load_dataset_entries():
    entries = []
    with open(DATASET_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries

def generate_response(query, index, texts, embedder, dataset_entries):
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec), 1)
    closest_q = texts[I[0][0]]
    distance = D[0][0]

    closest_answer = next(
        (entry["output"] for entry in dataset_entries if entry["input"].lower() == closest_q.lower()),
        None
    )

    if closest_answer:
        return closest_answer
    else:
        return f"I found something similar: {closest_q}"

def main():
    tokenizer, intent_model = load_intent_model()
    index, texts, embedder = load_faiss()
    dataset_entries = load_dataset_entries()

    print("🎉 All models and data successfully loaded. Ready for interaction!\n")

    while True:
        text = input("You: ").strip()
        if text.lower() == "exit":
            print("👋 Goodbye!")
            break

        input_text = f"Classify the intent of the following sentence. {text}"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(DEVICE)
        with torch.no_grad():
            outputs = intent_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        class_id = torch.argmax(probs, dim=1).item()
        predicted_label = id2label[class_id]
        confidence = probs.max().item()

        if any(kw in text for kw in reminder_keywords) and predicted_label not in ["remember-synapse", "remember-todo"]:
            predicted_label = "remember-synapse"

        if confidence >= 0.9 and predicted_label != "generative":
            print(f"Kalliope will now run: {predicted_label}")
            subprocess.run(["kalliope", "start", "--run-synapse", predicted_label])
        else:
            if index:
                try:
                    response = generate_response(text, index, texts, embedder, dataset_entries)
                    # Ovo je ključno: Kalliope će pročitati samo linije koje počinju
                    print("Kalliope will now run: generate.yml")
                    print(f"SAY::{response}")
                except Exception:
                    print("SAY::I don't know.")
            else:
                print("SAY::I don't know.")

if __name__ == "__main__":
    main()

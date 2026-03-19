# -*- coding: utf-8 -*-
import json
import os
import subprocess
import sys

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config.labels import id2label, reminder_keywords

BASE_PATH = os.path.expanduser("~/kalliope_starter_gr/Chatgpt_testground")
INTENT_MODEL_PATH = os.path.join(BASE_PATH, "models", "intent-medium-model-multilang-v2.0")
FAISS_INDEX_PATH = os.path.join(BASE_PATH, "faiss/esdalab_index.bin")
FAISS_TEXTS_PATH = os.path.join(BASE_PATH, "faiss/esdalab_texts.json")
DATASET_PATH = os.path.join(BASE_PATH, "TalkChatBot/dataset/train_dataset/esdalab_english.jsonl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.9
FAISS_TOP_K = 1
INTENT_MAX_LENGTH = 128
INTENT_PROMPT_PREFIX = "Classify the intent of the following sentence."
FALLBACK_RESPONSE = "I don't know."


def load_intent_model():
    tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL_PATH).to(DEVICE)
    model.eval()
    return tokenizer, model


def load_faiss():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_TEXTS_PATH, "r") as f:
        texts = json.load(f)
    embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return index, texts, embedder


def load_dataset_entries(path: str) -> list[dict]:
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def classify_intent(text: str, tokenizer, model) -> tuple[str, float]:
    input_text = f"{INTENT_PROMPT_PREFIX} {text}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=INTENT_MAX_LENGTH,
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1)
    class_id = torch.argmax(probs, dim=1).item()
    confidence = probs.max().item()
    label = id2label[class_id]

    return label, confidence


def apply_reminder_override(text: str, label: str) -> str:
    if label in ("remember-synapse", "remember-todo"):
        return label
    if any(kw in text for kw in reminder_keywords):
        return "remember-synapse"
    return label


def retrieve_answer(query: str, index, texts, embedder, dataset_entries) -> str:
    query_vec = embedder.encode([query])
    distances, indices = index.search(np.array(query_vec), FAISS_TOP_K)

    matched_text = texts[indices[0][0]]
    answer = next(
        (entry["output"] for entry in dataset_entries if entry["input"].lower() == matched_text.lower()),
        None,
    )
    return answer if answer else f"I found something similar: {matched_text}"


def handle_query(text, tokenizer, intent_model, index, texts, embedder, dataset_entries):
    label, confidence = classify_intent(text, tokenizer, intent_model)
    label = apply_reminder_override(text, label)

    if confidence >= CONFIDENCE_THRESHOLD and label != "generative":
        print(f"Kalliope will now run: {label}")
        subprocess.run(["kalliope", "start", "--run-synapse", label])
        return

    if index is None:
        print(f"SAY::{FALLBACK_RESPONSE}")
        return

    try:
        response = retrieve_answer(text, index, texts, embedder, dataset_entries)
        print("Kalliope will now run: generate.yml")
        print(f"SAY::{response}")
    except Exception:
        print(f"SAY::{FALLBACK_RESPONSE}")


def main():
    tokenizer, intent_model = load_intent_model()
    index, texts, embedder = load_faiss()
    dataset_entries = load_dataset_entries(DATASET_PATH)

    print("All models loaded. Ready for interaction.\n")

    while True:
        text = input("You: ").strip()
        if text.lower() == "exit":
            print("Goodbye.")
            break
        handle_query(text, tokenizer, intent_model, index, texts, embedder, dataset_entries)


if __name__ == "__main__":
    main()

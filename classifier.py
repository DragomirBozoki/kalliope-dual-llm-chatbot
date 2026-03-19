import json
import os
import sys

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from QuestionChatBot.config.labels import id2label, reminder_keywords

BASE_PATH = "/mnt/c/users/bozok/onedrive/desktop/kalliope/Chatgpt_testground"
INTENT_MODEL_PATH = os.path.join(BASE_PATH, "models", "intent-medium-model-multilang-v2.0")
FAISS_INDEX_PATH = os.path.join(BASE_PATH, "faiss/esdalab_index.bin")
FAISS_TEXTS_PATH = os.path.join(BASE_PATH, "faiss/esdalab_texts.json")
DATASET_PATH = os.path.join(BASE_PATH, "TalkChatBot/dataset/train_dataset/esdalab_english.jsonl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.9
INTENT_MAX_LENGTH = 128
INTENT_PROMPT_PREFIX = "Classify the intent of the following sentence."
FAISS_TOP_K = 1


class IntentClassifier:

    def __init__(self, model_path: str, device: torch.device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        self.model.eval()
        self.device = device

    def classify(self, text: str) -> tuple[str, float]:
        input_text = f"{INTENT_PROMPT_PREFIX} {text}"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=INTENT_MAX_LENGTH,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=1)
        class_id = torch.argmax(probs, dim=1).item()
        return id2label[class_id], probs.max().item()


class FAISSRetriever:

    def __init__(self, index_path: str, texts_path: str, dataset_path: str):
        self.index = faiss.read_index(index_path)
        self.embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        with open(texts_path, "r") as f:
            self.texts = json.load(f)

        self.entries = self._load_dataset(dataset_path)
        self._answer_lookup = {
            entry["input"].lower(): entry["output"] for entry in self.entries
        }

    @staticmethod
    def _load_dataset(path: str) -> list[dict]:
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

    def search(self, query: str) -> str:
        query_vec = self.embedder.encode([query])
        _, indices = self.index.search(np.array(query_vec), FAISS_TOP_K)
        matched_text = self.texts[indices[0][0]]
        return self._answer_lookup.get(
            matched_text.lower(),
            f"I found something similar: {matched_text}",
        )


def apply_reminder_override(text: str, label: str) -> str:
    if label in ("remember-synapse", "remember-todo"):
        return label
    if any(kw in text for kw in reminder_keywords):
        return "remember-synapse"
    return label


def process_audio_input(
    text: str,
    classifier: IntentClassifier,
    retriever: FAISSRetriever,
) -> str | dict:
    label, confidence = classifier.classify(text)
    label = apply_reminder_override(text, label)

    if confidence >= CONFIDENCE_THRESHOLD and label != "generative":
        return label

    return {
        "order": "say-dynamic-message",
        "text": retriever.search(text),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python audio_processor.py <text>", file=sys.stderr)
        sys.exit(1)

    text = " ".join(sys.argv[1:])

    classifier = IntentClassifier(INTENT_MODEL_PATH, DEVICE)
    retriever = FAISSRetriever(FAISS_INDEX_PATH, FAISS_TEXTS_PATH, DATASET_PATH)

    result = process_audio_input(text, classifier, retriever)

    if isinstance(result, dict):
        print(json.dumps(result))
    else:
        print(result)


if __name__ == "__main__":
    main()

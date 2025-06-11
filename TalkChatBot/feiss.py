from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import faiss
import numpy as np
import json

# Load dataset
dataset = load_dataset("json", data_files="TalkChatBot/dataset/train_dataset/esdalab_english.jsonl")["train"]

# Load SentenceTransformer model
encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Encode all inputs
sentences = dataset["input"]
embeddings = encoder.encode(sentences, show_progress_bar=True)

# Build FAISS index
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save index + data
faiss.write_index(index, "faiss/esdalab_index.bin")
with open("faiss/esdalab_texts.json", "w") as f:
    json.dump(sentences, f)

print("✅ FAISS index built and saved.")

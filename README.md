# STILL IN PROGRESS

This repository is part of a larger Kalliope voice assistant system and enhances its capabilities with a hybrid NLP pipeline that combines intent classification, generative answers, and semantic search with FAISS.
# 🧠 Models

    Instruction Model (Classifier)
    Classifies user commands into predefined Kalliope synapses (e.g., "Turn on kitchen light" → kitchen-on).

    Natural Response Model (Generator)
    Generates natural, human-like answers to open-ended or factual questions (e.g., "Who is your professor?" → "My professor is Nikolaos Voros.").

    FAISS Semantic Search (Retriever)
    Provides fast retrieval of similar questions and answers from a knowledge base (faq.csv or faq.jsonl), enabling the generative model to deliver more context-aware and relevant responses.

---

## 🧩 Project Structure

DualLLM_Multilang_Chatbot/
├── CommandAI/
│   ├── train.py                # Training script for the classifier
│   ├── test.py                 # Evaluation/inference for the classifier
│   ├── config/
│   │   ├── labels.py           # label2id and id2label mappings
│   └── dataset/
│       ├── *.jsonl             # Dataset for classification
│
├── GenAI/
│   ├── training.py             # Training script for the generative model
│   ├── test.py                 # Evaluation/inference for the generative model
│   └── dataset/
│       └── questions.jsonl     # Dataset for generative responses
│
├── faiss_retrieval.py          # FAISS retriever for context-based retrieval
├── main.py                     # Hybrid runtime: classification + generation + FAISS
├── requirements.txt            # Required libraries
└── README.md                   # This file

# ⚙️ Hybrid Logic

🔹 If the classifier predicts a known intent with high confidence (e.g., >0.7):
➡️ Triggers the appropriate Kalliope synapse.

🔹 If the classifier confidence is low:
➡️ FAISS retriever searches the knowledge base for the most similar question.
➡️ The generative model (mT5) uses the retrieved context to generate a natural, informative response.
User Input	Step 1	Step 2
Turn on the kitchen lights	Classifier	Runs Kalliope synapse: kitchen-on
What is embedded systems design?	Classifier (low confidence) + FAISS	Generator answers naturally using context

---

# 💬 Dataset Formats
Intent Classification – dataset/*.jsonl
{"instruction": "Classify the intent of the following sentence.", "input": "Turn on the fan", "output": "fan-on"}

Generative Question-Answering – dataset/questions.jsonl
{"instruction": "Answer the user's question about the lab.", "input": "Who is your supervisor?", "output": "My supervisor is Professor Nikolaos Voros."}

---
# ⚙️ How to Use
Step 1: Install dependencies

pip install -r requirements.txt

Ensure Kalliope is already installed and configured.
Step 2: Train the Classifier Model

python CommandAI/train.py

Step 3: Train the Generative Model

python GenAI/training.py

Step 4: Run the Hybrid NLP Engine

python main.py

Example interaction:

> Turn on the fan
[Classifier] → fan-on (confidence: 0.89)
🧠 Running synapse 'fan-on'

> Who is your supervisor?
[Classifier] → UNKNOWN (confidence: 0.34)
🔎 Searching knowledge base with FAISS...
🤖 Generator answer: "My supervisor is Professor Nikolaos Voros."

🌍 Multilingual Support

Thanks to xlm-roberta-base and mT5, the system natively supports English and Greek questions and commands.
# 🧠 Tech Stack

    HuggingFace transformers

    datasets for JSONL loading

    torch (PyTorch backend)

    faiss for semantic search

    kalliope voice assistant platform

    xlm-roberta-base for intent classification

    google/mt5-small for text generation

    tensorboard for training tracking

---

# 📄 License

MIT License – feel free to use, adapt, and integrate it in your projects!
👤 Author

Developed by Dragomir Bozoki, Faculty of Technical Sciences – Signal Processing,
as part of an Erasmus research exchange at the University of Peloponnese, Patras, Greece.

This module brings intelligent, multilingual dialogue to open-source voice assistants, bridging structured command classification with natural, generative answers – all powered by hybrid transformer-based models.

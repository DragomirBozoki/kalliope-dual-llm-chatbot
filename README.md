# 🤖 Kalliope LLM Hybrid NLP Module

# ***STILL IN PROGRESS***

This repository is part of a **larger Kalliope voice assistant system**. It contains a hybrid pipeline combining two multilingual transformer-based models:

### 🧠 Models Used
1. **Instruction Model (Classifier)** – Classifies user speech into predefined intent labels (e.g., `"turn on kitchen light" → kitchen-on`).
2. **Natural Response Model (Generator)** – Answers open-ended or factual user questions with human-like responses (e.g., `"Who is your professor?" → "My professor is Nikolaos Voros.")`.

Both models are integrated into one test script, allowing dynamic interaction depending on the user's input.

---

## 🧩 Project Structure

DualLLM_Multilang_Chatbot
│
│
├── CommandAI/
│   ├── train.py                # Training script for classification
│   ├── test.py                 # Evaluation/inference for classification
│   ├── config/
│   │   ├── labels.py           # Contains label2id and id2label mappings
│   └── dataset/
│       ├── *.jsonl             # Training data files with instruction/input/output format
│
├── GenAI/
│   ├── training.py             # Training script for generative model
│   ├── test.py                 # Evaluation/inference for generative model
│   └── dataset/
│       └── questions.jsonl     # Input-output pairs for generative training
│──requirments.txt              # Required libaries
│
└── README.md                   # This file
```

---

## 🧠 Hybrid Logic

During execution, the system uses the classifier model to determine whether a command maps to a known Kalliope intent. If the prediction confidence is high (e.g., > 0.70), it runs the appropriate Kalliope synapse. Otherwise, the generative model answers naturally.

### Examples:
| User Input                                 | Model Used      | Action                                  |
|-------------------------------------------|------------------|------------------------------------------|
| `Turn on the fan`                         | Classifier       | → Runs `fan-on` synapse in Kalliope      |
| `Remind me to drink tea in 30 seconds`    | Classifier       | → Runs `remember-synapse`               |
| `Who is your professor?`                  | Generator (mT5)  | → Answers: "My professor is ..."        |
| `What is embedded systems design?`        | Generator (mT5)  | → Answers: "It is the field of ..."     |

---

## ⚙️ How to Use

### Step 1: Install dependencies
```bash
pip install transformers datasets torch tensorboard protobuf==3.20.*

```
Make sure Kalliope is already installed and configured.

### Step 2: Train the Classifier Model
```bash
python training.py
```
This will:
- Load the `.jsonl` dataset
- Balance the examples per class - optional if you are using a subset
- Fine-tune an `xlm-roberta-base` classifier
- Save it to disk for use in `test.py`

### Step 3: Run the Hybrid NLP Engine
```bash
python test.py
```
Type any command or question:
```
> Turn on the kitchen lights
[Classifier] → kitchen-on (confidence: 0.89)
🧠 Running synapse 'kitchen-on'

> Who is your supervisor?
[Classifier] → UNKNOWN (confidence: 0.34)
🤖 Generating natural answer...
💬 My supervisor is Professor Nikolaos Voros.
```

---

## 💬 Dataset Formats

### Intent Classification – `dataset/*.jsonl`
```json
{"instruction": "Classify the intent of the following sentence.", "input": "Turn on the fan", "output": "fan-on"}
```

### Generative Question-Answering – `dataset/questions.jsonl`
```json
{"instruction": "Answer the user's question about the lab.", "input": "Who is your supervisor?", "output": "My supervisor is Professor Nikolaos Voros."}
```

---

## 🌍 Multilingual Support
Thanks to `xlm-roberta-base` and `mT5`, this system supports both **English and Greek** questions and commands.

---

## 🧠 Tech Stack
- `transformers` by HuggingFace
- `datasets` for JSONL loading
- `torch` (PyTorch backend)
- `kalliope` voice assistant platform
- `xlm-roberta-base` for intent classification
- `google/mt5-small` for text generation
- `tensorboard` for tracking training

---

## 📄 License
MIT License – open to use, adapt, and integrate.

---

## 👤 Author
Made by **Dragomir Bozoki**, Faculty of Technical Sciences – Signal Processing,
as part of an Erasmus research exchange at the **University of Peloponnese**, Patras, Greece.

This hybrid NLP module bridges command classification and natural language answers inside Kalliope.
It aims to bring intelligent, multilingual dialogue to open-source voice assistants.

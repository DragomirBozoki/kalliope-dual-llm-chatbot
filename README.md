# Kalliope Dual-LLM Chatbot

A multilingual intent classification and semantic fallback system for the [Kalliope](https://kalliope-project.github.io/) voice assistant platform. Developed by Dragomir Bozoki in collaboration with the University of Patras and ESDA Lab.

---

## Table of Contents

- [Overview](#overview)
- [Available Intents](#available-intents)
- [Installation](#installation)
- [Training the Classifier](#training-the-classifier)
- [Building the FAISS Index](#building-the-faiss-index)
- [Architecture](#architecture)
  - [Classification Workflow](#classification-workflow)
  - [Semantic Fallback with FAISS](#semantic-fallback-with-faiss)
  - [Kalliope Integration](#kalliope-integration)
- [Adding a New Intent](#adding-a-new-intent)
- [Module Reference](#module-reference)
- [Installing Snowboy on Python 3.8+](#installing-snowboy-on-python-38)
- [Author](#author)

---

## Overview

The system combines a fine-tuned transformer-based intent classifier, semantic retrieval via FAISS, and optional generative fallback to enable robust understanding of spoken commands in smart environments.

**Key capabilities:**

- Multilingual intent classification using fine-tuned DistilBERT
- Semantic fallback when classification confidence is low
- Dynamic generative responses for queries beyond the trained intent set
- Modular architecture for straightforward extension with new intents

The system is designed for deployment in domestic, educational, and research contexts where rigid command matching is insufficient.

---

## Available Intents

| Label | ID | Description |
|---|---|---|
| `lab-on` | 0 | Turn on lab lights |
| `lab-off` | 1 | Turn off lab lights |
| `Meeting-on` | 2 | Turn on meeting room lights |
| `Meeting-off` | 3 | Turn off meeting room lights |
| `kitchen-on` | 4 | Turn on kitchen lights |
| `kitchen-off` | 5 | Turn off kitchen lights |
| `Livingroom-on` | 6 | Turn on living room lights |
| `Livingroom-off` | 7 | Turn off living room lights |
| `Livingroom-dim` | 8 | Dim the living room lights |
| `room-on` | 9 | Turn on general room lights |
| `room-off` | 10 | Turn off general room lights |
| `reading-on` | 11 | Turn on reading lights |
| `reading-off` | 12 | Turn off reading lights |
| `ambient-random` | 13 | Play random ambient sounds |
| `ambient-stop` | 14 | Stop ambient sounds |
| `ambient-specific` | 15 | Play specific ambient sound |
| `ambient-sleep` | 16 | Play sleep ambient sound |
| `find-my-phone` | 17 | Locate phone |
| `findkeys` | 18 | Locate keys |
| `run-web-radio` | 19 | Start web radio |
| `run-web-radio2` | 20 | Start alternative web radio |
| `stop-web-radio-stop-web-radio2` | 21 | Stop all web radios |
| `exting` | 22 | Emergency exit command |
| `check-email` | 23 | Check emails |
| `news-sport` | 24 | Get sports news |
| `run-web-esda` | 25 | Open ESDA website |
| `close-web-esda` | 26 | Close ESDA website |
| `goodbye` | 27 | Say goodbye |
| `dinner` | 28 | Announce dinner time |
| `apartment` | 29 | Apartment mode command |
| `sonos-play` | 30 | Play music on Sonos |
| `sonos-stop` | 31 | Stop music on Sonos |
| `fan-on` | 32 | Turn on fan |
| `fan-off` | 33 | Turn off fan |
| `door-on` | 34 | Open door |
| `Temperature-set` | 35 | Set temperature |
| `fan-lab1` | 36 | Turn on fan in lab 1 |
| `fan-lab2` | 37 | Turn on fan in lab 2 |
| `room-on1` | 38 | Turn on room 1 lights |
| `room-off2` | 39 | Turn off room 2 lights |
| `kitchen-on1` | 40 | Turn on kitchen zone 1 |
| `kitchen-off1` | 41 | Turn off kitchen zone 1 |
| `saytemp` | 42 | Say current temperature |
| `get-the-weather` | 43 | Get weather report |
| `say-local-date` | 44 | Say today's date |
| `say-local-date-from-template` | 45 | Say date from preset template |
| `tea-time` | 46 | Notify tea time |
| `remember-synapse` | 47 | Store memory command |
| `remember-todo` | 48 | Add to to-do list |
| `generative` | 49 | Fallback to generative response |

---

## Installation

```bash
git clone https://github.com/DragomirBozoki/kalliope-dual-llm-chatbot
cd kalliope-dual-llm-chatbot

python3 -m venv kalliope_venv
source kalliope_venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

The trained model files must be placed manually under `models/intent-medium-model-multilang-v2.0/`. These files are excluded from version control via `.gitignore` due to their size.

---

## Training the Classifier

The intent classifier is a fine-tuned multilingual transformer (DistilBERT) that maps input text to predefined command labels.

### Dataset Format

Training data is stored as JSONL files in `QuestionChatBot/dataset/`. Each line follows this structure:

```json
{
  "instruction": "Classify the intent of the following sentence.",
  "input": "turn on the kitchen light please",
  "output": "kitchen-on"
}
```

Intents can be organized into separate files (e.g., `lights.jsonl`, `ambient.jsonl`, `radio.jsonl`).

### Running Training

```bash
python3 QuestionChatBot/training.py
```

The trained model is saved to `models/intent-medium-model-multilang-v2.0/`.

---

## Building the FAISS Index

FAISS provides semantic fallback when the classifier's confidence falls below the threshold (0.9). It retrieves the most similar question from the dataset and returns the associated answer.

### Retrieval Dataset

Stored in `TalkChatBot/dataset/train_dataset/esdalab_english.jsonl` using the same JSONL format as the training data.

### Building the Index

```python
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

with open("TalkChatBot/dataset/train_dataset/esdalab_english.jsonl") as f:
    entries = [json.loads(line) for line in f]
    questions = [entry["input"] for entry in entries]

embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = embedder.encode(questions)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

faiss.write_index(index, "faiss/esdalab_index.bin")
with open("faiss/esdalab_texts.json", "w") as f:
    json.dump(questions, f, indent=2)
```

---

## Architecture

### Classification Workflow

1. **Speech-to-Text**: Kalliope captures audio and transcribes it via a speech recognition engine.
2. **Intent Classification**: The transcribed text is passed through the fine-tuned transformer. If confidence is at or above 0.9 and the predicted label is not `generative`, the label is returned directly.
3. **Fallback**: If confidence is below the threshold, semantic retrieval is triggered.

### Semantic Fallback with FAISS

When the classifier is not confident, FAISS retrieves the most semantically similar query from the prepared dataset. If a match is found, its associated response is returned. Otherwise, a dynamic fallback message is generated.

### Kalliope Integration

Each classified label corresponds to a synapse in the Kalliope brain configuration:

```yaml
- name: "kitchen-on"
  signals:
    - order: "kitchen-on"
  neurons:
    - say:
        message: "Turning on the kitchen lights."
    - shell:
        cmd: "python3 scripts/kitchen_on.py"
```

For generative fallback, a dedicated synapse handles dynamic messages:

```yaml
- name: "say-dynamic-message"
  signals:
    - order: "say-dynamic-message"
  neurons:
    - shell:
        cmd: "/path/to/dynamic_message_script.py"
        var: "text"
        stdout_output: true
    - say:
        message: "{{ text }}"
```

The core processing logic:

```python
from classifier import classify_text, faiss_search

def process_audio_input(text):
    predicted_label, confidence = classify_text(text)

    if confidence >= 0.9:
        return predicted_label
    else:
        response = faiss_search(text)
        return {"order": "say-dynamic-message", "text": response}
```

---

## Adding a New Intent

1. **Update the label map** in `config/labels.py`:

    ```python
    label2id = {
        # ...existing labels...
        "new-intent-name": 50
    }
    ```

2. **Add training examples** to a JSONL file in `QuestionChatBot/dataset/`:

    ```json
    {
      "instruction": "Classify the intent of the following sentence.",
      "input": "Turn on the garden lights",
      "output": "new-intent-name"
    }
    ```

3. **Retrain the model**:

    ```bash
    python3 QuestionChatBot/training.py
    ```

4. **Add a Kalliope synapse** in `brain.yml`:

    ```yaml
    - name: "new-intent-name"
      signals:
        - order: "new-intent-name"
      neurons:
        - say:
            message: "Turning on the garden lights."
    ```

---

## Module Reference

| Module | Description |
|---|---|
| `training.py` | Trains the intent classifier on JSONL-formatted datasets |
| `classifier.py` | Loads the model and returns predictions with confidence scores |
| `faiss_search.py` | Performs semantic similarity search using SentenceTransformer |
| `process_audio_input.py` | Manages classification logic and fallback decisions |
| `brain.yml` | Maps output labels to executable Kalliope synapses |
| `dynamic_message_script.py` | Returns fallback message for natural language response |

---

## Installing Snowboy on Python 3.8+

Snowboy is no longer officially maintained and requires manual patching for Python 3.8+.

```bash
pip uninstall snowboy

git clone https://github.com/Kitt-AI/snowboy.git
cd snowboy/swig/Python
python3 setup.py build
python3 setup.py install
```

Alternatively, install from a precompiled wheel:

```bash
pip install https://github.com/johnbriant/snowboy-whl/raw/main/snowboy-1.3.0-py3-none-any.whl
```

> **Note:** Snowboy is incompatible with Python 3.11+. Consider [Porcupine](https://picovoice.ai/platform/porcupine/) as an alternative wake word engine.

---

## Author

**Dragomir Bozoki**
bozokidragomir@gmail.com

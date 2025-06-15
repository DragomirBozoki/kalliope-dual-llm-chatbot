### 📑 Table of Contents

    📑   Overview

    🧠 Available Intents

    🔧 Setup Instructions

    🛠️ Installation

    🎓 Training the Classifier

    🚀 Building the FAISS Index (Semantic Fallback)

    📦 Classifier Architecture and Integration with Kalliope

        1. Classification Workflow

        2. Semantic Fallback with FAISS

        3. Kalliope Integration

    🧩 How to Add a New Intent Category

    🧊 Installing Snowboy on Python 38

    👤 Author

---

## Overview

This project implements a multilingual intent classification and fallback system for voice assistants, designed and developed by Dragomir Bozoki in collaboration with the University of Patras and the ESDA Lab. It integrates a fine-tuned transformer-based classifier, semantic retrieval using FAISS, and optional generative fallback mechanisms into the open-source Kalliope voice assistant platform.
Objective

The primary goal of this project is to enable robust and flexible understanding of spoken commands in smart environments. The system is capable of classifying user intents with high accuracy and gracefully handling unexpected or ambiguous queries through semantic or generative fallback mechanisms.
Motivation and Usefulness

Voice assistants are increasingly used in smart homes, assistive technologies, and context-aware systems. However, most assistants rely on rigid command matching and often fail to interpret unfamiliar phrasing or multilingual input. This project addresses that gap by offering:

    Multilingual support based on fine-tuned transformer models

    Semantic fallback when classification confidence is low

    Dynamic generative responses for queries beyond the trained set

    Easy extensibility through modular dataset updates

The resulting system is highly adaptable and suitable for deployment in diverse environments, including educational, domestic, and research contexts.

## 🧠 Available Intents

Below is the full list of supported intent labels used by the multilingual classifier:

| Label                             | ID  | Description                                 |
|----------------------------------|-----|---------------------------------------------|
| `lab-on`                         | 0   | Turn on lab lights                          |
| `lab-off`                        | 1   | Turn off lab lights                         |
| `Meeting-on`                     | 2   | Turn on meeting room lights                 |
| `Meeting-off`                    | 3   | Turn off meeting room lights                |
| `kitchen-on`                     | 4   | Turn on kitchen lights                      |
| `kitchen-off`                    | 5   | Turn off kitchen lights                     |
| `Livingroom-on`                  | 6   | Turn on living room lights                  |
| `Livingroom-off`                 | 7   | Turn off living room lights                 |
| `Livingroom-dim`                 | 8   | Dim the living room lights                  |
| `room-on`                        | 9   | Turn on general room lights                 |
| `room-off`                       | 10  | Turn off general room lights                |
| `reading-on`                     | 11  | Turn on reading lights                      |
| `reading-off`                    | 12  | Turn off reading lights                     |
| `ambient-random`                 | 13  | Play random ambient sounds/music            |
| `ambient-stop`                   | 14  | Stop ambient sounds/music                   |
| `ambient-specific`               | 15  | Play specific ambient sound                 |
| `ambient-sleep`                  | 16  | Play sleep ambient sound                    |
| `find-my-phone`                  | 17  | Locate my phone                             |
| `findkeys`                       | 18  | Locate my keys                              |
| `run-web-radio`                  | 19  | Start web radio                             |
| `run-web-radio2`                 | 20  | Start alternative web radio                 |
| `stop-web-radio-stop-web-radio2` | 21  | Stop all web radios                         |
| `exting`                         | 22  | Execute emergency exit command              |
| `check-email`                    | 23  | Check emails                                |
| `news-sport`                     | 24  | Get latest sports news                      |
| `run-web-esda`                   | 25  | Open ESDA website                           |
| `close-web-esda`                 | 26  | Close ESDA website                          |
| `goodbye`                        | 27  | Say goodbye                                 |
| `dinner`                         | 28  | Announce or prepare dinner time             |
| `apartment`                      | 29  | Apartment mode command                      |
| `sonos-play`                     | 30  | Play music on Sonos                         |
| `sonos-stop`                     | 31  | Stop music on Sonos                         |
| `fan-on`                         | 32  | Turn on the fan                             |
| `fan-off`                        | 33  | Turn off the fan                            |
| `door-on`                        | 34  | Open the door                               |
| `Temperature-set`                | 35  | Set the temperature                         |
| `fan-lab1`                       | 36  | Turn on fan in lab 1                        |
| `fan-lab2`                       | 37  | Turn on fan in lab 2                        |
| `room-on1`                       | 38  | Turn on room 1 lights                       |
| `room-off2`                      | 39  | Turn off room 2 lights                      |
| `kitchen-on1`                    | 40  | Turn on kitchen zone 1                      |
| `kitchen-off1`                   | 41  | Turn off kitchen zone 1                     |
| `saytemp`                        | 42  | Say current temperature                     |
| `get-the-weather`                | 43  | Get weather report                          |
| `say-local-date`                 | 44  | Say today's date                            |
| `say-local-date-from-template`   | 45  | Say date using a preset template            |
| `tea-time`                       | 46  | Notify tea time                             |
| `remember-synapse`               | 47  | Store memory command                        |
| `remember-todo`                  | 48  | Add to to-do list                           |
| `generative`                     | 49  | Fallback to generative response             |


## 🔧 Setup Instructions

---

### 1. Required Files for Classifier

These are the essential files needed to **train** and **run** the intent classification model and FAISS model:

You must manually place the required model files under:
~kalliope/Chatbot/models/intent-medium-model-multilang-v2.0/
These files are large and are excluded via .gitignore.


---

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/ESDA-LAB/Alexandra-AI-chatbot.git
cd Alexandra-AI-chatbot/Chatbot

python3 -m venv kalliope_venv
source kalliope_venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

🎓 Training the Classifier

1. Prepare your training data:
```bash
{"instruction": "Classify the intent of the following sentence.", "input": "turn on the kitchen light please", "output": "kitchen-on"}
```
Examples for Alexandra Voice Assistant are in QuestionChatBot/dataset/.

Run the training script:
```bash
python3 QuestionChatBot/training.py
```
  
3. After training completes, your model will be saved in:
```bash
models/intent-medium-model-multilang-v2.0/
```
Modifying or Extending the Dataset - add new examples or new folder in Datasets/name_of_json_file.jsonl
 
```bash
{"instruction": "Classify the intent of the following sentence.", "input": "I am done reading a book", "output": "reading-off"}
```

TalkChatBot: Hybrid Classifier with Generative Fallback

TalkChatBot combines a trained intent classifier (BERT model) with semantic search using FAISS and optional generative fallback using GPT-2. This hybrid architecture ensures both precision and flexibility in understanding voice commands or user queries.
🏋️‍♂️ Training the Intent Classifier

The intent classifier is a fine-tuned transformer (like BERT) that maps input queries to predefined command labels (kitchen-on, lab-off, etc.).

📁 Dataset Format

Training data is stored in JSONL files like:
```bash
{
  "instruction": "Classify the intent.",
  "input": "Turn on the lights in the kitchen",
  "output": "kitchen-on"
}
```
Multiple examples should be added in:
```bash
QuestionChatBot/dataset/*.jsonl
```

You can group intents into files such as lights.jsonl, ambient.jsonl, radio.jsonl, etc.
🚀 Train the Classifier

Run the following script:
```bash
python3 QuestionChatBot/training.py
```
The trained model will be saved to:
```bash
models/intent-medium-model-multilang-v2.0/
```

###  Building the FAISS Index (Semantic Fallback)

FAISS is used when the classifier isn't confident (confidence < 0.9). It retrieves the most similar question from your dataset.

Dataset for Retrieval
Your retrieval dataset should also be in JSONL format, e.g.:
```bash
{
  "instruction": "Classify the intent.",
  "input": "Turn on the lights in the kitchen",
  "output": "kitchen-on"
}
```

Stored in
```bash
TalkChatBot/dataset/train_dataset/esdalab_english.jsonl
```
Build the FAISS index:

Use this script (or create build_faiss.py):
```bash
from sentence_transformers import SentenceTransformer
import faiss, json, numpy as np

with open("TalkChatBot/dataset/train_dataset/esdalab_english.jsonl") as f:
    entries = [json.loads(l) for l in f]
    questions = [e["input"] for e in entries]

embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = embedder.encode(questions)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

faiss.write_index(index, "faiss/esdalab_index.bin")
with open("faiss/esdalab_texts.json", "w") as f:
    json.dump(questions, f, indent=2)
```
---
### Classifier Architecture and Integration with Kalliope

The system is built around a hybrid intent classification pipeline that combines a fine-tuned transformer-based classifier, semantic retrieval using FAISS, and an optional generative fallback. This design ensures robust handling of both known and previously unseen user inputs in natural language.

## 1. Classification Workflow

## Step 1: Speech-to-Text (STT)

Kalliope captures the user's voice through a microphone and converts it into text using a speech recognition engine. The transcribed text is passed to the classifier module for intent detection.
Step 2: Intent Classification

    The transcribed text is processed using a fine-tuned transformer model (e.g., BERT).

    The model outputs a label (e.g., "kitchen-on") and a confidence score.

    If the confidence score is ≥ 0.9, the predicted label is accepted and returned.

    If the confidence score is < 0.9, semantic fallback is triggered.

## Step 2: Semantic Fallback with FAISS

    FAISS retrieves the most semantically similar query from a prepared training dataset.

    If a suitable match is found, its associated label is returned.

    If no appropriate label is identified, a fallback mechanism returns a dynamic message or triggers a generative model response.

    3. Kalliope Integration
    
## Synapse for Direct Intent Execution

Each classified label must correspond to a synapse in the Kalliope brain.yml:
```bash
- name: "kitchen-on"
  signals:
    - order: "kitchen-on"
  neurons:
    - say:
        message: "Turning on the kitchen lights."
    - shell:
        cmd: "python3 scripts/kitchen_on.py"
```
## Synapse for Generative Fallback

If no high-confidence intent is predicted, the classifier returns a dynamic fallback order:

```bash
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
The script dynamic_message_script.py is responsible for retrieving or generating the fallback message and passing it to the say neuron.

### 3. Processing Logic in Python
The core script that handles classification and fallback logic is structured as follows:

```bash
from classifier import classify_text, faiss_search

def process_audio_input(text):
    predicted_label, confidence = classify_text(text)

    if confidence >= 0.9:
        return predicted_label
    else:
        response = faiss_search(text)
        return {"order": "say-dynamic-message", "text": response}
```
This function is called by Kalliope's shell neuron or via a wrapper script connected to the voice recognition pipeline.

---
### Summary of Responsibilities
| Module                      | Description                                                    |
| --------------------------- | -------------------------------------------------------------- |
| `training.py`               | Trains the intent classifier on JSONL-formatted datasets       |
| `classifier.py`             | Loads the model and returns predictions with confidence scores |
| `faiss_search.py`           | Performs semantic similarity search using SentenceTransformer  |
| `process_audio_input.py`    | Manages classification logic and fallback decisions            |
| `brain.yml`                 | Maps output labels to executable Kalliope synapses             |
| `dynamic_message_script.py` | Returns fallback message for use in natural language response  |

---

### How to Add a New Intent Category

To extend the system with a new intent:

1. Update label2id dictionary
    Open your classifier configuration or label2id map (typically in config/ or classifier.py) and add your new label. For example

```bash
    label2id = {
    ...
    "new-intent-name": 50
}
```
    
2. Add training examples
    Add new entries in your dataset (e.g., train.jsonl) using the format:

```bash
{
  "instruction": "Classify the intent of the following sentence.",
  "input": "Turn on the garden lights",
  "output": "new-intent-name"
}
```

3. Retrain the model
    After editing the dataset, retrain your intent classifier:
   
```bash
    python train.py
```

4. Update Kalliope synapse
    Create a new synapse in your Kalliope brain config:

```bash
- name: garden-on
  signals:
    - order: "garden-on"
  neurons:
    - say:
        message: "Turning on the garden lights."
```

🧊 Installing Snowboy on Python 3.8+
    Snowboy is no longer officially maintained and does not support Python 3.8+ out of the box, but you can install a patched version like this:

✅ Steps to install Snowboy on Python 3.8+:

 Uninstall any broken version first:
 
```bash
pip uninstall snowboy
```

Clone the patched repo manually:

```bash
git clone https://github.com/Kitt-AI/snowboy.git
cd snowboy
```

Build the Python bindings:

```bash
cd swig/Python
python3 setup.py build
python3 setup.py install
```

Verify installation:

    python3
    >>> import snowboydecoder
    >>> print("Snowboy works!")

Alternatively, if using a precompiled wheel:

pip install https://github.com/johnbriant/snowboy-whl/raw/main/snowboy-1.3.0-py3-none-any.whl

⚠️ Note: If you're using Python 3.11 or newer, Snowboy likely won’t work. Consider switching to Porcupine as an alternative wake word engine.

## 👤 Author

**Dragomir Bozoki**  
📧 bozokidragomir@gmail.com  

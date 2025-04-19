# Import necessary modules from HuggingFace Transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
import subprocess
from Chatgpt_testground.QuestionChatBot.config.labels import label2id, id2label, reminder_keywords

# Mapping of intent labels to numerical IDs
# label2id

# Reverse mapping from ID to label
id2label = {v: k for k, v in label2id.items()}

# ----------- LOAD TRAINED MODEL AND TOKENIZER -----------

# Path to the trained model
model_path = "./intent-medium-model-multilang-v2.0"

# Load tokenizer and model from the saved path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Start Kalliope in background silently (output redirected to /dev/null)
subprocess.Popen(["kalliope", "start"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Instruction used as context during inference
instruction = "Classify the intent of the following sentence."

# Continuous input loop
while True:
    print("Write the command (or type 'exit' to quit):")
    text = input("> ").lower()

    if text.lower() == "exit":
        print("quiting the program...")
        break

    # Combine instruction with user input for classification
    input_text = f'{instruction} {text}'
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=32)

    # Run inference without gradient tracking
    with torch.no_grad():
        outputs = model(**inputs)

    # Compute softmax probabilities
    probabilities = torch.softmax(outputs.logits, dim=1)
    predicted_class_id = torch.argmax(probabilities, dim=1).item()
    predicted_label = id2label[predicted_class_id]
    
    # Extra fallback check for reminder-style sentences
   

    if any(kw in text for kw in reminder_keywords) and predicted_label not in ["remember-synapse", "remember-todo"]:
        predicted_label = "remember-synapse"

    # Output prediction and probability
    print(f"Input: '{text}'")
    print(f"→ Predicted instruction: '{predicted_label}' (Probability: {probabilities.max().item():.4f})")
    print("-" * 60)
    
    if probabilities.max().item() < 0.5:

        print("Not confident. Asking user for clarification...")
        print('-'*60)
        print()

    else:
        # Run corresponding Kalliope synapse
        print(f'kalliope will now run this neuron: {predicted_label}')
        print("-" * 60)
    
        # Run the synapse using Kalliope CLI
        subprocess.run(["kalliope", "start", "--run-synapse", predicted_label])
from transformers import TrainerCallback
import random
import os
import torch

class SaveEveryNEpochsCallback(TrainerCallback):
    def __init__(self, save_every_n_epochs, output_dir):
        self.save_every_n_epochs = save_every_n_epochs
        self.output_dir = output_dir

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch is not None and int(state.epoch + 1) % self.save_every_n_epochs == 0:
            output_path = os.path.join(self.output_dir, f"checkpoint-epoch-{int(state.epoch + 1)}")
            kwargs["model"].save_pretrained(output_path)
            kwargs["tokenizer"].save_pretrained(output_path)
            print(f"✅ Model saved at epoch {int(state.epoch + 1)} to {output_path}")

class PrintPredictionCallback(TrainerCallback):
    def __init__(self, tokenizer, model, dataset, print_every_n_epoch=5):
        self.tokenizer = tokenizer
        self.model = model
        self.dataset = dataset
        self.print_every_n_epochs = print_every_n_epoch

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch is None or int(state.epoch) % self.print_every_n_epochs != 0:
            return

        sample = random.choice(self.dataset)

        input_text = f"{sample['instruction']} {sample['input']}"
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            pred_id = torch.argmax(logits, dim=-1).item()

        pred_label = self.model.config.id2label[pred_id]
        true_label = sample['output']

        print("\n🔍 Sample Prediction:")
        print(f"Text      : {input_text}")
        print(f"True label: {true_label}")
        print(f"Predicted : {pred_label}")

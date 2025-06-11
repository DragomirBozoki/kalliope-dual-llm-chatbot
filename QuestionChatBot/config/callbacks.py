from transformers import TrainerCallback
import random
import os
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.model = model.to(device)  # Ensure model is on the correct device
        self.dataset = dataset
        self.print_every_n_epoch = print_every_n_epoch

    def on_epoch_end(self, args, state, control, **kwargs):
        if (state.epoch + 1) % self.print_every_n_epoch == 0:
            # Move inputs to the same device as the model
            inputs = {key: value.to(device) for key, value in self.dataset[0].items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            print("Sample Predictions:", outputs.logits.argmax(dim=-1).cpu().numpy())


from transformers import TrainerCallback
import os

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

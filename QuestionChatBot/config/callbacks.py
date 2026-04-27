import os
from typing import Any, Dict

import torch
from transformers import TrainerCallback


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SaveEveryNEpochsCallback(TrainerCallback):
    def __init__(self, save_every_n_epochs: int, output_dir: str) -> None:
        self.save_every_n_epochs = save_every_n_epochs
        self.output_dir = output_dir

    def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if state.epoch is None:
            return

        epoch = int(state.epoch + 1)

        if epoch % self.save_every_n_epochs != 0:
            return

        output_path = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch}")

        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer")

        if model is None:
            raise ValueError("Model is missing from callback kwargs.")

        model.save_pretrained(output_path)

        if tokenizer is not None:
            tokenizer.save_pretrained(output_path)

        print(f"Model saved at epoch {epoch} to {output_path}")


class PrintPredictionCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer: Any,
        model: torch.nn.Module,
        dataset: Any,
        print_every_n_epoch: int = 5,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.dataset = dataset
        self.print_every_n_epoch = print_every_n_epoch

    def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if state.epoch is None:
            return

        epoch = int(state.epoch + 1)

        if epoch % self.print_every_n_epoch != 0:
            return

        sample = self.dataset[0]
        inputs = self._move_inputs_to_device(sample)

        self.model.eval()

        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = outputs.logits.argmax(dim=-1).cpu().numpy()
        print("Sample predictions:", predictions)

    @staticmethod
    def _move_inputs_to_device(sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in sample.items()
        }

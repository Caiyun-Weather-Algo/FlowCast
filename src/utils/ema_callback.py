import torch
from torch_ema import ExponentialMovingAverage
import pytorch_lightning as pl
from pytorch_lightning.callbacks import StochasticWeightAveraging

class EMACallback(pl.Callback):
    def __init__(self, decay: float = 0.999):
        """
        :param decay: EMA decay rate, typically close to 1 (e.g., 0.999 or 0.9999).
        """
        self.decay = decay
        self.ema = None

    def on_train_start(self, trainer, pl_module):
        """Initialize EMA at the start of training."""
        if self.ema is None:
            self.ema = ExponentialMovingAverage(pl_module.parameters(), decay=self.decay)

    def on_after_backward(self, trainer, pl_module):
        """Update EMA after every backward pass."""
        self.ema.update()

    def on_validation_start(self, trainer, pl_module):
        """Before validation, apply EMA weights."""
        self.ema.store()  # Backup current parameters
        self.ema.copy_to()  # Apply EMA weights

    def on_validation_end(self, trainer, pl_module):
        """After validation, restore original weights."""
        self.ema.restore()  # Restore original weights after validation

    def on_test_start(self, trainer, pl_module):
        """Before validation, apply EMA weights."""
        self.ema.store()  # Backup current parameters
        self.ema.copy_to()  # Apply EMA weights

    def on_test_end(self, trainer, pl_module):
        """After validation, restore original weights."""
        self.ema.restore()  # Restore original weights after validation
        
    def on_train_end(self, trainer, pl_module):
        """At the end of training, store EMA for final evaluation."""
        self.ema.store()
        self.ema.copy_to()  # You can keep the EMA weights for final model evaluation.

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.ema is not None:
            return self.ema.state_dict()

    def on_load_checkpoint(self, callback_state):
        if self.ema is not None:
            self.ema.module.load_state_dict(callback_state["state_dict_ema"])
    
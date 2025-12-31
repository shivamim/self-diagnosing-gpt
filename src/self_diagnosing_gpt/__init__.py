"""Self-Diagnosing GPT with Mechanistic Interpretability"""

from .model import (
    SelfDiagnosingGPT,
    train_model,
    ActivationLogger,
    ablate_neurons,
)

__version__ = "0.1.0"
__all__ = [
    "SelfDiagnosingGPT",
    "train_model",
    "ActivationLogger",
    "ablate_neurons",
]

# Self-Diagnosing GPT with Mechanistic Interpretability

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A research-grade GPT implementation with self-diagnosis capabilities and mechanistic interpretability tools.

## Features

- **Self-Diagnosing Architecture**: Dual-head model with confidence estimation
- **Uncertainty Estimation**: Entropy-based confidence supervision
- **Mechanistic Interpretability**: Activation logging and visualization
- **Neuron Ablation**: Causal intervention framework

## Installation
```bash
git clone https://github.com/yourusername/self-diagnosing-gpt.git
cd self-diagnosing-gpt
pip install -e .
```

## Quick Start
```python
from self_diagnosing_gpt.model import SelfDiagnosingGPT, train_model

# Train model
model = train_model(max_iters=3000)

# Generate with confidence
output, confidence = model.generate(
    prompt="ROMEO:",
    max_tokens=200
)
```

## Usage Examples

See `examples/demo.py` for complete demonstrations:
- Text generation with confidence tracking
- Activation visualization
- Neuron ablation experiments

## Architecture
Input → Embeddings → Transformer Blocks → LayerNorm
├─→ LM Head → Next Token
└─→ Confidence Head → Uncertainty

## Documentation

- **Training**: Character-level GPT on Shakespeare dataset
- **Confidence**: Supervised by normalized entropy
- **Interpretability**: Forward hooks on transformer blocks
- **Ablation**: Context manager for neuron zeroing

## Citation
```bibtex
@software{self_diagnosing_gpt2024,
  author = {Shivam Shukla},
  title = {Self-Diagnosing GPT with Mechanistic Interpretability},
  year = {2026},
  url = {https://github.com/shivamim/self-diagnosing-gpt}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Based on Andrej Karpathy's GPT tutorial
- Inspired by Anthropic's interpretability research

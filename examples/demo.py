#!/usr/bin/env python
"""Demo script for Self-Diagnosing GPT"""

import torch
from self_diagnosing_gpt.model import (
    SelfDiagnosingGPT,
    train_model,
    ActivationLogger,
    ablate_neurons,
)

def main():
    print("="*80)
    print("Self-Diagnosing GPT Demo")
    print("="*80)
    
    # Download data first
    print("\n Download data:")
    print("wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
    
    # Train model
    print("\n Training model...")
    model = train_model(data_path='input.txt', max_iters=3000)
    
    # Demo 1: Generation
    print("\n" + "="*80)
    print("DEMO 1: Text Generation with Confidence")
    print("="*80)
    
    prompt = "ROMEO:"
    context = torch.tensor([model.encode(prompt)], dtype=torch.long, device='cuda' if torch.cuda.is_available() else 'cpu')
    generated, conf_log = model.generate(
        context, 
        max_new_tokens=200,
        temperature=0.8,
        decode_fn=model.decode,
        show_confidence=True
    )
    
    text = model.decode(generated[0].tolist())
    print(f"\nGenerated:\n{text}")
    
    # Demo 2: Interpretability
    print("\n" + "="*80)
    print("DEMO 2: Activation Visualization")
    print("="*80)
    
    logger = ActivationLogger(model)
    logger.register_hooks(layer_indices=[0, 3])
    
    model.eval()
    with torch.no_grad():
        model.generate(context, max_new_tokens=50, show_confidence=False)
    model.train()
    
    logger.visualize_activations('block_0')
    logger.remove_hooks()
    
    # Demo 3: Ablation
    print("\n" + "="*80)
    print("DEMO 3: Neuron Ablation")
    print("="*80)
    
    print("\nNormal:")
    normal, _ = model.generate(context, max_new_tokens=100, decode_fn=model.decode, show_confidence=False)
    print(model.decode(normal[0].tolist()))
    
    print("\nAblated:")
    with ablate_neurons(model, block_idx=2, neuron_indices=[10, 20, 30]):
        ablated, _ = model.generate(context, max_new_tokens=100, decode_fn=model.decode, show_confidence=False)
    print(model.decode(ablated[0].tolist()))
    
    print("\n Demo complete!")


if __name__ == "__main__":
    main()

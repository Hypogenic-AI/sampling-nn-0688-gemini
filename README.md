# Sampling Neural Network Research

This repository contains experiments investigating the effect of sampling from categorical distributions of intermediate activations in neural networks, similar to the final softmax layer.

## Key Findings Summary
- **Signal Attenuation**: Naive categorical sampling (softmax-based) at every layer causes signal to vanish, preventing training.
- **Differentiability**: Gumbel-Softmax "soft" sampling allows for convergence, whereas "hard" sampling (Straight-Through) is too noisy for deep architectures.
- **Selective Application**: Applying sampling only in final feature layers is viable, achieving **83.09% accuracy** on CIFAR-10 (compared to 86.46% for ReLU baseline).
- **Bottleneck Effect**: The competitive nature of categorical sampling acts as a capacity bottleneck, slightly reducing performance but providing a novel stochastic behavior.

## Repository Structure
- `src/`: Implementation of SSA and training scripts.
  - `ssa.py`: The Softmax Sampling Activation module.
  - `model.py`: VGG architecture with SSA integration.
  - `train.py`: Training and evaluation harness.
- `results/`: JSON logs of all experiments.
- `figures/`: Visualizations of accuracy and loss.
- `REPORT.md`: Detailed research report and analysis.

## How to Reproduce
1. Install dependencies:
   ```bash
   uv pip install torch torchvision datasets tqdm pandas numpy matplotlib scipy scikit-learn
   ```
2. Run the baseline:
   ```bash
   export PYTHONPATH=$PYTHONPATH:.
   python -m src.train --exp_name baseline --activation relu
   ```
3. Run the Mixed VGG (Selective SSA):
   ```bash
   python -m src.train_mixed --exp_name ssa_mixed --tau 1.0
   ```

For full details, see [REPORT.md](REPORT.md).

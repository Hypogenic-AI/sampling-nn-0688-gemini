# Research Plan: Sampling Neural Network

## Motivation & Novelty Assessment

### Why This Research Matters
Most modern neural networks rely on deterministic activation functions like ReLU. While effective, they lack inherent uncertainty quantification and can be prone to overfitting. Introducing stochasticity at intermediate layers can serve as a powerful regularizer and provide a natural way for the network to express uncertainty, similar to how the final softmax layer represents class probabilities.

### Gap in Existing Work
Existing stochastic methods often add noise (like ProbAct: $f(x) = \mu(x) + \epsilon$) or drop neurons (Dropout). The specific idea of treating an intermediate layer's activations as a probability distribution and *sampling* an activation according to that distribution (akin to the final softmax layer) is less commonly used as a primitive activation function. This approach forces the network to focus on the most "salient" features in a stochastic manner.

### Our Novel Contribution
We propose the "Softmax Sampling Activation" (SSA). Instead of simply applying a non-linearity, we treat the layer's activations as logits, apply a softmax to generate a probability distribution over the neurons, and sample one (or a few) neurons to pass forward. We will use the Gumbel-Softmax reparameterization trick to make this process differentiable.

### Experiment Justification
- **Experiment 1: Feasibility and Baseline Comparison**: Can a network with SSA layers even converge? We compare SSA against standard ReLU on CIFAR-10.
- **Experiment 2: Temperature Sensitivity**: The Gumbel-Softmax temperature $\tau$ controls the "hardness" of the sampling. We will investigate how $\tau$ affects convergence and final performance.
- **Experiment 3: Comparison with Other Stochasticity**: We will compare SSA with Dropout and ProbAct to see if categorical sampling provides unique benefits in terms of regularization or accuracy.

## Research Question
Does introducing categorical sampling from intermediate layer activations (Softmax Sampling Activation) improve generalization, robustness, or uncertainty calibration compared to deterministic activations and other stochastic methods?

## Hypothesis Decomposition
1. **Convergence**: Networks with SSA layers can be trained effectively using Gumbel-Softmax.
2. **Regularization**: SSA will provide a stronger regularization effect than ReLU, reducing the gap between training and validation accuracy.
3. **Efficiency**: SSA might lead to sparser activations, potentially improving computational efficiency if implemented with hard sampling.

## Proposed Methodology

### Approach
We will implement a custom PyTorch module for Softmax Sampling Activation.
$y = \text{Gumbel-Softmax}(x, \tau, \text{hard}=\text{False}) \cdot x$
We will integrate this into a standard VGG or ResNet architecture.

### Experimental Steps
1. **Baseline Implementation**: Train a VGG-16 model with ReLU on CIFAR-10.
2. **SSA Implementation**: Implement the SSA layer and integrate it into VGG-16.
3. **Training & Evaluation**: Train SSA-VGG with various temperatures.
4. **Analysis**: Evaluate Top-1 accuracy and calibration (ECE).

### Baselines
- **Standard ReLU**: The primary deterministic baseline.
- **Dropout (0.5)**: The standard stochastic regularization baseline.
- **ProbAct**: A representative noise-based stochastic activation.

### Evaluation Metrics
- **Top-1 Accuracy**: Standard performance metric.
- **Training vs. Validation Accuracy**: To measure regularization effect.
- **Expected Calibration Error (ECE)**: To measure uncertainty quantification.

## Expected Outcomes
- SSA might perform slightly worse than ReLU on training but better on test (regularization).
- Very low temperatures (harder sampling) might make training unstable.
- SSA might show better calibration than ReLU.

## Timeline
- Phase 1: Planning (Complete)
- Phase 2: Setup (10 min)
- Phase 3: Implementation (30 min)
- Phase 4: Experimentation (60 min)
- Phase 5: Analysis (20 min)
- Phase 6: Documentation (20 min)

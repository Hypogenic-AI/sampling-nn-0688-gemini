# Literature Review: Sampling Neural Network

## Research Area Overview
The research focuses on introducing stochasticity at intermediate layers of neural networks through sampling from the distribution of activations. This is a move away from deterministic activation functions (like ReLU, Sigmoid) towards probabilistic ones, aiming to improve generalization, provide uncertainty estimates, and potentially uncover new insights into neural representational associations.

## Key Papers

### Paper 1: ProbAct: A Probabilistic Activation Function for Deep Neural Networks (2019)
- **Authors**: Kumar Shridhar, Joonho Lee, Hideaki Hayashi, et al.
- **Key Contribution**: Proposes ProbAct, a stochastic activation function that samples from a normal distribution $\mathcal{N}(\mu(x), \sigma^2)$.
- **Methodology**: $\mu(x)$ is typically initialized as ReLU. $\sigma$ can be fixed or trained via backpropagation.
- **Datasets Used**: CIFAR-10, CIFAR-100, STL-10, IMDb.
- **Results**: Accuracy improvements of 2-3% on image tasks; provides inherent ensemble properties and uncertainty estimates.
- **Code Available**: [GitHub](https://github.com/kumar-shridhar/ProbAct-Probabilistic-Activation-Function)
- **Relevance**: Directly implements the hypothesis of sampling from intermediate activations.

### Paper 2: Flex-Act: Why Learn when you can Pick? (2026)
- **Authors**: Ramnath Kumar, Kyle Ritscher, et al.
- **Key Contribution**: Introduces a framework for discrete activation selection using the Gumbel-Softmax trick.
- **Methodology**: Layers choose from a basis set of activations (ReLU, Tanh, etc.) during training.
- **Relevance**: Another form of sampling/selection at the intermediate layer level, focused on functional diversity.

### Paper 3: Noisy Activation Functions (2016)
- **Authors**: Caglar Gulcehre, Marcin Moczulski, Misha Denil, Yoshua Bengio.
- **Key Contribution**: Injects noise into saturated regimes of activation functions.
- **Relevance**: Shows how stochasticity can help explore the boundary between degenerate and well-behaved parts of the activation function.

### Paper 4: Randomized Automatic Differentiation (2020)
- **Authors**: Deniz Oktay, Nick McGreivy, et al.
- **Relevance**: Provides a framework for unbiased stochastic gradients through randomized path sampling, relevant for training stochastic networks efficiently.

## Common Methodologies
- **Reparameterization Trick**: Used to make stochastic sampling differentiable (e.g., in VAEs and ProbAct).
- **Gumbel-Softmax**: Used for differentiable discrete selection (FlexAct).
- **Trainable Variance**: Learning the amount of noise needed per layer or neuron.

## Standard Baselines
- **Deterministic ReLU**: The standard against which stochastic activations are compared.
- **Dropout**: A related stochastic regularization technique.
- **Bayesian Neural Networks**: A more complex approach to uncertainty quantification.

## Evaluation Metrics
- **Top-1 Accuracy**: Standard classification performance.
- **Expected Calibration Error (ECE)**: Measures how well uncertainty estimates match actual error rates.
- **Robustness to Noisy Inputs**: Measuring performance under perturbations.

## Recommendations for Our Experiment
1. **Primary Dataset**: CIFAR-10 (standard, well-understood).
2. **Baseline Methods**: Standard ReLU-based CNN/ResNet.
3. **Primary Method to Implement**: A custom implementation of ProbAct or a variation that samples from a softmax distribution over neurons (as suggested by the "similar to sampling from the final softmax layer" part of the hypothesis).
4. **Evaluation**: Focus on both accuracy improvements and uncertainty calibration.

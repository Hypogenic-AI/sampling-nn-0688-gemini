# Resources Catalog

### Summary
This document catalogs all resources gathered for the Sampling Neural Network research project.

### Papers
Total papers downloaded: 6

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| ProbAct: A Probabilistic Activation Function | Kumar Shridhar et al. | 2019 | papers/1905.10761... | Stochastic activation function $f(x) = \mu(x) + \sigma \epsilon$. |
| Flex-Act: Why Learn when you can Pick? | Ramnath Kumar et al. | 2026 | papers/2601.06441... | Discrete activation selection via Gumbel-Softmax. |
| Noisy Activation Functions | Caglar Gulcehre et al. | 2016 | papers/1603.00391... | Noise in saturated regimes. |
| Randomized Automatic Differentiation | Deniz Oktay et al. | 2020 | papers/2007.10412... | Unbiased stochastic gradients. |
| Weight Uncertainty in Neural Networks | Charles Blundell et al. | 2015 | papers/1505.05424... | Bayes by Backprop. |
| Auto-Encoding Variational Bayes | Kingma & Welling | 2013 | papers/1312.6114... | Reparameterization trick. |

See `papers/README.md` for detailed descriptions.

### Datasets
Total datasets downloaded: 1

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| CIFAR-10 | HuggingFace (uoft-cs/cifar10) | 60K samples | Classification | datasets/cifar10/ | Standard image classification benchmark. |

See `datasets/README.md` for detailed descriptions.

### Code Repositories
Total repositories cloned: 1

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| ProbAct | github.com/kumar-shridhar/ProbAct... | Official implementation | code/ProbAct/ | Built with PyTorch. |

See `code/README.md` for detailed descriptions.

### Recommendations for Experiment Design

1. **Primary Dataset**: CIFAR-10 is the best starting point as it was used in the key papers.
2. **Baseline Methods**: Standard CNN or ResNet with ReLU activations.
3. **Evaluation Metrics**: Top-1 Accuracy and uncertainty metrics (e.g., ECE).
4. **Code to adapt/reuse**: The `ProbAct` implementation in `code/ProbAct/models/probact.py` can be directly adapted.

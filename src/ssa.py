import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxSamplingActivation(nn.Module):
    """
    Softmax Sampling Activation (SSA).
    Treats activations as logits, computes a softmax distribution over them,
    and samples using the Gumbel-Softmax trick.
    Includes scaling by number of categories to maintain signal magnitude.
    """
    def __init__(self, tau=1.0, hard=False, dim=1, scale=True):
        super(SoftmaxSamplingActivation, self).__init__()
        self.tau = tau
        self.hard = hard
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        num_categories = x.shape[self.dim]
        if self.training:
            weights = F.gumbel_softmax(x, tau=self.tau, hard=self.hard, dim=self.dim)
            y = weights * x
            return y
        else:
            if self.hard:
                probs = F.softmax(x, dim=self.dim)
                shape = probs.shape
                flat_probs = probs.view(-1, shape[self.dim])
                # Use argmax for deterministic inference if desired, or multinomial for true sampling
                sampled_idx = torch.argmax(flat_probs, dim=1).view(*[shape[i] if i != self.dim else 1 for i in range(len(shape))])
                mask = torch.zeros_like(probs).scatter_(self.dim, sampled_idx, 1.0)
                y = mask * x
                return y
            else:
                y = F.softmax(x, dim=self.dim) * x
                return y

def ssa_activation(x, tau=1.0, hard=False, dim=1, training=True):
    if training:
        weights = F.gumbel_softmax(x, tau=tau, hard=hard, dim=dim)
        return weights * x
    else:
        if hard:
            probs = F.softmax(x, dim=dim)
            # Simplified hard sampling for inference
            sampled_idx = torch.argmax(probs, dim=dim, keepdim=True)
            mask = torch.zeros_like(probs).scatter_(dim, sampled_idx, 1.0)
            return mask * x
        else:
            return F.softmax(x, dim=dim) * x

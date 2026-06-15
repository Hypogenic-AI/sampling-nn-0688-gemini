import torch
import torch.nn.functional as F

def test_ssa_scale(channels=64, shape=(1, 64, 32, 32)):
    x = torch.randn(shape)
    print(f"Original mean: {x.abs().mean().item():.4f}")
    
    # Softmax
    weights = F.softmax(x, dim=1)
    y = weights * x
    print(f"SSA (unscaled) mean: {y.abs().mean().item():.4f}")
    
    y_scaled = channels * weights * x
    print(f"SSA (scaled by {channels}) mean: {y_scaled.abs().mean().item():.4f}")

if __name__ == "__main__":
    test_ssa_scale()

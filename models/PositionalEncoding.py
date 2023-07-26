import torch

class PositionalEncoding(torch.nn.Module):
    def __init__(self, L: int, passIdentity=False):
        super().__init__()
        self.L = L
        sin_base = [lambda x: torch.sin(2**i*torch.pi*x) for i in range(L)]
        cos_base = [lambda x: torch.cos(2**i*torch.pi*x) for i in range(L)]
        self.func_base = sin_base + cos_base
        if passIdentity:
            self.func_base.append(lambda x: x)
        self.out_features = len(self.func_base) * 3
    
    def forward(self, x):
        return torch.cat([f(x) for f in self.func_base], dim=-1).flatten()

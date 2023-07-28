import torch

def sin_base_gen(n):
    return lambda y: torch.sin(2**n*torch.pi*y)

def cos_base_gen(n):
    return lambda y: torch.cos(2**n*torch.pi*y)

class PositionalEncoding(torch.nn.Module):
    def __init__(self, L: int, passIdentity=False):
        super().__init__()
        self.L = L
        print(L)
        sin_base = [sin_base_gen(i) for i in range(L)]
        cos_base = [cos_base_gen(i) for i in range(L)]
        self.func_base = sin_base + cos_base
        
        
        if passIdentity:
            self.func_base.append(lambda y: y)

        self.out_features = len(self.func_base)
    
    def forward(self, x):
        return torch.cat([f(x) for f in self.func_base], dim=-1)

if __name__ == "__main__":
    print("Testing PositionalEncoding")
    import numpy as np
    import matplotlib.pyplot as plt
    L = 6
    posEnc = PositionalEncoding(L, True)
    x = np.linspace(0, 1, 1000)
    x = torch.tensor(x[..., None])
    y = posEnc(x)
    print(y.shape)

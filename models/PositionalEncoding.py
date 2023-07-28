import torch

class PositionalEncoding(torch.nn.Module):
    def __init__(self, L: int, passIdentity=False):
        super().__init__()
        self.L = L
        sin_base = [(lambda x, i=i: torch.sin(2**i*torch.pi*x)) for i in range(L)]
        cos_base = [(lambda x, i=i: torch.cos(2**i*torch.pi*x)) for i in range(L)]
        self.func_base = sin_base + cos_base
        if passIdentity:
            self.func_base.append(lambda x: x)
        self.out_features = len(self.func_base) * 3
    
    def forward(self, x):
        return [f(x) for f in self.func_base]
        return torch.cat([f(x) for f in self.func_base], dim=-1)

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(-1, 1, 1000)
    x = torch.tensor(x, dtype=torch.float32)
    x = x[None, :]
    print(x.shape)
    pe = PositionalEncoding(6)
    y = pe(x)
    for i in range(len(y)):
        plt.plot(x[0], y[i][0].detach().numpy())
        plt.show()
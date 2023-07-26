import torch
from positional_encoding import PositionalEncoding

class TinyNERF(torch.nn.Module):
    def __init__(self, L, midle_dim=256):
        super().__init__()
        self.positional_encoding = PositionalEncoding(L, passIdentity=True)
        input_dim = self.positional_encoding.out_features
        self.fc1 = torch.nn.Linear(input_dim, midle_dim)
        self.fc2 = torch.nn.ModuleList([torch.nn.Linear(midle_dim, midle_dim) for i in range(3)])
        self.residual = torch.nn.Linear(midle_dim+input_dim, midle_dim)
        self.fc3 = torch.nn.ModuleList([torch.nn.Linear(midle_dim, midle_dim) for i in range(2)])
        self.out = torch.nn.Linear(midle_dim, 4)
        print(len(self._modules))
    
    def forward(self, x):
        x = self.positional_encoding(x)
        y = torch.nn.functional.relu(self.fc1(x))
        for i in range(len(self.fc2)):
            y = torch.nn.functional.relu(self.fc2[i](y))
        y = torch.cat([y, x], dim=-1)
        y = torch.nn.functional.relu(self.residual(y))
        for i in range(len(self.fc3)):
            y = torch.nn.functional.relu(self.fc3[i](y))
        y = self.out(y)
        rgb = torch.sigmoid(y[..., :3])
        sigma = torch.nn.functional.relu(y[..., 3])
        return torch.cat([rgb, sigma], dim=-1)



if __name__ == '__main__':
    model = TinyNERF(3, 256)
    print(model)
    for parameter in model.parameters():
        print(parameter.shape)
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name())
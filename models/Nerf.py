import torch
from models.PositionalEncoding import PositionalEncoding

class Nerf(torch.nn.Module):
    def __init__(self, Lxyz, Ldir, midle_dim=256):
        super().__init__()
        self.xyz_encoding = PositionalEncoding(Lxyz, passIdentity=True)
        self.dir_encoding = PositionalEncoding(Ldir, passIdentity=True)
        xyz_dim = self.xyz_encoding.out_features
        dir_dim = self.dir_encoding.out_features
        self.fc1 = torch.nn.Linear(xyz_dim, midle_dim)
        self.fc2 = torch.nn.ModuleList([torch.nn.Linear(midle_dim, midle_dim) for i in range(3)])
        self.residual = torch.nn.Linear(midle_dim+xyz_dim, midle_dim)
        self.fc3 = torch.nn.ModuleList([torch.nn.Linear(midle_dim, midle_dim) for i in range(2)])
        self.out1 = torch.nn.Linear(midle_dim, midle_dim + 1)
        self.fc4 = torch.nn.Linear(midle_dim + dir_dim, midle_dim//2)
        self.out2 = torch.nn.Linear(midle_dim//2, 3)
        print(len(self._modules))
    
    def forward(self, x):
        xyz = self.positional_encoding(x[..., :3])
        dir = self.positional_encoding(x[..., 3:])
        y = torch.nn.functional.relu(self.fc1(xyz))
        for i in range(len(self.fc2)):
            y = torch.nn.functional.relu(self.fc2[i](y))
        y = torch.cat([y, xyz], dim=-1)
        y = torch.nn.functional.relu(self.residual(y))
        for i in range(len(self.fc3)):
            y = torch.nn.functional.relu(self.fc3[i](y))
        
        y = torch.nn.functional.relu(self.out1(y))
        sigma = y[..., 0]

        y = torch.cat([y[...,1:], dir], dim=-1)
        y = torch.nn.functional.relu(self.fc4(y))
        rgb = torch.nn.functional.sigmoid(self.out2(y))

        return rgb, sigma

if __name__ == '__main__':
    model = Nerf(3, 2, 256)
    print(model)
    for parameter in model.parameters():
        print(parameter.shape)
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name())
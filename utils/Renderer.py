import torch

class Renderer():
    def __init__(self, config):
        if 'Nc' in config:
            self.Nc = config['Nc']
        else:
            self.Nc = 64

        if 'Nf' in config:
            self.Nf = config['Nf']
        else:
            self.Nf = 128
        
        if 'min_depth' in config:
            self.near = config['min_depth']
        else:
            self.near = 2

        if 'max_depth' in config:
            self.far = config['max_depth']
        else:
            self.far = 6
        
        if 'stratified' in config:
            self.stratified = config['stratified']
        else :
            self.stratified = True

    def getSparsePoints(self, ray_origins, ray_dirs):

        z_samples = torch.linspace(self.near, self.far, self.Nc + 1)[:-1].expand(ray_origins.shape[:-1]+ (self.Nc,))

        if(self.stratified):
            z_samples = z_samples + torch.rand(ray_origins.shape[:-1]+ (self.Nc,))*(self.far-self.near)/self.Nc
        
        points = z_samples[...,None]*ray_dirs[...,None,:] + ray_origins[...,None,:]

        return points

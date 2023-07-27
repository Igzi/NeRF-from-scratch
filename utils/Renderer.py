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
        z_samples = torch.linspace(self.near, self.far, self.Nc + 1)[:-1].expand(ray_origins.shape[:-1] + (self.Nc,))

        if(self.stratified):
            z_samples = z_samples + torch.rand(ray_origins.shape[:-1]+ (self.Nc,))*(self.far-self.near)/self.Nc
        
        points = z_samples[...,None]*ray_dirs[...,None,:] + ray_origins[...,None,:]
        dists = z_samples
        
        r = torch.norm(ray_dirs, dim = -1)
        dirs = torch.stack([torch.acos(ray_dirs[...,2]/r), torch.atan2(ray_dirs[...,1],ray_dirs[...,0])], dim = -1)
        dirs = dirs[..., None, :].expand((ray_origins.shape[:-1] + (self.Nc, 2)))

        points = torch.cat([points, dirs], dim = -1)

        return points, dists

    def getPixelValues(self, model, points, dists, chunk = 1024):
        assert points.dim() == 3 and points.shape[-1]==5 and points.shape[:2] == dists.shape[:2]

        rgb = torch.zeros(points.shape[:-1]+(3,))
        sigma = torch.zeros(points.shape[:-1])
        
        for i in range(0, points.shape[0], chunk):
            rgb[i:i+chunk,...], sigma[i:i+chunk,...] = model(points[i:i+chunk,...])

        inf_distance = 1e10
        delta = torch.cat([dists[...,1:]-dists[...,:-1], inf_distance*torch.ones(dists.shape[:-1] + (1,))], dim=-1)

        alpha = 1 - torch.exp(-sigma*delta)
        T = torch.exp(-torch.cumsum(sigma*delta, dim=-1))

        # Shift T by one and set T_0 to 1 for every point
        T = T.roll(1, dims=-1)
        T[:,0] = 1

        weights = alpha*T

        pixel_rgb = torch.sum(weights[...,None]*rgb, dim = -2)
        pixel_sigma = torch.sum(weights*sigma, dim = -1)

        return pixel_rgb, pixel_sigma



        


        


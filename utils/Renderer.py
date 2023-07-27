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
        assert ray_dirs.device == ray_origins.device and ray_dirs.shape == ray_origins.shape
        device = ray_dirs.device

        z_samples = torch.linspace(self.near, self.far, self.Nc + 1, device = device)[:-1].expand(ray_origins.shape[:-1] + (self.Nc,))

        if(self.stratified):
            z_samples = z_samples + torch.rand(ray_origins.shape[:-1]+ (self.Nc,), device = device)*(self.far-self.near)/self.Nc
        
        points = z_samples[...,None]*ray_dirs[...,None,:] + ray_origins[...,None,:]
        dists = z_samples * torch.norm(ray_dirs[...,None,:], dim = -1)
        
        r = torch.norm(ray_dirs, dim = -1)
        dirs = torch.stack([torch.acos(ray_dirs[...,2]/r), torch.atan2(ray_dirs[...,1], ray_dirs[...,0])], dim = -1)
        dirs = dirs[..., None, :].expand((ray_origins.shape[:-1] + (self.Nc, 2)))

        points = torch.cat([points, dirs], dim = -1)

        return points, dists

    def getPixelValues(self, model, points, dists, chunk = 1024):
        if points.dim() == 4:
            points = points.reshape((-1,)+points.shape[2:])
            dists = dists.reshape((-1,)+dists.shape[2:])

        assert model.device == points.device and points.device == dists.device
        assert points.dim() == 3 and points.shape[-1]==5 and points.shape[:2] == dists.shape[:2]

        device = points.device

        rgb = torch.zeros(points.shape[:-1]+(3,), device = device)
        sigma = torch.zeros(points.shape[:-1], device = device)
        
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

        return pixel_rgb



        


        


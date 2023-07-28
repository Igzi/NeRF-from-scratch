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

    def getPointsFromDepth(self, ray_origins, ray_dirs, z_samples):
        points = z_samples[...,None]*ray_dirs[...,None,:] + ray_origins[...,None,:]
        dists = z_samples * torch.norm(ray_dirs[...,None,:], dim = -1)
        
        r = torch.norm(ray_dirs, dim = -1)
        dirs = torch.stack([torch.acos(ray_dirs[...,2]/r), torch.atan2(ray_dirs[...,1], ray_dirs[...,0])], dim = -1)
        dirs = dirs[..., None, :].expand((ray_origins.shape[:-1] + (z_samples.shape[-1], 2)))

        points = torch.cat([points, dirs], dim = -1)

        return points, dists

    def getSparsePoints(self, ray_origins, ray_dirs, return_samples = False):
        assert ray_dirs.device == ray_origins.device and ray_dirs.shape == ray_origins.shape
        device = ray_dirs.device

        z_samples = torch.linspace(self.near, self.far, self.Nc + 1, device = device)[:-1]

        if(self.stratified):
            z_samples = z_samples + torch.rand(ray_origins.shape[:-1] + (self.Nc,), device = device)*(self.far-self.near)/self.Nc
        
        points, dists = self.getPointsFromDepth(ray_origins, ray_dirs, z_samples)
        if return_samples:
            return points, dists, z_samples
        else:
            return points, dists
    
    def getFinePoints(self, ray_origins, ray_dirs, sparse_samples, weights):
        assert weights.dim() == 2 and sparse_samples.dim() == 2
        assert ray_dirs.device == ray_origins.device and ray_dirs.shape == ray_origins.shape

        device = ray_dirs.device

        H, W = sparse_samples.shape[0], self.Nf # size of samples

        sparse_samples = torch.cat([sparse_samples, self.far*torch.ones((H, 1), device=device)], dim = -1)

        sample_idx = torch.multinomial(weights + 1e-8, num_samples = self.Nf, replacement=True)
        
        rows = (torch.arange(0, H)[:,None]) @ torch.ones((1, self.Nf)).long()
        rows = rows.to(device)
        
        samples = sparse_samples[rows, sample_idx]+(sparse_samples[rows, sample_idx+1]-sparse_samples[rows, sample_idx])*torch.rand((H, W),device = device)
        samples, _ = torch.sort(samples, dim = -1)

        ray_origins = ray_origins.reshape((-1,3))
        ray_dirs = ray_dirs.reshape((-1,3))
        
        return self.getPointsFromDepth(ray_origins, ray_dirs, samples)


    def getPixelValues(self, model, points, dists, return_weights = False):
        if points.dim() == 4:
            points = points.reshape((-1,)+points.shape[2:])
            dists = dists.reshape((-1,)+dists.shape[2:])

        assert points.device == dists.device
        assert points.dim() == 3 and points.shape[-1]==5 and points.shape[:2] == dists.shape[:2]
        
        device = points.device

        rgb = torch.zeros(points.shape[:-1]+(3,), device = device)
        sigma = torch.zeros(points.shape[:-1], device = device)
        
        rgb, sigma = model(points)
        
        inf_distance = 1e10
        delta = torch.cat([dists[...,1:]-dists[...,:-1], inf_distance*torch.ones(dists.shape[:-1] + (1,),device=device)], dim=-1)

        alpha = 1 - torch.exp(-sigma*delta)
        T = torch.cumprod(1 - alpha + 1e-8, dim=-1)

        # Shift T by one and set T_0 to 1 for every point
        T = T.roll(1, dims=-1)
        T[:,0] = 1

        weights = alpha*T

        pixel_rgb = torch.sum(weights[...,None]*rgb, dim = -2)
        
        if return_weights:
            return pixel_rgb, weights
        else:
            return pixel_rgb



        


        


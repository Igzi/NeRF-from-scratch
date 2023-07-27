import torch
import numpy as np
from models.TinyNerf import TinyNerf
from utils.Trainer import Trainer

class TrainerTinyNerf(Trainer):
    def __init__(self, model, device, images, cameras, renderer, config):
        super().__init__(model, device, images, cameras, renderer, config)
        self.Lxyz = config['L_xyz']

    def train(self):
        model = TinyNerf(self.Lxyz)
        model.to(self.device)

        optimizer = torch.optim.Adam(self.lr)
        criterion = torch.nn.MSELoss()

        for i in range(self.max_epochs):
            optimizer.zero_grad()

            rnd_img = np.random.randint(0, self.images.shape[0])
            img = self.images[rnd_img]
            ray_origins, ray_dirs = self.cameras[i].getRays()

            points, dists = self.renderer.getSparsePoints(ray_origins, ray_dirs)
            points = points.to(self.device)
            dists = dists.to(self.device)

            rgb = self.renderer.getPixelValues(model, points, dists)
            loss = criterion(rgb, img)

            loss.backward()
            optimizer.step()




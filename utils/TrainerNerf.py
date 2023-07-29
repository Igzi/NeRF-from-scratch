import time
from datetime import datetime
import torch
import numpy as np
from models.Nerf import Nerf
from utils.Trainer import Trainer
from utils.Camera import Camera
from utils.Visualizer import Visualizer
import matplotlib.pyplot as plt

class TrainerNerf(Trainer):
    def __init__(self, model, device, images, cameras, renderer, config):
        super().__init__(model, device, images, cameras, renderer, config)
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
    def train(self, test_img, test_pose, focal):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        model_sparse, model_fine = self.model
        
        ray_origins = torch.zeros(self.images.shape, device=self.device)
        ray_dirs = torch.zeros(self.images.shape, device=self.device)
        for i in range(len(self.cameras)):
            ray_origins[i], ray_dirs[i] = self.cameras[i].getRays()

        # ray_dirs, ray_origins, images
        all_samples = torch.stack([ray_dirs, ray_origins, self.images], dim=-1)
        all_samples = all_samples.reshape((-1, all_samples.shape[-2], all_samples.shape[-1]))
        all_samples = all_samples.to('cpu') # remove rays from gpu memory
        
        perm = torch.randperm(all_samples.shape[0])
        all_samples = all_samples[perm]
        del ray_dirs
        del ray_origins
        del self.images
        optimizer = torch.optim.Adam(list(model_sparse.parameters()) + list(model_fine.parameters()),lr=self.lr)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, 5e-6)
        criterion = torch.nn.MSELoss()

        test_camera = Camera(test_img.shape[1], test_img.shape[2], test_pose[0], focal)
        psnr_list = []
        visualizer = Visualizer(test_img=test_img, test_camera=test_camera, renderer=self.renderer, device=self.device, criterion=criterion)
        start = time.time()
        losses = torch.zeros(self.stats_step)

        batch_size = self.batch_size
        j = 0
        for i in range(self.max_epochs):
            if (j+1)*self.batch_size > len(all_samples):
                perm = torch.randperm(all_samples.shape[0])
                all_samples = all_samples[perm]
                j = 0

            batch_samples = all_samples[(j*self.batch_size):((j+1)*self.batch_size)]
            j += 1
            batch_samples = batch_samples.to(self.device)
            batch_ray_dirs = batch_samples[:,:,0]
            batch_ray_origins = batch_samples[:,:,1]
            batch_images = batch_samples[:,:,2]

            optimizer.zero_grad()
                        
            
            points, dists, sparse_samples = self.renderer.getSparsePoints(batch_ray_origins, batch_ray_dirs, return_samples=True)
            sparse_rgb, weights = self.renderer.getPixelValues(model_sparse, points, dists, return_weights=True)

            fine_points, fine_dists = self.renderer.getFinePoints(batch_ray_origins, batch_ray_dirs, sparse_samples, weights)
            rgb = self.renderer.getPixelValues(model_fine, fine_points, fine_dists)

            assert not torch.any(rgb > 1.1)
            assert not torch.any(rgb < -0.1)
            assert not torch.any(sparse_rgb > 1.1) 
            assert not torch.any(sparse_rgb < -0.1)
            assert not torch.any(batch_images > 1.1) 
            assert not torch.any(batch_images < -0.1)

            loss = criterion(rgb, batch_images) + criterion(sparse_rgb, batch_images)
            
            losses[i % self.stats_step] = loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % self.checkpoint_step == 0 and i > 0:
                now = datetime.now()
                dt_string = now.strftime("%d%m%Y%H%M%S")
                torch.save({
                    'epoch': i,
                    'model_fine_state_dict': model_fine.state_dict(),
                    'model_sparse_state_dict': model_sparse.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss_history': psnr_list,
                    }, self.checkpoint_path + f"{dt_string}.pt")

            if i % self.stats_step == 0:
                model_sparse.eval()
                model_fine.eval()
                loss_mean = losses.mean()
                print(f'Epoch: {i}, Average loss: {loss_mean}, Secs per iter: {(time.time()-start)/self.stats_step}')
                visualizer.visualize(i, time.time()-start, psnr_list, model_fine)
                start = time.time()
                model_sparse.train()
                model_fine.train()

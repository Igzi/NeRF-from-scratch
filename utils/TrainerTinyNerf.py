import time
from datetime import datetime
import torch
import numpy as np
from utils.Trainer import Trainer
from utils.Camera import Camera
from utils.Visualizer import Visualizer
import matplotlib.pyplot as plt

class TrainerTinyNerf(Trainer):
    def __init__(self, model, device, images, cameras, renderer, config):
        super().__init__(model, device, images, cameras, renderer, config)
    
    def train(self, test_img, test_pose, focal):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        self.model.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        criterion = torch.nn.MSELoss()

        test_camera = Camera(test_img.shape[1], test_img.shape[2], test_pose[0], focal)
        visualizer = Visualizer(test_img=test_img, test_camera=test_camera, renderer=self.renderer, device=self.device, criterion=criterion)
        psnr_list = []
        start = time.time()
        losses = torch.zeros(self.stats_step)

        for i in range(self.max_epochs):
            optimizer.zero_grad()

            rnd_img = np.random.randint(0, self.images.shape[0])
            img = self.images[rnd_img]
            
            ray_origins, ray_dirs = self.cameras[rnd_img].getRays()
            
            points, dists = self.renderer.getSparsePoints(ray_origins, ray_dirs)
            
            rgb = self.renderer.getPixelValues(self.model, points, dists)
            loss = criterion(rgb, img.reshape((-1,3)).to(rgb.device))
            losses[i % self.stats_step] = loss.item()
            loss.backward()
            optimizer.step()

            if i % self.checkpoint_step == 0 and i > 0:
                now = datetime.now()
                dt_string = now.strftime("%d%m%Y%H%M%S")
                torch.save({
                    'epoch': i,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss_history': psnr_list,
                    }, self.checkpoint_path + f"{dt_string}.pt")

            if i % self.stats_step == 0:
                loss_mean = losses.mean()
                print(f'Epoch: {i}, Average loss: {loss_mean}, Secs per iter: {(time.time()-start)/self.stats_step}')
                visualizer.visualize(i, time.time()-start, psnr_list, self.model)
                start = time.time()

import torch
import numpy as np
from models.TinyNerf import TinyNerf
from utils.Trainer import Trainer
from utils.Camera import Camera
import matplotlib.pyplot as plt

class TrainerTinyNerf(Trainer):
    def __init__(self, model, device, images, cameras, renderer, config):
        super().__init__(model, device, images, cameras, renderer, config)
        self.Lxyz = config['L_xyz']
    
    def train(self, test_img, test_pose, focal):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        model = TinyNerf(self.Lxyz)
        model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(),lr=self.lr, betas=(0.9, 0.999), eps=1e-7)
        criterion = torch.nn.MSELoss()

        test_camera = Camera(test_img.shape[1], test_img.shape[2], test_pose[0], focal)
        psnr_list = []
        for i in range(self.max_epochs):
            optimizer.zero_grad()

            rnd_img = np.random.randint(0, self.images.shape[0])
            img = self.images[rnd_img]
            ray_origins, ray_dirs = self.cameras[rnd_img].getRays()

            points, dists = self.renderer.getSparsePoints(ray_origins, ray_dirs)
            points = points.to(self.device)
            dists = dists.to(self.device)

            rgb = self.renderer.getPixelValues(model, points, dists)
            loss = criterion(rgb, img.reshape((-1,3)).to(rgb.device))

            loss.backward()
            optimizer.step()

            if i % 2 == 0:
                print(f'Epoch: {i}, Loss: {loss.item()}')
                test_o, test_d = test_camera.getRays()
                test_o = test_o.to(rgb.device)
                test_d= test_d.to(rgb.device)
                
                test_points, test_dists = self.renderer.getSparsePoints(test_o, test_d)
                with torch.no_grad():
                    test_rgb = self.renderer.getPixelValues(model, test_points, test_dists)
                    test_loss = criterion(test_rgb, test_img.reshape((-1,3)))
                    test_psnr = -10*torch.log10(test_loss)
                    psnr_list.append(test_psnr.item())
                
                print(f'Test PSNR: {test_psnr.item()}')
                print(test_rgb.shape)
                plt.subplot(1,2,1)
                plt.imshow(test_rgb.cpu().reshape((100,100,3)).numpy())
                plt.subplot(1,2,2)
                plt.plot(psnr_list)
                plt.show()

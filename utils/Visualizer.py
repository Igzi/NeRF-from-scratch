import torch
import matplotlib.pyplot as plt


class Visualizer():
    def __init__(self, test_img, test_camera, renderer, device, criterion):
        self.test_img = test_img
        self.test_camera = test_camera
        self.renderer = renderer
        self.device = device
        self.criterion = criterion

    def visualize(self, epoch, elapsed_time, psnr_list, model):        
        test_o, test_d = self.test_camera.getRays()
        test_o = test_o.to(self.device)
        test_d = test_d.to(self.device)
        
        test_points, test_dists = self.renderer.getSparsePoints(test_o, test_d)
        with torch.no_grad():
            test_rgb = self.renderer.getPixelValues(model, test_points, test_dists)
            test_loss = self.criterion(test_rgb, self.test_img.reshape((-1,3)).to(test_rgb.device))
            test_psnr = -10*torch.log10(test_loss)
            psnr_list.append(test_psnr.item())
        
        print(f'Test PSNR: {test_psnr.item()}')
        print(test_rgb.shape)
        plt.subplot(2,2,3)
        plt.imshow(test_rgb.cpu().reshape((100,100,3)).numpy())
        plt.subplot(2,2,4)
        plt.imshow(self.test_img[0])
        plt.subplot(2,1,1)
        plt.plot(psnr_list)
        plt.show()
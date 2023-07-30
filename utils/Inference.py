import torch
import matplotlib.pyplot as plt
import numpy as np


class Inference:
    def __init__(self, models, checkpoint_path, device, renderer, dir_path):
        self.model_sparse, self.model_fine = models
        self.load_model(checkpoint_path, models)
        self.renderer = renderer
        self.dir_path = dir_path

    def eval(self, camera):
        rays_origins, ray_dirs = camera.getRays()
g        rays_origins = rays_origins.reshape((-1,3))
        ray_dirs = ray_dirs.reshape((-1,3))
        chunk_size = 40000
        test_rgb = torch.zeros_like(rays_origins).reshape((-1,3))
        for i in range(len(test_rgb)//chunk_size):
            points, dists, sparse_samples = self.renderer.getSparsePoints(rays_origins[i*chunk_size:(i+1)*chunk_size,...], ray_dirs[i*chunk_size:(i+1)*chunk_size,...], return_samples=True)
            sparse_rgb, weights = self.renderer.getPixelValues(self.model_sparse, points, dists, return_weights=True)

            fine_points, fine_dists = self.renderer.getFinePoints(rays_origins[i*chunk_size:(i+1)*chunk_size,...], ray_dirs[i*chunk_size:(i+1)*chunk_size,...], sparse_samples, weights)
            test_rgb[i*chunk_size:(i+1)*chunk_size,...] = self.renderer.getPixelValues(self.model_fine, fine_points, fine_dists)
        return test_rgb

    def inference(self, test_cameras, test_images, plot=True, save=False):
        for i in range(len(test_cameras)):
            with torch.no_grad():
                eval_img = self.eval(test_cameras[i])
            if eval_img.device != 'cpu':
                eval_img = eval_img.cpu()
                eval_img = eval_img.reshape(800,800,3)
            if plot:
                gt_img = test_images[i]
                self.plot_eval_gt(eval_img, gt_img)
            if save:
                gt_img = test_images[i]
                self.save_eval_gt(eval_img, gt_img)

    def plot_eval_gt(self, eval_img, gt_img):
        plt.subplot(2, 1, 1)
        plt.imshow(eval_img)
        plt.title('Eval Image')
        plt.subplot(2, 1, 2)
        plt.imshow(gt_img)
        plt.title('Ground Truth Image')
        plt.show()

    def eval_and_save(self, camera, path):
        path = self.dir_path + path
        with torch.no_grad():
            img = self.eval(camera)
        if img.device != 'cpu':
            img = img.cpu()
            img = img.reshape(800,800,3)
        img = img*255
        img = img.long()
        img = img.numpy()
        img = img.astype(np.uint8)
        plt.imsave(path, img)

    def save_eval_gt(self, eval_img, gt_img):
        plt.subplot(2, 1, 1)
        plt.imshow(eval_img)
        plt.title('Eval Image')
        plt.subplot(2, 1, 2)
        plt.imshow(gt_img)
        plt.title('Ground Truth Image')
        plt.savefig('books_read.png')

    def load_model(self, checkpoint_path, models):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        for name in checkpoint:
            print(name)
            # print(checkpoint[name])
        model_sparse, model_fine = models
        model_sparse.load_state_dict(checkpoint['model_sparse_state_dict'])
        model_fine.load_state_dict(checkpoint['model_fine_state_dict'])
        model_sparse.eval()
        model_fine.eval()
        print(len(checkpoint['train_loss_history']))

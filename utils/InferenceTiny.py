import torch
import matplotlib.pyplot as plt


class Inference:
    def __init__(self, model, checkpoint_path, device, renderer):
        self.model = model
        self.load_model(checkpoint_path, model)
        self.model.to(device)
        self.renderer = renderer

    def eval(self, camera):
        rays_origins, ray_dirs = camera.getRays()
        print(rays_origins.shape, ray_dirs.shape)
        rays_origins = rays_origins.reshape((-1, 3))
        ray_dirs = ray_dirs.reshape((-1, 3))
        chunk_size = 40000
        test_rgb = torch.zeros_like(rays_origins).reshape((-1, 3))
        for i in range(len(test_rgb)//chunk_size):
            points, dists = self.renderer.getSparsePoints(rays_origins[i*chunk_size:(
                i+1)*chunk_size, ...], ray_dirs[i*chunk_size:(i+1)*chunk_size, ...])
            test_rgb[i*chunk_size:(i+1)*chunk_size, ...] = self.renderer.getPixelValues(
                self.model, points, dists)
        return test_rgb

    def inference(self, test_cameras, test_images, plot=True, save=False):
        for i in range(len(test_cameras)):
            with torch.no_grad():
                eval_img = self.eval(test_cameras[i])
            if eval_img.device != 'cpu':
                eval_img = eval_img.cpu()
                eval_img = eval_img.reshape(800, 800, 3)
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

    def save_eval_gt(self, eval_img, gt_img):
        plt.subplot(2, 1, 1)
        plt.imshow(eval_img)
        plt.title('Eval Image')
        plt.subplot(2, 1, 2)
        plt.imshow(gt_img)
        plt.title('Ground Truth Image')
        plt.savefig('books_read.png')

    def load_model(self, checkpoint_path, model):
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device('cpu'))
        for name in checkpoint:
            print(name)
            # print(checkpoint[name])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(len(checkpoint['train_loss_history']))

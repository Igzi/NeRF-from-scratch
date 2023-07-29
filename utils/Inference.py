import torch
import matplotlib.pyplot as plt


class Inference:
    def __init__(self, fn, model, checkpoint_path, device):
        pass

    def inference(self, test_cameras, test_images, plot=False, save=False):
        for i in range(len(test_cameras)):
            eval_img = self.fn(test_cameras[i])
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

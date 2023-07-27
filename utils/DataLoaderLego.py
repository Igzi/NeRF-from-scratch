import json
import torch
import numpy as np
import imageio.v3 as iio
from utils.DataLoader import DataLoader
import torchvision

class DataLoaderLego(DataLoader):
    def __init__(self, config):
        super().__init__(config)

    def getDataset(self, type):
        if type == 'train':
            dataset = 'train'
            test_size = self.train_size
        elif type == 'validation':
            dataset = 'val'
            test_size = self.validation_size
        elif type == 'test':
            dataset = 'test'
            test_size = self.test_size
        else:
            print("Dataset type ", type, " does not exist")
            return

        transform_path = self.dataset_path + "/transforms_" + dataset + ".json"
        transform_file = open(transform_path)
        transforms = json.load(transform_file)

        for i in range(test_size):
            frame = transforms['frames'][i]
            image_path = self.dataset_path + frame['file_path'] + ".png"
            image = iio.imread(image_path)

            if i==0:
                images = torch.zeros((test_size,) + image.shape)
                poses = torch.zeros((test_size,) + torch.tensor(frame['transform_matrix']).shape)

            poses[i] = torch.tensor(frame['transform_matrix'])
            images[i] = torch.tensor(image)

        if(image.shape[-1]==4):
            images = images[:,:,:,:3]

        H, W = images[0].shape[:2]
        focal = .5 * W / np.tan(.5 * transforms['camera_angle_x'])
        images = torch.nn.functional.interpolate(images.permute(
            0, 3, 1, 2), (100, 100)).permute(0, 2, 3, 1)        
        images /= 255
        H = H//8
        W = W//8
        focal = focal/8.

        return images, poses, focal

import json
import torch
import numpy as np
import imageio.v3 as iio
from utils.DataLoader import DataLoader

class DataLoaderReal(DataLoader):
    def __init__(self, config):
        super().__init__(config)

    def getDataset(self, type, downsample = False):
        transform_path = self.dataset_path + "/transforms.json"
        transform_file = open(transform_path)
        transforms = json.load(transform_file)

        fx, fy = transforms['fl_x'], transforms['fl_y']

        torch.manual_seed(0)
        perm = torch.randperm(self.train_size + self.validation_size + self.test_size)

        if type == 'train':
            data_start = 0 
            data_end = self.train_size
        elif type == 'validation':
            data_start = self.train_size 
            data_end = self.train_size + self.validation_size
        elif type == 'test':
            data_start = self.train_size + self.validation_size
            data_end = self.train_size + self.validation_size + self.data_size
        else:
            print("Dataset type ", type, " does not exist")
            return
        
        data_size = data_end - data_start

        for i in range(data_start, data_end):
            frame = transforms['frames'][perm[i]]
            image_path = self.dataset_path + frame['file_path']
            image = iio.imread(image_path)

            if i==data_start:
                images = torch.zeros((data_size,) + image.shape)
                poses = torch.zeros((data_size,) + torch.tensor(frame['transform_matrix']).shape)

            poses[i-data_start] = torch.tensor(frame['transform_matrix'])
            images[i-data_start] = torch.tensor(image)

        if(image.shape[-1]==4):
            images = images[:,:,:,:3]

        H, W = images[0].shape[:2]
        focal = .5 * W / np.tan(.5 * transforms['camera_angle_x'])
        if downsample:
            images = torch.nn.functional.interpolate(images.permute(
                0, 3, 1, 2), (100, 100)).permute(0, 2, 3, 1)
                    
            fx = fx * 100 / W
            fy = fy * 100 / H
            H = 100
            W = 100
        images /= 255

        focal = fx, fy

        return images, poses, focal

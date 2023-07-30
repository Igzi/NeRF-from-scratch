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
            test_start = 0 
            test_end = self.train_size
        elif type == 'validation':
            test_start = self.train_size 
            test_end = self.train_size + self.validation_size
        elif type == 'test':
            test_start = self.train_size + self.validation_size
            test_end = self.train_size + self.validation_size + self.test_size
        else:
            print("Dataset type ", type, " does not exist")
            return
        
        test_size = test_end - test_start
        images = torch.zeros((test_size,) + image.shape)
        poses = torch.zeros((test_size,) + torch.tensor(frame['transform_matrix']).shape)

        for i in perm[test_start:test_end]:
            frame = transforms['frames'][i]
            image_path = self.dataset_path + frame['file_path'] + ".png"
            image = iio.imread(image_path)

            poses[i] = torch.tensor(frame['transform_matrix'])
            images[i] = torch.tensor(image)

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

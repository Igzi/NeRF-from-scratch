import json
import torch
import imageio.v3 as iio
from utils.DataLoader import DataLoader

class DataLoaderReal(DataLoader):
    def __init__(self, config):
        super().__init__(config)

    def getDataset(self, type, downsample = False):
        transform_path = self.dataset_path + "/transforms.json"
        transform_file = open(transform_path)
        transforms = json.load(transform_file)

        focal = transforms['fl_x'], transforms['fl_y']

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

        return images, poses, focal

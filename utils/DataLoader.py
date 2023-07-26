import json
import torch
import imageio.v3 as iio

class DataLoader():
    def __init__(self, config):
        self.dataset_path = config['dataset_path']
        if 'train_size' in config:
            self.train_size = config['train_size']
        if 'validation_size' in config:
            self.validation_size = config['validation_size']
        if 'test_size' in config:
            self.test_size = config['test_size']

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

        transform_path = self.dataset_path + "/transforms_" + type + ".json"
        transform_file = open(transform_path)
        transforms = json.load(transform_file)

        for i in range(test_size):
            frame = transforms['frames'][0]
            image_path = self.dataset_path + frame['file_path'] + ".png"
            image = iio.imread(image_path)

            if i==0:
                images = torch.zeros((test_size,) + image.shape)
                poses = torch.zeros((test_size,) + torch.tensor(frame['transform_matrix']).shape)

            poses[i] = torch.tensor(frame['transform_matrix'])
            images[i] = torch.tensor(image)

        if(image.shape[-1]==4):
            images = images[:,:,:,:3]

        return images, poses




        

        

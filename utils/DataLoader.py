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
        
        images = torch.tensor([])
        poses = torch.tensor([])

        transform_path = self.dataset_path + "/transforms_" + type + ".json"
        transform_file = open(transform_path)
        transforms = json.load(transform_file)

        for i in range(1,test_size+1):
            frame = transforms['frames'][0]
            
            torch.cat([poses, torch.tensor(frame['transform_matrix'])], axis=0) 

            image_path = self.dataset_path + frame['file_path'] + ".png"
            image = iio.imread(image_path)
            torch.cat([images, torch.tensor(image)], axis=0) 




        

        

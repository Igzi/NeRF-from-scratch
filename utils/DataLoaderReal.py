import json
import torch
import numpy as np
import imageio.v3 as iio
from utils.DataLoader import DataLoader
import torchvision

class DataLoaderReal(DataLoader):
    def __init__(self, config):
        super().__init__(config)

    def getDataset(self, type, downsample = False):
        

        return images, poses, focal

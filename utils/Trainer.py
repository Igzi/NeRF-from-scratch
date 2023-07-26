import torch

class Trainer():
    def __init__(self, model, images, cameras, renderer, config):
        self.model = model
        self.images = images
        self.cameras = cameras
        self.renderer = renderer
        self.seed = config['seed']
        self.resume = config['resume']
        self.max_epochs = config['max_epochs']
        self.lr = config['lr']
        self.stats_step = config['stats_print_interval']
        self.checkpoint_step = config['checkpoint_epoch_interval']
        self.checkpoint_path = config['checkpoint_path']

    def train():
        optimizer = torch.optim.Adam(self.lr)

        for i in range(self.max_epochs):
            optimizer.zero_grad()

            
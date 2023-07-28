from utils.Renderer import Renderer
from utils.Camera import Camera
class Trainer():
    def __init__(self, model, device, images, cameras, renderer, config):
        self.model = model
        self.device = device
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

            
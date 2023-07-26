class DataLoader():
    def __init__(self, config):
        self.dataset_path = config['dataset_path']
        if 'train_size' in config:
            self.train_size = config['train_size']
        if 'validation_size' in config:
            self.validation_size = config['validation_size']
        if 'test_size' in config:
            self.test_size = config['test_size']

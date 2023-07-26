import torch

class TrainerTinyNerf():
    def __init__(self, model, images, cameras, renderer, config):
        super().__init__(model, images, cameras, renderer, config)
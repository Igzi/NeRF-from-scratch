import torch

class Camera():
    def __init__(self, H, W, pose, focal):
        self.H = H
        self.W = W
        self.pose = pose
        self.focal = focal

    def getRays(self):
        i = torch.ones((self.H,1)) @ (torch.arange(0,self.W)[None,:].float())
        j = (torch.arange(0,self.H)[:,None].float()) @ torch.ones((1,self.W))

        camera_dirs = torch.stack([(i-self.W/2)/self.focal, (j-self.H/2)/self.focal, -torch.ones(self.H,self.W)],dim=2)
        world_dirs = camera_dirs @ self.pose[:3,:3].T

        world_pos = self.pose[:3,-1]

        return world_pos, world_dirs

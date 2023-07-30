import torch

class Camera():
    def __init__(self, H, W, pose, focal, cx = W/2, cy = H/2, device = 'cpu'):
        self.H = H
        self.W = W
        self.device = device
        self.pose = pose.to(device)
        if type(focal) is tuple:
            self.fx, self.fy = focal
        else:
            self.fx = self.fy = focal
        self.cx = cx
        self.cy = cy

    def getRays(self):
        i = torch.ones((self.H, 1), device = self.device) @ (torch.arange(0, self.W, device = self.device)[None,:].float())
        j = (torch.arange(0, self.H, device = self.device)[:,None].float()) @ torch.ones((1, self.W),device = self.device)

        camera_dirs = torch.stack([(i-self.cx)/self.fx, -(j-self.cy)/self.fy, -torch.ones(self.H,self.W,device = self.device)],dim=2)
        world_dirs = camera_dirs @ self.pose[:3,:3].T

        world_pos = self.pose[:3,-1]
        world_pos = world_pos.expand(world_dirs.shape)

        return world_pos, world_dirs
    
    def getRay(self, i, j):
        camera_dir = torch.tensor([(i-self.W/2)/self.focal, (j-self.H/2)/self.focal, -1], device = self.device)
        world_dir = camera_dir.float() @ self.pose[:3,:3].T

        world_pos = self.pose[:3,-1]

        return world_pos, world_dir

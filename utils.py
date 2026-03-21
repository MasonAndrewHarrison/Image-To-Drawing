import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import filter
from scipy.ndimage import distance_transform_edt
import numpy as np


def points_from_pixel(image_sdf, positions):

    B, _, H, W = image_sdf.shape
    N = positions.shape[1]

    x = positions[:, :, 0]  
    y = positions[:, :, 1] 

    x_norm = (x / (W - 1)) * 2 - 1  
    y_norm = (y / (H - 1)) * 2 - 1  

    grid = torch.stack([x_norm, y_norm], dim=-1) 
    grid = grid.unsqueeze(2)                      

    sampled = F.grid_sample(image_sdf, grid, mode='bilinear', align_corners=True)

    return sampled.squeeze(1).squeeze(-1)
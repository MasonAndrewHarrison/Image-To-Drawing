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


def points_from_sdf(image_sdf, positions):

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


def point_from_image(xdog_image, positions):

    img_np = (image > threshold).cpu().numpy()
    sdf = distance_transform_edt(img_np) 

    sdf = torch.from_numpy(sdf).float().to(image.device)

    B, _, H, W = sdf.shape
    N = positions.shape[1]

    x = positions[:, :, 0]  
    y = positions[:, :, 1] 

    x_norm = (x / (W - 1)) * 2 - 1  
    y_norm = (y / (H - 1)) * 2 - 1  

    grid = torch.stack([x_norm, y_norm], dim=-1) 
    grid = grid.unsqueeze(2)                      

    sampled = F.grid_sample(sdf, grid, mode='bilinear', align_corners=True)

    return sampled.squeeze(1).squeeze(-1)


def smoothstep(a, b, t):
    """
    Smooth interpolation from a to b
    f: ℝ²×[0,1] → [a,b],  f(a,b,α) = (b-a)·α²(3-2α) + a
    """

    smooth_funct = (t**2)*(3-2*t)
    ab_lerp = (b-a)*smooth_funct + a

    return ab_lerp

def nested_smoothstep(t, iterations=1):  
    """
    Applies f(t) = t²(3-2t) recursively for n iterations
    out = f∘f∘...∘f(t), t ∈ [0, 1]
    """

    t = torch.clamp(t, 0, 1)
    smooth_funct = lambda t : (t**2)*(3-2*t)
    out = t

    for _ in range(iterations):
        out = smooth_funct(out)

    return out


def total_loss(strokes):

    

    return 0 
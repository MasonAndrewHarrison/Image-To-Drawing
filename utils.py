import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import filter
from differentiable_rasterizer import image_to_sdf, render_point_sdf
from scipy.ndimage import distance_transform_edt
import numpy as np


def points_from_sdf(image_sdf, positions, interpolation_mode: str = 'bilinear'):

    """
    Image_sdf is expected to be in [H, W]. 
    Positions is expected to be in [N, 2]
    Output will be [N]
    """

    H, W = image_sdf.shape
    N = positions.shape[1]
    x, y = torch.unbind(positions, dim=1)

    x = x.unsqueeze(0).unsqueeze(0)
    y = y.unsqueeze(0).unsqueeze(0)
    image_sdf = image_sdf.unsqueeze(0).unsqueeze(0)

    x_norm = (x / (W - 1)) * 2 - 1  
    y_norm = (y / (H - 1)) * 2 - 1  

    grid = torch.stack([x_norm, y_norm], dim=-1) 
    sampled = F.grid_sample(image_sdf, grid, mode=interpolation_mode, align_corners=True)

    return sampled[0, 0, 0, :]

def point_from_image(edge_image, position):

    """
    Image_sdf is expected to be in [H, W]. 
    Positions is expected to be in [2]
    Output will be a scaler
    """

    H, W = edge_image.shape
    x, y = torch.unbind(position, dim=0)
    x_norm = (x / (W - 1)) * 2 - 1  
    y_norm = (y / (H - 1)) * 2 - 1 

    point = torch.tensor(
        [x_norm, y_norm, 0, 0],
        device=edge_image.device
    ).unsqueeze(0)
    sdf_from_point = render_point_sdf(point, H, W, raw_sdf=True).squeeze(0)

    print(sdf_from_point.shape)

    plt.imshow(sdf_from_point.detach().cpu(), cmap='gray')
    plt.show()

    return 0

def points_from_image(edge_image, positions, interpolation_mode: str = 'bilinear'):

    """
    Edge Image is expected to be in [H, W]. 
    Positions is expected to be in [N, 2]
    Output will be [N]
    """

    edge_image = edge_image.squeeze(0)
    image_sdf = image_to_sdf(image=edge_image).squeeze(0)

    H, W = image_sdf.shape
    N = positions.shape[1]
    x, y = torch.unbind(positions, dim=1)

    x = x.unsqueeze(0).unsqueeze(0)
    y = y.unsqueeze(0).unsqueeze(0)
    image_sdf = image_sdf.unsqueeze(0).unsqueeze(0)
    

    x_norm = (x / (W - 1)) * 2 - 1  
    y_norm = (y / (H - 1)) * 2 - 1  

    grid = torch.stack([x_norm, y_norm], dim=-1) 
    sampled = F.grid_sample(image_sdf, grid, mode=interpolation_mode, align_corners=True)

    return sampled[0, 0, 0, :]


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




if __name__ == "__main__":


    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms = transforms.Compose([transforms.ToTensor()])
    dataset = ImageFolder(root='dataset_images/', transform=transforms)

    image,_ = dataset[0]
    _, height, width = image.shape
    image = image.mean(0).unsqueeze(0).unsqueeze(0).to(device)
    canny = filter.ex_difference_of_gaussians(image).squeeze(0).squeeze(0)
    
    
    point = torch.tensor([50, 50], device=device)
    dist = point_from_image(canny, point)
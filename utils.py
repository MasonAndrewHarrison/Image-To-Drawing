import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import filter
from differentiable_rasterizer import image_to_sdf, render_point_sdf, image_to_negative_sdf
from scipy.ndimage import distance_transform_edt
import numpy as np
import time
from typing_extensions import deprecated

def points_from_sdf(image_sdf, positions, interpolatiowidthn_mode: str = 'bilinear'):

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

def partial_point_from_image(edge_image, position, radius):

    """
    Image_sdf is expected to be in [H, W]. 
    Positions is expected to be in [2] (x, y).
    Output will be a scaler. Output will be -1 if 
    nothing is found or outside of radius.
    This function is not differantable and incompatable
    with pytorch autograd.
    """

    H, W = edge_image.shape
    x, y = torch.unbind(position, dim=0)

    x_start = torch.clip(x - radius, 0, W).__int__()
    x_end = torch.clip(x + radius, 0, W).__int__()
    y_start = torch.clip(y - radius, 0, H).__int__()
    y_end = torch.clip(y + radius, 0, H).__int__()

    edge_image = edge_image[y_start:y_end, x_start:x_end]
    edge_coords = torch.nonzero(~edge_image, as_tuple=False).float() 

    edge_y = edge_coords[:, 0] + y_start
    edge_x = edge_coords[:, 1] + x_start
    
    distance = torch.sqrt((edge_x - x)**2 + (edge_y - y)**2 + 1e-8)

    if distance.shape[0] == 0:
        return -1
    else:
        smallest_dist, closest_idx = torch.min(distance, dim=0)

        if smallest_dist > radius:
            return -1
        else:
            return smallest_dist

def faster_point_from_image(edge_image, position):

    """
    Image_sdf is expected to be in [H, W]. 
    Positions is expected to be in [2] (x, y).
    Output will be a scaler.
    """

    H, W = edge_image.shape
    x, y = torch.unbind(position, dim=0)

    edge_coords = torch.nonzero(~edge_image, as_tuple=False).float() 

    edge_y = edge_coords[:, 0]
    edge_x = edge_coords[:, 1]
    
    distance = torch.sqrt((edge_x - x)**2 + (edge_y - y)**2 + 1e-8)
    
    return distance.min()

@deprecated("Use faster_point_from_image instead")
def point_from_image(edge_image, position, interpolation_mode: str = 'bilinear'):

    """
    Image_sdf is expected to be in [H, W]. 
    Positions is expected to be in [2] (x, y).
    Output will be a scaler.
    """

    edge_image = edge_image.squeeze(0)
    sdf = image_to_sdf(image=edge_image).squeeze(0)

    H, W = sdf.shape
    x, y = torch.unbind(position.unsqueeze(0), dim=1)
    sdf = sdf.squeeze() 

    x0, y0 = x.floor().long(), y.floor().long()
    x1, y1 = (x0 + 1).clamp(max=W-1), (y0 + 1).clamp(max=H-1)
    x0, y0 = x0.clamp(min=0, max=W-1), y0.clamp(min=0, max=H-1)

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    x0, y0, x1, y1 = x0.squeeze(), y0.squeeze(), x1.squeeze(), y1.squeeze()
    distance = \
        wa.squeeze() * sdf[y0, x0] + \
        wb.squeeze() * sdf[y1, x0] + \
        wc.squeeze() * sdf[y0, x1] + \
        wd.squeeze() * sdf[y1, x1]

    return distance 


def points_from_image(edge_image, positions, interpolation_mode: str = 'bilinear'):

    """
    Edge Image is expected to be in [H, W]. 
    Positions is expected to be in [N, 2]
    Output will be [N]
    """

    edge_image = edge_image.squeeze(0)
    image_sdf = image_to_sdf(image=edge_image).squeeze(0)

    H, W = image_sdf.shape
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

def create_circle_mask(radius, device):

    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(radius*2, device=device),
        torch.arange(radius*2, device=device),
        indexing='ij'
    )
    circle_sdf = torch.sqrt((pixel_x - radius)**2 + (pixel_y - radius)**2 + 1e-8)
    mask = circle_sdf < radius

    return mask

def search_lowest_point(negative_sdf, position, radius: int = -1):

    """
    Image_sdf is expected to be in [H, W]. 
    Positions is expected to be in [2] (x, y).
    Output will be a the position of the lowest point in the range.
    """

    H, W = negative_sdf.shape
    x, y = torch.unbind(position, dim=0)
    x_start = 0
    y_start = 0

    if radius != -1:
        x_start = torch.clip(x - radius, 0, W).__int__()
        x_end = torch.clip(x + radius, 0, W).__int__()
        y_start = torch.clip(y - radius, 0, H).__int__()
        y_end = torch.clip(y + radius, 0, H).__int__()

        negative_sdf = negative_sdf[y_start:y_end, x_start:x_end]
        mask = create_circle_mask(radius, device=position.device)
        negative_sdf = torch.where(mask, negative_sdf, 0)
        plt.imshow(negative_sdf.detach().cpu(), cmap="grey")
        plt.show()
    
    updated_H, updated_W = negative_sdf.shape
    flatten_sdf = negative_sdf.view(-1)
    lowest_value, lowest_idx = torch.min(flatten_sdf, dim=0)

    print(lowest_value, lowest_idx)
    
    if lowest_value == 0:
        return torch.tensor([0, 0], device=position.device)

    else:
        lowest_x = x_start + lowest_idx % (updated_W)
        lowest_y = y_start + torch.floor(lowest_idx/ updated_H)

        return torch.tensor([lowest_x, lowest_y], device=position.device)

def vec_to_lowest_point(negative_sdf, position, search_radius: int = -1, magnatude: int = -1):

    lowest_point = search_lowest_point(negative_sdf, position, search_radius)
    print(lowest_point)
    if lowest_point[0] == 0 and lowest_point[1] == 0:
        return lowest_point

    vector = lowest_point - position
    if magnatude == -1:
        return vector
    else:
        current_magnatude = torch.sqrt(vector[0]**2 + vector[1]**2)
        k = magnatude / current_magnatude
        scaled_vec = vector * k
        return scaled_vec

def erase_negative_sdf(negative_sdf, point, radius):

    H, W = negative_sdf.shape
    x, y = torch.unbind(point, dim=0)
    radius = 10

    x_start = torch.clip(x - radius, 0, W).__int__()
    x_end = torch.clip(x + radius, 0, W).__int__()
    y_start = torch.clip(y - radius, 0, H).__int__()
    y_end = torch.clip(y + radius, 0, H).__int__()

    mask = ~create_circle_mask(radius, device=device)

    partial_negative_sdf = negative_sdf[y_start:y_end, x_start:x_end]
    negative_sdf[y_start:y_end, x_start:x_end] = torch.where(mask, partial_negative_sdf, 0)

    return negative_sdf



if __name__ == "__main__":


    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms = transforms.Compose([transforms.ToTensor()])
    dataset = ImageFolder(root='dataset_images/', transform=transforms)

    image,_ = dataset[0]
    _, height, width = image.shape
    image = image.mean(0).unsqueeze(0).unsqueeze(0).to(device)
    canny = filter.ex_difference_of_gaussians(image)
    scaler = 5
    
    canny = F.interpolate(
        canny.float(),
        size=[height*scaler, width*scaler],
        mode="bicubic",
        align_corners=False
    ).squeeze(0).squeeze(0)

    point = torch.tensor([(height*scaler)/2, (width*scaler)/2], device=device)
    og = torch.tensor([(height*scaler)/2, (width*scaler)/2], device=device)

    negative_sdf = image_to_negative_sdf(canny.unsqueeze(0)).squeeze(0)
    vector = vec_to_lowest_point(negative_sdf, point, search_radius=20)

    #TODO why is this vector so large?????
    print(f"print first vector{vector}")
    point = point + vector

    #negative_sdf = erase_negative_sdf(negative_sdf, point, 20)


    """vector = vec_to_lowest_point(negative_sdf, point, search_radius=22)
    print(vector)
    point2 = point + vector"""
    
    plt.imshow(negative_sdf.detach().cpu(), cmap="grey")
    plt.scatter(point[0].cpu().numpy(), point[1].cpu().numpy())
    #plt.scatter(point2[1].cpu().numpy(), point2[0].cpu().numpy())
    plt.scatter(og[0].cpu().numpy(), og[1].cpu().numpy())

    plt.annotate(
        "OG",
        (point[0].cpu().numpy(), point[1].cpu().numpy()),
        color="blue",
        fontsize=10,
    )
    plt.show()


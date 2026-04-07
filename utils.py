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
import cv2
import numpy as np


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
    
    updated_H, updated_W = negative_sdf.shape
    flatten_sdf = negative_sdf.view(-1)
    lowest_value, lowest_idx = torch.min(flatten_sdf, dim=0)
    
    if lowest_value == 0:
        return torch.tensor([0, 0], device=position.device)

    else:
        lowest_x = x_start + lowest_idx % (updated_W)
        lowest_y = y_start + torch.floor(lowest_idx/ updated_W)

        return torch.tensor([lowest_x, lowest_y], device=position.device)

def vec_to_lowest_point(negative_sdf, position, search_radius: int = -1, magnatude: int = -1):

    lowest_point = search_lowest_point(negative_sdf, position, search_radius)
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

def filter_noise(num_labels, labels, stats, size_threshold: int = 300):
    """
    This takes in np.uint8 as input.
    """

    filtered = np.zeros_like(labels, dtype=np.uint8)

    for i in range(1, num_labels):
        size = stats[i, cv2.CC_STAT_AREA]
        if size > size_threshold:
            filtered[labels == i] = labels[labels == i]

    return num_labels, filtered, stats

def density_filter(num_labels, labels, stats, density_threshold: float = 17.25):
    """
    This takes in np.uint8 as input.
    """

    filtered = np.zeros_like(labels, dtype=np.uint8)

    for i in range(1, num_labels):
        height = stats[i, cv2.CC_STAT_HEIGHT]
        width = stats[i, cv2.CC_STAT_WIDTH]
        area = stats[i, cv2.CC_STAT_AREA]

        vertical_density = area / height
        horizontal_density = area / width

        if (vertical_density < density_threshold) or (horizontal_density < density_threshold):
            filtered[labels == i] = labels[labels == i]

    return num_labels, filtered, stats


def separate_pixels(image):
    """
    This takes in torch.uint8 as input. And
    return np.uint8 as output.
    """

    image = image.detach().cpu().numpy().astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image)

    return num_labels, labels, stats

def separate_to_layers(num_labels, labels):

    layers = np.zeros((0, *labels.shape), dtype=np.uint8)

    for i in range(1, num_labels):
        new_layer = (labels == i)
        not_empty = new_layer.sum() > 0
        if not_empty:

            new_layer = new_layer.reshape(1, *new_layer.shape)
            layers = np.concatenate([layers, new_layer], axis=0)

    return layers


if __name__ == "__main__":


    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms = transforms.Compose([transforms.ToTensor()])
    dataset = ImageFolder(root='dataset_images/', transform=transforms)

    image,_ = dataset[0]
    _, height, width = image.shape
    image = image.mean(0).unsqueeze(0).unsqueeze(0).to(device)
    ex_dog = filter.ex_difference_of_gaussians(image, threshold=0.05)
    scaler = 4
    
    ex_dog = F.interpolate(
        ex_dog.float(),
        size=[height*scaler, width*scaler],
        mode="bicubic",
        align_corners=False
    ).squeeze(0).squeeze(0)
    binary = (ex_dog < 0.5).to(torch.uint8)

    data = separate_pixels(binary)
    data = filter_noise(*data)
    num_labels, labels,_ = density_filter(*data)
    layers = separate_to_layers(num_labels, labels)


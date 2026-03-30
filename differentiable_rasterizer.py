import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import filter
import kornia.contrib
from scipy.ndimage import distance_transform_edt
import numpy as np

@torch.compile
def render_lines_sdf(strokes: torch.Tensor, height: int, width: int, raw_sdf: bool) -> torch.Tensor:

    device = strokes.device

    line_count = strokes.shape[0]
    canvas = torch.zeros(1, height, width).to(device)

    pixel_x = torch.linspace(0, 1, width, device=device).view(1, 1, -1)   # (1, 1, W)
    pixel_y = torch.linspace(0, 1, height, device=device).view(1, -1, 1)  # (1, H, 1)

    x1 = strokes[:-1, 0].view(-1, 1, 1)
    y1 = strokes[:-1, 1].view(-1, 1, 1)
    x2 = strokes[1:, 0].view(-1, 1, 1)
    y2 = strokes[1:, 1].view(-1, 1, 1)
    
    sigma = strokes[1:, 2].view(-1, 1, 1)
    radius = strokes[1:, 3].view(-1, 1, 1)

    # AB→ = B - A
    vector_ab_x = x2 - x1
    vector_ab_y = y2 - y1
    vector_ab_length_squared = vector_ab_x**2 + vector_ab_y**2 + 1e-8

    # AP→ = P - A
    vector_ap_x = (pixel_x - x1)
    vector_ap_y = (pixel_y - y1)

    # t = (AP→ · AB→) / (||AB→||²) 
    t = (vector_ap_x * vector_ab_x + vector_ap_y * vector_ab_y) / vector_ab_length_squared
    
    # This keeps the t value in between A and B
    t = t.clamp(0, 1)

    # AT→ = A + t * AB→
    vector_at_x = x1 + t * vector_ab_x
    vector_at_y = y1 + t * vector_ab_y

    # distance = √( (P_x - AT→_x)² + (P_y - AT→_y)² )
    distance = torch.sqrt((pixel_x - vector_at_x)**2 + (pixel_y - vector_at_y)**2 + 1e-8)
    
    if raw_sdf:
        sdf = distance - radius
        sdf = sdf.clamp(min=0)

        canvas = sdf.min(dim=0).values
        canvas = canvas.unsqueeze(0)

    else:
        
        # Sign Distance Function
        sdf = distance - radius
        sdf = sdf.clamp(min=0)

        # g(sdf) = e^( -sdf² / 2σ²)
        gaussian_sdf = torch.exp(-sdf**2 / (2 * sigma**2 + 1e-8)).clamp(min=1e-6)
        gaussian_sdf = gaussian_sdf #3 * opacity
        canvas = gaussian_sdf
        #1 - canvas

        canvas = canvas.max(dim=0).values
        canvas = canvas.unsqueeze(0)

        canvas = 1 - canvas

    return canvas
    

def render_point_sdf(strokes: torch.Tensor, height: int, width: int, raw_sdf: bool) -> torch.Tensor:

    device = strokes.device
    canvas = torch.zeros(1, height, width).to(device)

    pixel_x = torch.linspace(0, 1, width, device=device).repeat(1, height, 1)
    pixel_y = torch.linspace(0, 1, height, device=device).repeat(1, width, 1).permute(0, 2, 1)

    x = strokes[0, 0].view(1, 1, 1).expand(1, height, width)
    y = strokes[0, 1].view(1, 1, 1).expand(1, height, width)
    
    sigma = strokes[0, 2].view(1, 1, 1).expand(1, height, width)
    radius = strokes[0, 3].view(1, 1, 1).expand(1, height, width)

    # distance = √( (P_x - x)² + (P_y - y)² )
    distance = torch.sqrt((pixel_x - x)**2 + (pixel_y - y)**2 + 1e-8)
    
    if raw_sdf:
        sdf = distance - radius
        canvas = sdf.clamp(min=0)

    else:
        
        # Sign Distance Function
        sdf = distance - radius
        sdf = sdf.clamp(min=0)

        # g(sdf) = e^( -sdf² / 2σ²)
        gaussian_sdf = torch.exp(-sdf**2 / (2 * sigma**2 + 1e-8)).clamp(min=1e-6)
        canvas = gaussian_sdf

        canvas = 1 - canvas

    return canvas


def render_sdf_batched(strokes: torch.Tensor, height: int, width: int, raw_sdf: bool) -> torch.Tensor:

    line_count = strokes.shape[1]
    batch_size = strokes.shape[0]
    all_canvas = torch.ones(batch_size, 1, height, width).to(device=strokes.device)

    if line_count == 1:
        all_canvas = torch.func.vmap(
        lambda stroke: render_point_sdf(stroke, height, width, raw_sdf)
        )(strokes)

    elif line_count > 1:
        all_canvas = torch.func.vmap(
        lambda stroke: render_lines_sdf(stroke, height, width, raw_sdf)
        )(strokes)

    return all_canvas


def image_to_sdf(image: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:

    binary_np = (image > threshold).float().squeeze().cpu().numpy()
    edt = distance_transform_edt(binary_np)
    sdf = torch.from_numpy(edt).float().to(image.device).unsqueeze(1)

    return sdf


import numpy as np
import torch
import torch.nn.functional as F
import time

def scale(image: torch.Tensor, scaler: float=1) -> torch.Tensor:

    _, _, H, W = image.shape
    H *= scaler
    W *= scaler

    image = F.interpolate(image, (H, W), mode="bilinear", align_corners=False)

    return image

def invert(image: torch.Tensor) -> torch.Tensor:

    return image*-1 + 1

@torch.jit.script
def guassian_blur(image: torch.Tensor, big_blur: bool = True) -> torch.Tensor:

    device = image.device

    gaussian_blur = torch.tensor([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1],
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 16

    big_gaussian_blur = torch.tensor([
        [1, 4, 6, 4, 1],
        [4,16,24,16, 4],
        [6,24,36,24, 6],
        [4,16,24,16, 4],
        [1, 4, 6, 4, 1],
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 256.0

    if big_blur:
        image = F.conv2d(image, big_gaussian_blur, padding=2)
    else:
        image = F.conv2d(image, gaussian_blur, padding=1)

    return image

@torch.jit.script
def angle_rounder(theda: torch.Tensor) -> torch.Tensor:
    factor = 4.0 / 3.14159265359  # (180/π) / 45
    return torch.fmod(torch.round(theda * factor) * 45.0, 180.0).long()

def non_maximum_suppression(magnitude, round_angle, threshold=0.005):

    m = magnitude
    mp = F.pad(m, (1, 1, 1, 1))
    
    B, C, H, W = m.shape
    left       = mp[:, :, 1:H+1, 0:W]
    right      = mp[:, :, 1:H+1, 2:W+2]
    top        = mp[:, :, 0:H,   1:W+1]
    bottom     = mp[:, :, 2:H+2, 1:W+1]
    top_right  = mp[:, :, 0:H,   2:W+2]
    bot_left   = mp[:, :, 2:H+2, 0:W]
    top_left   = mp[:, :, 0:H,   0:W]
    bot_right  = mp[:, :, 2:H+2, 2:W+2]

    is_max_0   = (m > left)      & (m > right)
    is_max_45  = (m > top_right) & (m > bot_left)
    is_max_90  = (m > top)       & (m > bottom)
    is_max_135 = (m > top_left)  & (m > bot_right)

    mask_0   = round_angle == 0
    mask_45  = round_angle == 45
    mask_90  = round_angle == 90
    mask_135 = round_angle == 135
    
    is_max = (
        (mask_0   & is_max_0)   |
        (mask_45  & is_max_45)  |
        (mask_90  & is_max_90)  |
        (mask_135 & is_max_135)
    )

    image = torch.where(is_max & (m >= threshold), m, torch.zeros_like(m))
    
    return image

@torch.jit.script
def hysteresis(image: torch.Tensor, threshold: float=0.06) -> tuple[torch.Tensor, bool]:

    old_img = image.clone().detach()

    _, _, H, W = image.shape
    padded_img = F.pad(image, (1, 1, 1, 1))
    
    surrounding_pixels = torch.concat([
        padded_img[:, :, 0:H,   1:W+1],
        padded_img[:, :, 2:H+2, 1:W+1],
        padded_img[:, :, 1:H+1, 0:W],
        padded_img[:, :, 1:H+1, 2:W+2],
        padded_img[:, :, 0:H,   2:W+2],
        padded_img[:, :, 2:H+2, 0:W],
        padded_img[:, :, 0:H,   0:W],
        padded_img[:, :, 2:H+2, 2:W+2],
    ], dim=1)

    has_strong_neighbor = (surrounding_pixels >= threshold).any(dim=1, keepdim=True)
    active_pixel = image > 0.0

    image = torch.where(has_strong_neighbor & active_pixel, torch.ones_like(image), image)
    is_complete = torch.equal(old_img, image)

    return image, is_complete

def sobel_edge_detection(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

    device = image.device

    sobel_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ], dtype=torch.float32)

    sobel_y = torch.tensor([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ], dtype=torch.float32)

    sobel_x = sobel_x.unsqueeze(0).unsqueeze(0).to(device)
    sobel_y = sobel_y.unsqueeze(0).unsqueeze(0).to(device)
    
    sharp_x = F.conv2d(image, sobel_x, padding=1, groups=1) 
    sharp_y = F.conv2d(image, sobel_y, padding=1, groups=1)

    magnitude = torch.sqrt(sharp_x * sharp_x + sharp_y * sharp_y)
    magnitude = magnitude / magnitude.max()

    # θ = arctan(y/x) * 180 / π
    angle = torch.atan2(sharp_y, sharp_x) * 180.0 / np.pi
    angle = angle % 180

    return magnitude, angle

def canny(image, device="cpu", threshold_1=0.005, threshold_2=0.06):

    _, _, H, W = image.shape

    blurred = guassian_blur(image, big_blur=True)
    magnitude, angle = sobel_edge_detection(blurred)

    round_angle = angle_rounder(angle)

    image = non_maximum_suppression(magnitude, round_angle, threshold=threshold_1)
    image = image[:, :, 1:H-1, 1:W-1]

    is_complete = False
    while not is_complete:
        image, is_complete = hysteresis(image, threshold=threshold_2)

    image = image - 0.99
    image = image.clamp_(min=0.00) * 100

    return image



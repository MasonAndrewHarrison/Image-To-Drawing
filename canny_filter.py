import numpy as np
import torch
import torch.nn.functional as F
import time


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

    img = torch.where(is_max & (m >= threshold), m, torch.zeros_like(m))
    
    return img

def hysteresis(img, threshold=0.06):

    old_img = img.clone().detach()

    _, _, H, W = img.shape
    padded_img = F.pad(img, (1, 1, 1, 1))
    
    top       = padded_img[:, :, 0:H,   1:W+1]
    bottom    = padded_img[:, :, 2:H+2, 1:W+1]
    left      = padded_img[:, :, 1:H+1, 0:W]
    right     = padded_img[:, :, 1:H+1, 2:W+2]
    top_right = padded_img[:, :, 0:H,   2:W+2]
    bot_left  = padded_img[:, :, 2:H+2, 0:W]
    top_left  = padded_img[:, :, 0:H,   0:W]
    bot_right = padded_img[:, :, 2:H+2, 2:W+2]

    surrounding_pixels = torch.concat([
        top,
        bottom,
        left,
        right,
        top_right,
        bot_left,
        top_left,
        bot_right
    ], dim=1)

    mask_1 = (surrounding_pixels >= torch.ones_like(img)*threshold).any(dim=1)
    mask_2 = (img != torch.zeros_like(img))

    img = torch.where(mask_1 & mask_2, torch.ones_like(img), img)
    is_complete = (old_img == img).all()

    return img, is_complete

def convert(img, device="cpu", threshold_1=0.005, threshold_2=0.06):

    gaussian_blur = torch.tensor([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1],
    ], dtype=torch.float32)

    big_gaussian_blur = torch.tensor([
        [1, 4, 6, 4, 1],
        [4,16,24,16, 4],
        [6,24,36,24, 6],
        [4,16,24,16, 4],
        [1, 4, 6, 4, 1],
    ], dtype=torch.float32)

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

    gaussian_blur = gaussian_blur.unsqueeze(0).unsqueeze(0).to(device)
    big_gaussian_blur = big_gaussian_blur.unsqueeze(0).unsqueeze(0).to(device)
    sobel_x = sobel_x.unsqueeze(0).unsqueeze(0).to(device)
    sobel_y = sobel_y.unsqueeze(0).unsqueeze(0).to(device)

    blurred = F.conv2d(img, big_gaussian_blur, padding=1)
    sharp_x = F.conv2d(blurred, sobel_x, padding=1, groups=1) 
    sharp_y = F.conv2d(blurred, sobel_y, padding=1, groups=1)

    magnitude = torch.sqrt(sharp_x * sharp_x + sharp_y * sharp_y)
    magnitude = magnitude / magnitude.max()

    # θ = arctan(y/x) * 180 / π
    angle = torch.atan2(sharp_y, sharp_x) * 180.0 / np.pi
    angle = angle % 180

    _, _, H, W = angle.shape
    img = torch.zeros_like(magnitude)
    round_angle = angle_rounder(angle)

    img = non_maximum_suppression(magnitude, round_angle, threshold=threshold_1)
    img = img[:, :, 1:H-1, 1:W-1]

    hysteresis_finished = False
    while not hysteresis_finished:
        img, hysteresis_finished = hysteresis(img, threshold=threshold_2)

    img = img - 0.99
    img = img.clamp_(min=0.00) * 100

    return img



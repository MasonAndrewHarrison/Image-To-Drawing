import numpy as np
import torch
import torch.nn.functional as F
from nms import non_maximum_suppression

def angle_rounder(theda):
    if theda >= torch.tensor([0], device=device) and theda <= torch.tensor([22.5], device=device):
        return 0
    elif theda > torch.tensor([22.5], device=device) and theda <= torch.tensor([67.5], device=device):
        return 45
    elif theda > torch.tensor([67.5], device=device) and theda <= torch.tensor([112.5], device=device):
        return 90
    elif theda > torch.tensor([112.5], device=device) and theda <= torch.tensor([157.5], device=device):
        return 135
    elif theda > torch.tensor([157.5], device=device) and theda <= torch.tensor([180], device=device):
        return 0
    else:
        return -1

def _non_maximum_suppression(magnitude, angle, threshold=0.05):

    for x in range(1, W-1):
        for y in range(1, H-1):

            if angle_rounder(angle[:, :, y, x]) == 0:
                if magnitude[:, :, y, x] > magnitude[:, :, y, x-1] and magnitude[:, :, y, x] > magnitude[:, :, y, x+1]:
                    img[:, :, y, x] = magnitude[:, :, y, x] if (magnitude[:, :, y, x] >= 0.05) else 0

            elif angle_rounder(angle[:, :, y, x]) == 45:
                if magnitude[:, :, y, x] > magnitude[:, :, y-1, x+1] and magnitude[:, :, y, x] > magnitude[:, :, y+1, x-1]:
                    img[:, :, y, x] = magnitude[:, :, y, x] if (magnitude[:, :, y, x] >= 0.05) else 0
            
            elif angle_rounder(angle[:, :, y, x]) == 90:
                if magnitude[:, :, y, x] > magnitude[:, :, y-1, x] and magnitude[:, :, y, x] > magnitude[:, :, y+1, x]:
                    img[:, :, y, x] = magnitude[:, :, y, x] if (magnitude[:, :, y, x] >= 0.05) else 0

            elif angle_rounder(angle[:, :, y, x]) == 135:
                if magnitude[:, :, y, x] > magnitude[:, :, y-1, x-1] and magnitude[:, :, y, x] > magnitude[:, :, y+1, x+1]:
                    img[:, :, y, x] = magnitude[:, :, y, x] if (magnitude[:, :, y, x] >= 0.05) else 0

    return img

def convert(img, device="cpu"):

    gaussian_blur = torch.tensor([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1],
    ], dtype=torch.float32)

    big_blur = torch.tensor([
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
    big_blur = big_blur.unsqueeze(0).unsqueeze(0).to(device)
    sobel_x = sobel_x.unsqueeze(0).unsqueeze(0).to(device)
    sobel_y = sobel_y.unsqueeze(0).unsqueeze(0).to(device)

    inverted = img.mean(0).unsqueeze(0).unsqueeze(0).to(device) / 256.0

    blurred = F.conv2d(inverted, gaussian_blur, padding=1)
    sharp_x = F.conv2d(blurred, sobel_x, padding=1, groups=1) 
    sharp_y = F.conv2d(blurred, sobel_y, padding=1, groups=1)

    magnitude = torch.sqrt(sharp_x * sharp_x + sharp_y * sharp_y)
    magnitude = magnitude / magnitude.max()

    # θ = arctan(y/x) * 180 / π
    angle = torch.atan2(sharp_y, sharp_x) * 180.0 / np.pi
    angle = angle % 180

    _, _, H, W = angle.shape

    img = torch.zeros_like(magnitude)
    img = non_maximum_suppression(magnitude, angle)

    '''
    # Hysteresis
    for x in range(1, W-1):
        for y in range(1, H-1):
            
            if img[:, :, y, x] is not torch.tensor([0], device=device):
                top_left = img[:, :, y+1, x-1]
                left = img[:, :, y, x-1]
                bottom_left = img[:, :, y-1, x-1]
                
                top_right = img[:, :, y+1, x+1]
                right = img[:, :, y, x+1]
                bottom_right = img[:, :, y-1, x+1]

                bottom = img[:, :, y-1, x]
                top = img[:, :, y+1, x]

                surrounding_pixels = torch.stack([top_left, left, bottom_left, top_right, right, bottom_right, bottom, top])

                if (surrounding_pixels >= 0.5).any():
                    img[:, :, y, x] = torch.tensor([1], device=device)'''

    return img

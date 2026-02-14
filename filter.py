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
def gaussian_blur(image: torch.Tensor, big_blur: bool = False) -> torch.Tensor:

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

def gaussian(x: float, y: float, sigma: float = 1, mean: int = 0) -> float:

    x -= mean
    y -= mean

    # 1 / (2πσ^2)
    normalization_constant = 1/(2 * np.pi * sigma**2)
    # -[ (x^2 + y^2) / 2σ^2]
    kernel_exponent = -(x**2 + y**2)/(2 * sigma**2)
    # normalization_constant * ( e^kernel_exponent )
    gaussian = normalization_constant * np.exp(kernel_exponent)

    return gaussian

def variable_gaussian_blur(image: torch.Tensor, size: int, sigma: float = 1) -> torch.Tensor:

    device = image.device

    kernel = torch.empty((size, size), dtype=torch.float32)
    offset = int(np.floor(size/2))

    for i in range(size):
        for j in range(size):

            kernel[i, j] = gaussian(x=i, y=j, sigma=sigma, mean=offset)

    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0).to(device)
    image = F.conv2d(image, kernel, padding=offset)

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

    blurred = gaussian_blur(image, big_blur=True)
    magnitude, angle = sobel_edge_detection(blurred)

    round_angle = angle_rounder(angle)

    image = non_maximum_suppression(magnitude, round_angle, threshold=threshold_1)

    is_complete = False
    while not is_complete:
        image, is_complete = hysteresis(image, threshold=threshold_2)

    image = image - 0.99
    image = image.clamp_(min=0.00) * 100

    image = invert(image)

    return image

def difference_of_gaussians(image: torch, sigma: float = 1.4, k: float = 1.6) -> torch.Tensor:

    size1 = 2 * int(3 * sigma) + 1
    size2 = 2 * int(3 * k * sigma) + 1
    
    small_blur = variable_gaussian_blur(image, size=size1, sigma=sigma)
    big_blur = variable_gaussian_blur(image, size=size2, sigma=k * sigma)

    return small_blur - big_blur

def ex_difference_of_gaussians(
        image: torch.Tensor, 
        tau: float = 0.99, 
        epsilon: float = 0.0,
        phi: float = 50,
        sigma: float = 1.4, 
        threshold: float = 0.7,
        use_threshold: bool = True,
        k: float = 1.6) -> torch.Tensor:

    _, _, H, W = image.shape
    base_resolution = 300
    current_resolution = (H + W)/ 2
    scale_factor = current_resolution / base_resolution

    scaled_simga = sigma * scale_factor

    size1 = 2 * int(3 * scaled_simga) + 1
    size2 = 2 * int(3 * k * scaled_simga) + 1

    # range [0, 1]
    image = image / image.max()
    
    small_blur = variable_gaussian_blur(image, size=size1, sigma=scaled_simga)
    big_blur = variable_gaussian_blur(image, size=size2, sigma=k * scaled_simga)

    # D(σ,k,τ) = G_σ - τ·G_kσ
    image = small_blur - (tau * big_blur)

    # T(u,ε,φ) = 1 if u ≥ ε, else 1 + tanh(φ(u - ε))
    image = torch.where(
        image >= epsilon, 
        torch.ones_like(image), 
        1.0 + torch.tanh(phi * (image - epsilon))
    )

    if use_threshold:
        image = image > 0.7

    return image

def flow_ex_difference_of_gaussians(image: torch.Tensor) -> torch.Tensor:

    return image


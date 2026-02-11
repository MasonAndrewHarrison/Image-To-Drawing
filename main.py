
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

transforms = transforms.Compose([transforms.ToTensor()])
dataset = ImageFolder(root='img/', transform=transforms)

tensor,_ = dataset[1]
inverted = tensor.mean(0).unsqueeze(0).unsqueeze(0).to(device) / 256.0

gaussian_blur = torch.tensor([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1],
], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

big_blur = torch.tensor([
    [1, 4, 6, 4, 1],
    [4,16,24,16, 4],
    [6,24,36,24, 6],
    [4,16,24,16, 4],
    [1, 4, 6, 4, 1],
], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

sobel_x = torch.tensor([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

sobel_y = torch.tensor([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1],
], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

blurred = F.conv2d(inverted, gaussian_blur, padding=1)
sharp_x = F.conv2d(blurred, sobel_x, padding=1, groups=1) 
sharp_y = F.conv2d(blurred, sobel_y, padding=1, groups=1)

magnitude = torch.sqrt(sharp_x * sharp_x + sharp_y * sharp_y)
magnitude = magnitude / magnitude.max()

# θ = arctan(y/x) * 180 / π
angle = torch.atan2(sharp_y, sharp_x) * 180.0 / np.pi
angle = angle % 180

_, _, H, W = angle.shape

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

img = torch.zeros_like(magnitude)

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

'''
for x in range(1, W-1):
    for y in range(1, H-1):
        
        if img[:, :, y, x] >= torch.tensor([0.05], device=device):'''



tensor = img

print(tensor.shape)

#tensor = torch.ones_like(tensor) - tensor


tensor = tensor.squeeze(0).squeeze(0)
plt.imshow(tensor, cmap="grey")
plt.show()




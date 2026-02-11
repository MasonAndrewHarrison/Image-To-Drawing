
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms


transforms = transforms.Compose([transforms.ToTensor()])


dataset = ImageFolder(root='img/', transform=transforms)

tensor,_ = dataset[0]
inverted = tensor.mean(0).unsqueeze(0).unsqueeze(0) / 256.0

gaussian_blur = torch.tensor([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1],
], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

big_blur = torch.tensor([
    [1, 4, 6, 4, 1],
    [4,16,24,16, 4],
    [6,24,36,24, 6],
    [4,16,24,16, 4],
    [1, 4, 6, 4, 1],
], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

sobel_x = torch.tensor([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

sobel_y = torch.tensor([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1],
], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

blurred = F.conv2d(inverted, gaussian_blur, padding=1)
sharp_x = F.conv2d(blurred, sobel_x, padding=1, groups=1) 
sharp_y = F.conv2d(blurred, sobel_y, padding=1, groups=1)

magnitude = torch.sqrt(sharp_x * sharp_x + sharp_y * sharp_y)
magnitude = magnitude / magnitude.max()

# θ = arctan(y/x) * 180 / π
angle = torch.atan2(sharp_y, sharp_x) * 180.0 / np.pi
angle = angle % 180


tensor = angle

print(tensor.shape)
print(magnitude.shape)

tensor = torch.ones_like(tensor) - tensor


tensor = tensor.squeeze(0).squeeze(0).permute(0, 1)
plt.imshow(tensor, cmap="grey")
plt.show()




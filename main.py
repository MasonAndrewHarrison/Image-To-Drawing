import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from canny_filter import convert

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

transforms = transforms.Compose([transforms.ToTensor()])
dataset = ImageFolder(root='img/', transform=transforms)

tensor,_ = dataset[1]
tensor = tensor.requires_grad_(True)

tensor = tensor.mean(0).unsqueeze(0).unsqueeze(0).to(device) / 256.0

print(tensor.shape)
_, _, H, W = tensor.shape



img = convert(tensor, device=device, threshold_1=0.005, threshold_2=0.08)
tensor = img

tensor = F.interpolate(tensor, (H*2, W*2), mode="bilinear", align_corners=False)

gaussian_blur = torch.tensor([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1],
], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 16

big_blur = torch.tensor([
    [1, 4, 6, 4, 1],
    [4,16,24,16, 4],
    [6,24,36,24, 6],
    [4,16,24,16, 4],
    [1, 4, 6, 4, 1],
], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 256.0


tensor = F.conv2d(tensor, big_blur, padding=1)
#tensor = (tensor <= torch.ones_like(tensor)*0.05)
tensor = F.interpolate(tensor, (H, W), mode="bilinear", align_corners=False)

tensor = tensor.squeeze(0).squeeze(0).detach().cpu()
plt.imshow(tensor, cmap="grey")
plt.show()




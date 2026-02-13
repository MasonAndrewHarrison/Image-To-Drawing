import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
import filter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = transforms.Compose([transforms.ToTensor()])
dataset = ImageFolder(root='img/', transform=transforms)

tensor,_ = dataset[1]
tensor = tensor.requires_grad_(True)
tensor = tensor.mean(0).unsqueeze(0).unsqueeze(0).to(device) / 256.0

tensor = filter.canny(tensor, device=device, threshold_1=0.005, threshold_2=0.08)
tensor = filter.scale(tensor, 2)
tensor = filter.invert(tensor)

tensor = tensor.squeeze(0).squeeze(0).detach().cpu()
plt.imshow(tensor, cmap="grey")
plt.show()




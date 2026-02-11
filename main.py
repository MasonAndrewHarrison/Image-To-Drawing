import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from canny_filter import convert

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

transforms = transforms.Compose([transforms.ToTensor()])
dataset = ImageFolder(root='img/', transform=transforms)

tensor,_ = dataset[0]
tensor = tensor.requires_grad_(True)

print(tensor.requires_grad)

img = convert(tensor, device=device)

print(img.grad)

tensor = img

print(tensor.shape)

#tensor = torch.ones_like(tensor) - tensor


tensor = tensor.squeeze(0).squeeze(0)
plt.imshow(tensor, cmap="grey")
plt.show()




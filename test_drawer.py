import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import filter
from differentiable_rasterizer import image_to_sdf
from utils import *
from draw import Drawer


transforms = transforms.Compose([transforms.ToTensor()])
dataset = ImageFolder(root='dataset_images/', transform=transforms)

device = "cuda" if torch.cuda.is_available() else "cpu"
image,_ = dataset[0]
_, height, width = image.shape
image = image.mean(0).unsqueeze(0).unsqueeze(0).to(device)
canny = filter.canny(image)

drawer = Drawer(edge_image=canny)
point = drawer(0)

    
print(point)

point = point.detach().cpu()
plt.scatter(point[:, :, 0], point[:, :, 1])
plt.colorbar()
plt.gca().invert_yaxis()


xdog = filter.canny(image)
sdf = xdog[0, 0, :, :].detach().cpu()
plt.imshow(sdf, cmap='gray')

plt.show()
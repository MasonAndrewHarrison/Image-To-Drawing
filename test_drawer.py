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
canny = filter.canny(image).squeeze(0).squeeze(0)
sdf = image_to_sdf(canny).squeeze(0)

print(canny.shape)
print(sdf.shape)

drawer = Drawer(edge_image=canny)

stroke = torch.tensor([50 , 200, 0.005, 0.005], device="cuda").unsqueeze(0)
drawer.draw(stroke)

stroke = torch.tensor([200 , 50, 0.005, 0.005], device="cuda").unsqueeze(0)
drawer.draw(stroke)

print(sdf.shape)
distance = drawer.point_from_sdf(sdf, -1)
print(distance)

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


transforms = transforms.Compose([transforms.ToTensor()])
dataset = ImageFolder(root='dataset_images/', transform=transforms)

device = "cuda" if torch.cuda.is_available() else "cpu"
image,_ = dataset[0]
_, height, width = image.shape
image = image.mean(0).unsqueeze(0).unsqueeze(0).to(device)
canny = filter.canny(image)



grid_shape = (100, 100)
grid_w = grid_shape[1]
grid_h = grid_shape[0]
sdf = image_to_sdf(canny).repeat(grid_h, 1, 1, 1)

#TODO fix error around the board. Needs to stay in [1, :-2] for x and [1, :-3] for y
x_grid = torch.linspace(1, width-2, grid_w, device=device).repeat(grid_h, 1)
y_grid = torch.linspace(1, height-3, grid_h, device=device).repeat(grid_w, 1).permute(1, 0)
grid = torch.stack((x_grid, y_grid), dim=2)

grid = grid.detach().requires_grad_(True)


for _ in range(3):

    output = points_from_sdf(sdf, grid, interpolation_mode="bicubic")
    loss = output.unsqueeze(2).repeat(1, 1, 2)

    raw_grad = torch.autograd.grad(loss.mean(), grid,)[0]
    direction = raw_grad / (raw_grad.norm(dim=-1, keepdim=True) + 1e-8)
    scaled_grad = direction * loss

    with torch.no_grad():
        grid -= scaled_grad


    grid = grid.detach().requires_grad_(True)

    

output = output.detach().cpu()
grid = grid.detach().cpu()
plt.scatter(grid[:, :, 0], grid[:, :, 1], c=output, cmap="plasma")
plt.colorbar()
plt.gca().invert_yaxis()


xdog = filter.ex_difference_of_gaussians(image)
sdf = xdog[0, 0, :, :].detach().cpu()
plt.imshow(sdf, cmap='gray')

plt.show()
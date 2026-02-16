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

fig, axes = plt.subplots(3, len(dataset), figsize=(16, 4))


for i in range(len(dataset)):

    idx = 0
    tensor,_ = dataset[i]

    tensor = tensor.requires_grad_(True)
    tensor = tensor.mean(0).unsqueeze(0).unsqueeze(0).to(device) / 256.0

    image = tensor.clone().detach()

    tensor = filter.variable_gaussian_blur(tensor, size=3, sigma=1.0)
    
    canny = filter.canny(tensor)

    tensor = filter.scale(tensor, 5)

    xdog = filter.ex_difference_of_gaussians(tensor)

    canny_display = canny.squeeze(0).squeeze(0).detach().cpu()
    xdog_display = xdog.squeeze(0).squeeze(0).detach().cpu()
    image = image.squeeze(0).squeeze(0).detach().cpu()

    

    axes[idx, i].imshow(canny_display, cmap='gray')
    axes[idx, i].set_title('Canny')
    axes[idx, i].axis('off')

    idx += 1

    axes[idx, i].imshow(xdog_display, cmap='gray')
    axes[idx, i].set_title('XDoG')
    axes[idx, i].axis('off')

    idx += 1

    axes[idx, i].imshow(image, cmap='gray')
    axes[idx, i].axis('off')

    idx += 1



plt.tight_layout()
plt.show()
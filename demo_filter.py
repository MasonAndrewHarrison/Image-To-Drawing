import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
import filter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = transforms.Compose([transforms.ToTensor()])

dataset = ImageFolder(root='dataset_images/', transform=transforms)

fig, axes = plt.subplots(3, 5, figsize=(16, 4))

for i in range(5):

    idx = 0
    tensor,_ = dataset[i+500]
    print(tensor.shape)

    tensor = tensor.requires_grad_(True)

    tensor = tensor.mean(0).unsqueeze(0).unsqueeze(0).to(device)

    tensor = filter.scale(tensor, 3)
    image = tensor.clone().detach()

    xdog1 = filter.ex_difference_of_gaussians(tensor)
    xdog2 = filter.ex_difference_of_gaussians(tensor)

    xdog_display1 = xdog1.squeeze(0).squeeze(0).detach().cpu()
    xdog_display2 = xdog2.squeeze(0).squeeze(0).detach().cpu()
    image = image.squeeze(0).squeeze(0).detach().cpu()
    
    axes[idx, i].imshow(xdog_display1, cmap='gray')
    axes[idx, i].set_title('XDoG1')
    axes[idx, i].axis('off')

    idx += 1

    axes[idx, i].imshow(xdog_display2, cmap='gray')
    axes[idx, i].set_title('XDoG2')
    axes[idx, i].axis('off')

    idx += 1

    axes[idx, i].imshow(image, cmap='gray')
    axes[idx, i].axis('off')

    idx += 1



plt.tight_layout()
plt.show()
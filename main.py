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

tensor,_ = dataset[0]

tensor = tensor.requires_grad_(True)
tensor = tensor.mean(0).unsqueeze(0).unsqueeze(0).to(device) / 256.0

tensor = filter.scale(tensor, 5)
image = tensor.clone().detach()

tensor = filter.variable_gaussian_blur(tensor, size=3, sigma=1.0)
xdog = filter.ex_difference_of_gaussians(tensor)
canny = filter.canny(tensor)



canny_display = canny.squeeze(0).squeeze(0).detach().cpu()
xdog_display = xdog.squeeze(0).squeeze(0).detach().cpu()
image = image.squeeze(0).squeeze(0).detach().cpu()

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].imshow(canny_display, cmap='gray')
axes[0].set_title('Canny')
axes[0].axis('off')

axes[1].imshow(xdog_display, cmap='gray')
axes[1].set_title('XDoG')
axes[1].axis('off')

axes[2].imshow(image, cmap='gray')
axes[2].axis('off')



plt.tight_layout()
plt.show()
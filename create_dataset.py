import kaggle
import torch
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder 
from torchvision import transforms
import os
import shutil


os.makedirs("dataset_images/all", exist_ok=True)

print("Downloading data.")

kaggle.api.authenticate()
kaggle.api.dataset_download_files("prondeau/the-car-connection-picture-dataset", path='temp/all/', unzip=True)

print("Filtering data. May take some time.")

transform = transforms.Compose([transforms.ToTensor(),])
dataset = ImageFolder(root='temp', transform=transform)

true_image_index = 0

for i, (image,_) in enumerate(dataset):

    if i % 3000 == 0:
        print(f"{(i/len(dataset)*100):.2f}% Done")

    if image.shape == (3, 240, 320):

        first_pixel = image[:, 0, 0]
        last_pixel = image[:, 239, 319]
        target = torch.tensor([1.0, 1.0, 1.0])

        if torch.equal(first_pixel, target) and torch.equal(last_pixel, target):

            true_image_index += 1
            save_image(image, f"dataset_images/all/img{true_image_index}.png")

shutil.rmtree("temp/")
print("Finished :)")
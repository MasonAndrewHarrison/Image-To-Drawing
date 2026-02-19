import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
import filter
from draw import Strokes
from model import Model
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
import os
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

height = 240
width  = 320
batch_size = 64
learning_rate = 5e-3
epochs = 1
lines_drawn = 12


transforms = transforms.Compose([transforms.ToTensor()])
dataset = ImageFolder(root='dataset_images/', transform=transforms)

loader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True,
    pin_memory=True,
)

model = Model().to(device)

fixed_image = dataset[0]

scaler = GradScaler(device.__str__())
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

for epoch in range(epochs):
    model.train()

    for i, (image,_) in enumerate(loader):

        strokes = Strokes(height, width, device="cpu")
        images = image.to(device)

        for i in range(lines_drawn):

            output = model(images)
            print(output.requires_grad)
            stroke = output[1, 0:6].cpu()
            stroke[0] = stroke[0]+1 * random.randint(0, width)
            stroke[1] = stroke[1]+1 * random.randint(0, height)
            stroke[2] = stroke[2]+1 * random.randint(0, width)
            stroke[3] = stroke[3]+1 * random.randint(0, height)
            stroke[4] = (stroke[4]/1000)+1 * random.uniform(0.005, 0.02)
            stroke[5] = (stroke[5]/1000)+1 * random.uniform(0.005, 0.02)
            print(stroke.requires_grad)
            strokes.draw(stroke)

        canvas = strokes.canvas()
        print(canvas.shape, type(canvas), type(canvas[0, 0, 0]))
        print(canvas.requires_grad)
        strokes.render()

        break

            
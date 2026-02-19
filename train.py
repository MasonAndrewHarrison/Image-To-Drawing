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
batch_size = 32
learning_rate = 5e-3
epochs = 1
lines_drawn = 10


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

        strokes = Strokes(batch_size, height, width, device=device)
        images = image.to(device)

        for i in range(lines_drawn):

            output = model(images)
            output[:, 0] = output[:, 0]+1 * random.randint(0, width)
            output[:, 1] = output[:, 1]+1 * random.randint(0, height)
            output[:, 2] = output[:, 2]+1 * random.randint(0, width)
            output[:, 3] = output[:, 3]+1 * random.randint(0, height)
            output[:, 4] = (output[:, 4]/1000)+1 * random.uniform(0.005, 0.02)
            output[:, 5] = (output[:, 5]/1000)+1 * random.uniform(0.005, 0.02)
            strokes.draw(output[:, 0:6])

        canvas = strokes.canvas()
        strokes.render()

            
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
import filter
from draw import Strokes
from model import Model, initialize_weights
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
import os
import random

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

height = 240
width  = 320
batch_size = 1
learning_rate = 5e-3
epochs = 1
lines_drawn = 30
prefered_distance = 5
prefered_sigma = 0.005
prefered_radius = 0.01

transforms = transforms.Compose([transforms.ToTensor()])
dataset = ImageFolder(root='dataset_images/', transform=transforms)

loader = DataLoader(
    dataset, 
    batch_size=batch_size,
    #shuffle=True,
    pin_memory=True,
)

model = Model().to(device)

initialize_weights(model)

fixed_image,_ = next(iter(loader))

scaler = GradScaler(device.__str__())
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
criterion = nn.MSELoss()

for epoch in range(epochs): 
    model.train()

    for i, (images,_) in enumerate(loader):

        strokes = Strokes(batch_size, height, width, device=device)

        images = images.to(device)
        images = fixed_image.to(device)
        bw_images = images.mean(1).unsqueeze(1).clone().to(device)
        bw_images = filter.ex_difference_of_gaussians(bw_images).float()

        for j in range(lines_drawn):

            model.zero_grad()
            index_matrix = torch.ones_like(bw_images) * i

            combined_image = torch.cat([
                bw_images, 
                strokes.canvas().detach(),
                index_matrix,
            ], dim=1)

            strokes_copy = strokes.strokes.detach()

            if strokes_copy.shape[1] == 0:
                strokes_copy = torch.ones((1, 1, 5), device=device)

            combined_input = (combined_image, strokes_copy)
            output = model(combined_input)

            strokes.forget_grads()
            strokes.draw(output)

            loss = strokes.loss(
                prefered_distance=prefered_distance,
                prefered_sigma=prefered_sigma,
                prefered_radius=prefered_radius,
            )

            canvas = strokes.canvas(for_model=False)
            image_loss = criterion(canvas, bw_images) * 100
            angle_loss = strokes._angle_loss(-1)
            loss = loss #+ image_loss

            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

            if j % 5 == 0:
                print(f"loss: {loss.item():.4f} Image loss: {image_loss.item():.4f} Angle loss: {angle_loss.item():.4f} Grad norm: {total_norm:.4f}")

            optimizer.step()

        
        if i % 20 == 0:
            #TODO max alpha is too low
            #TODO first image sometimes is just white
            strokes.debug_printout()
            strokes.render(other_image=bw_images)

            
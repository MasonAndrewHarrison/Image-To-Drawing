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
learning_rate = 1e-6
epochs = 1
lines_drawn = 10
prefered_distance = 20
prefered_sigma = 0.001
prefered_radius = 0.001

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
        print(images.shape)
        bw_images = images.mean(1).unsqueeze(1).clone().to(device)
        bw_images = filter.ex_difference_of_gaussians(bw_images).float()

        for _ in range(lines_drawn):

            model.zero_grad()
            index_matrix = torch.ones_like(bw_images) * i

            combined_input = torch.cat([
                images, 
                strokes.canvas().detach(),
                index_matrix,
            ], dim=1)

            output = model(combined_input)

            strokes.forget_grads()
            strokes.draw(output)

            #TODO gradient explosion issue
            loss = strokes.loss(
                prefered_distance=prefered_distance,
                prefered_sigma=prefered_sigma,
                prefered_radius=prefered_radius,
            ) * 0.1

            avg_dist = strokes.get_distance(-1).sum()/batch_size
            canvas = strokes.canvas()
            image_loss = criterion(canvas, bw_images) * 40
            loss = loss + image_loss

            
            
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            print(f"loss: {loss:.4f} Image loss: {image_loss:.4f} grad norm: {total_norm:.4f}")
            optimizer.step()

        
        if i % 27 == 0:
            angle = strokes.get_line_angle(-1)
            strokes.render(other_image=bw_images)

            
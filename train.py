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
batch_size = 16
learning_rate = 5e-5
epochs = 1
lines_drawn = 12
prefered_line_dist = 25
prefered_sigma = 0.005
prefered_radius = 0.005

transforms = transforms.Compose([transforms.ToTensor()])
dataset = ImageFolder(root='dataset_images/', transform=transforms)

loader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True,
    pin_memory=True,
)

model = Model().to(device)

initialize_weights(model)

fixed_image = dataset[0]

prefered_ld = torch.ones(16, device=device) * prefered_line_dist
prefered_sig = torch.ones(16, device=device) * prefered_sigma
prefered_r = torch.ones(16, device=device) * prefered_radius

scaler = GradScaler(device.__str__())
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
criterion = nn.MSELoss()
scaler = GradScaler(device.__str__())

for epoch in range(epochs):
    model.train()

    for i, (image,_) in enumerate(loader):

        strokes = Strokes(batch_size, height, width, device=device)
        images = image.to(device)

        for _ in range(lines_drawn):

            model.zero_grad()

            output = model(images)

            '''output[:, 0] = output[:, 0]+random.randint(-50, 50)
            output[:, 1] = output[:, 1]+random.randint(-50, 50)
            output[:, 2] = output[:, 2]+random.randint(-50, 50)
            output[:, 3] = output[:, 3]+random.randint(-50, 50)
            output[:, 4] = (output[:, 4]/1000)+1 * random.uniform(0.003, 0.01)
            output[:, 5] = (output[:, 5]/1000)+1 * random.uniform(0.003, 0.01)
            output[:, 6] = output[:, 6]+1 * random.randint(0, 1)'''

            '''strokes.draw(output)

            line_loss1 = strokes.line_loss(prefered_ld, index=-2) 
            line_loss2 = strokes.line_loss(prefered_ld, index=-1) 

            loss = (line_loss1 + line_loss2)/2
        
            print("loss: ", loss)'''

            x_dist = output[:, 2] - output[:, 0]
            y_dist = output[:, 3] - output[:, 1]
            distance = torch.sqrt(x_dist**2 + y_dist**2 + 1e-8)

            loss_d = criterion(distance, prefered_ld)
            loss_sig = criterion(output[:, 4], prefered_sig)
            loss_r = criterion(output[:, 5], prefered_r)

            loss = (loss_d + loss_sig + loss_r)/3


            print(f"loss: {loss}")
            
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                print("output sample:", output[0])
                strokes.draw(output.detach())




        canvas = strokes.canvas()
        strokes.render()

            
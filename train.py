import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
import filter
from draw import Stroke
from model import Model, initialize_weights
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from differentiable_rasterizer import image_to_sdf
import os
from utils import *
import random
import yaml

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

height = 240
width  = 320
batch_size = 1
learning_rate = 5e-3
epochs = 1
lines_drawn = config["training"]["lines_drawn"]


transforms = transforms.Compose([transforms.ToTensor()])
dataset = ImageFolder(root='dataset_images/', transform=transforms)

loader = DataLoader(
    dataset, 
    batch_size=batch_size,
    #shuffle=True,
    pin_memory=True,
)

model = Model(features=config["model"]["features"]).to(device)

initialize_weights(model)

fixed_image,_ = next(iter(loader))

scaler = GradScaler(device.__str__())
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
criterion = nn.MSELoss()

#TODO change MSE to SSIM
#TODO rnn lstm

for epoch in range(epochs): 
    model.train()

    for i, (images,_) in enumerate(loader):

        strokes = Stroke(batch_size, height, width, device=device)

        #images = images.to(device)
        images = fixed_image.to(device)
        bw_images = images.mean(1).unsqueeze(1).clone().to(device)
        exdog_image = filter.ex_difference_of_gaussians(bw_images).float()
        sdf = image_to_sdf(exdog_image)

        for j in range(lines_drawn):

            model.zero_grad(set_to_none=True)
            strokes.forget_graph()

            strokes_copy = strokes.get_strokes(use_empty_buffer=True, new_graph=True)
            output = model(image=exdog_image, strokes=strokes_copy)
            strokes.draw(output)

            loss = total_loss(strokes, sdf)
            loss.backward()
            debug_loss(strokes, sdf, model.parameters()) if j % 5 == 0 else None
            
            optimizer.step()

        if i % 5 == 0:
            strokes.debug_printout()
            strokes.render(other_image=sdf)

            
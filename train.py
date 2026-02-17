import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from differentiable_rasterizer import render_lines


strokes = torch.tensor([
    [0.1, 0.1, 0.9, 0.9],  
    [0.5, 0.0, 0.5, 1.0],  
])

canvas = render_lines(strokes, height=100, width=100)
canvas = canvas.squeeze(0).detach().cpu()


plt.imshow(canvas, cmap='gray')
plt.show()
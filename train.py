import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from draw import Strokes


strokes = torch.tensor([
    [0, 0, 0, 0],  
    [0.5, 0.5, 0.5, 1.0],  
])

strokes = Strokes(240, 320)
strokes.draw(50, 50, 200, 200)
strokes.draw(50, 200, 200, 50)

strokes.render()
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import filter
from scipy.ndimage import distance_transform_edt
import numpy as np


def point_from_pixel(image_sdf: torch.Tensor, position: tuple[torch.Tensor, torch.Tensor]):

    x, y = position

    x_floor = torch.floor(x)
    x0 = x_floor.long()
    x1 = x0 + 1
    x_weight = x - x_floor

    y_floor = torch.floor(y)
    y0 = y_floor.long()
    y1 = y0 + 1
    y_weight = y - y_floor

    q00 = image_sdf[y0, x0]
    q01 = image_sdf[y1, x0]
    q10 = image_sdf[y0, x1]
    q11 = image_sdf[y1, x1]

    distance = (q00 * (1 - x_weight) * (1 - y_weight) +
                q10 * x_weight       * (1 - y_weight) +
                q01 * (1 - x_weight) * y_weight       +
                q11 * x_weight       * y_weight)

    return distance
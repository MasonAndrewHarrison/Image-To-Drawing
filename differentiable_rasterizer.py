import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def render_lines(strokes, height, width):

    device = strokes.device

    canvas = torch.zeros(1, height, width).to(device)

    pixel_x = torch.linspace(0, 1, width).view(1, -1).expand(height, width).to(device)
    pixel_y = torch.linspace(0, 1, height).view(-1, 1).expand(height, width).to(device)

    for stroke in strokes:

        x1, y1, x2, y2, sigma, radius = stroke

        # AB→ = B - A
        vector_ab_x = x2 - x1
        vector_ab_y = y2 - y1
        vector_ab_length_squared = vector_ab_x**2 + vector_ab_y**2 + 1e-8

        # AP→ = P - A
        vector_ap_x = (pixel_x - x1)
        vector_ap_y = (pixel_y - y1)

        # t = (AP→ · AB→) / (||AB→||²) 
        t = (vector_ap_x * vector_ab_x + vector_ap_y * vector_ab_y) / vector_ab_length_squared
        
        # This keeps the t value in between A and B
        t = t.clamp(0, 1)

        # AT→ = A + t * AB→
        vector_at_x = x1 + t * vector_ab_x
        vector_at_y = y1 + t * vector_ab_y

        # distance = √( (P_x - AT→_x)² + (P_y - AT→_y)² )
        distance = torch.sqrt((pixel_x - vector_at_x)**2 + (pixel_y - vector_at_y)**2 + 1e-8)

        # Sign Distance Function
        sdf = distance - radius
        sdf = sdf.clamp(min=0)

        # g(sdf) = e^( -sdf² / 2σ²)
        gaussian_sdf = torch.exp(-sdf**2 / (2 * sigma**2))

        canvas = torch.clamp(canvas + gaussian_sdf, 0, 1)


    return 1 - canvas



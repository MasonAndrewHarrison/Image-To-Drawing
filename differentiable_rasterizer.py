import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def render_lines(strokes, height, width, brush_thickness = 0.01):

    canvas = torch.zeros(1, height, width)
    sigma = brush_thickness

    pixel_x = torch.linspace(0, 1, width).view(1, -1).expand(height, width)  
    pixel_y = torch.linspace(0, 1, height).view(-1, 1).expand(height, width)  

    for stroke in strokes:

        x1, y1, x2, y2 = stroke

        # AB→ = B - A
        vector_ab_x = x2 - x1
        vector_ab_y = y2 - y1
        vector_ab_length_squared = vector_ab_x**2 + vector_ab_y**2 + 1e-8

        # AP→ = P - A
        vector_ap_x = (pixel_x - x1)
        vector_ap_y = (pixel_y - y1)

        # t = (AP→ · AB→) / (||AB→||²) 
        t = (vector_ap_x * vector_ab_x + vector_ap_y * vector_ab_y) / vector_ab_length_squared
        t = t.clamp(0, 1)

        # AT→ = A + t * AB→
        vector_at_x = x1 + t * vector_ab_x
        vector_at_y = y1 + t * vector_ab_y

        # distance = √( (P_x - AT→_x)² + (P_y - AT→_y)² )
        distance = torch.sqrt((pixel_x - vector_at_x)**2 + (pixel_y - vector_at_y)**2 + 1e-8)

        # brightness = e^( -distance² / 2σ²)
        brightness = torch.exp(-distance**2 / (2 * sigma**2))

        canvas = torch.clamp(canvas + brightness, 0, 1)

    return canvas 



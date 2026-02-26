import torch
import torch.nn as nn
import matplotlib.pyplot as plt

@torch.compile
def render_lines_sdf(strokes: torch.Tensor, height: int, width: int, for_model: bool) -> torch.Tensor:

    device = strokes.device

    canvas = torch.zeros(1, height, width).to(device)

    pixel_x = torch.linspace(0, 1, width, device=device).view(1, 1, -1)   # (1, 1, W)
    pixel_y = torch.linspace(0, 1, height, device=device).view(1, -1, 1)  # (1, H, 1)

    x1 = strokes[:-1, 0].view(-1, 1, 1)
    y1 = strokes[:-1, 1].view(-1, 1, 1)
    x2 = strokes[1:, 0].view(-1, 1, 1)
    y2 = strokes[1:, 1].view(-1, 1, 1)
    
    sigma = strokes[1:, 2].view(-1, 1, 1)
    radius = strokes[1:, 3].view(-1, 1, 1)
    opacity = strokes[1:, 4].view(-1, 1, 1)

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

    if not for_model:

        # Sign Distance Function
        sdf = distance - radius
        sdf = sdf.clamp(min=0)

        # g(sdf) = e^( -sdf² / 2σ²)
        gaussian_sdf = torch.exp(-sdf**2 / (2 * sigma**2 + 1e-8)).clamp(min=1e-6)
        gaussian_sdf = gaussian_sdf * opacity
        canvas = gaussian_sdf
        #1 - canvas
    
    else:
        # Sign Distance Function
        sdf = distance - radius
        sdf = sdf.clamp(min=0)

        canvas = sdf

    canvas = canvas.sum(dim=0)
    canvas = canvas.unsqueeze(0)

    if not for_model:
        canvas = canvas.clamp(0, 1)
        canvas = 1 - canvas


    return canvas

def render_lines_sdf_batched(strokes: torch.Tensor, height: int, width: int, for_model: bool) -> torch.Tensor:

    all_canvas = torch.func.vmap(
        lambda stroke: render_lines_sdf(stroke, height, width, for_model)
    )(strokes)

    return all_canvas

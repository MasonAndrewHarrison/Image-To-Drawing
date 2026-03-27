import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from differentiable_rasterizer import render_lines_sdf, render_sdf_batched, image_to_sdf
from utils import points_from_sdf, nested_smoothstep
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import filter
import yaml
import random
import time

class Stroke():

    def __init__(self, batch_size, height, width, device):

        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.device = device

        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        line_preference = config["prefered_line"]

        self.prefered_distance = line_preference["distance"]
        self.prefered_sigma = float(line_preference["sigma"])
        self.prefered_radius = float(line_preference["radius"])

        self.strokes = torch.zeros(batch_size, 0, 4).to(device)

    def get_strokes(self, use_empty_buffer: bool = False, new_graph: bool = True):

        new_stroke = self.strokes.clone()

        scale = new_stroke.new_tensor([self.width, self.height, 1, 1])
        new_stroke = new_stroke / scale

        if use_empty_buffer and new_stroke.shape[1] == 0:
            new_stroke = torch.ones((1, 1, 4), device=self.device)

        new_stroke = new_stroke.detach() if new_graph else new_stroke

        return new_stroke


    def draw(self, new_stroke):

        new_stroke = new_stroke.unsqueeze(1)
        self.strokes = torch.concat([self.strokes, new_stroke], dim=1)
        

    def point_from_sdf(self, sdf, stroke_index):

        point = self.strokes[:, stroke_index, 0:2].unsqueeze(1)
        distance = points_from_sdf(image_sdf=sdf, positions=point)

        return distance


    def all_point_from_sdf(self, sdf):

        positions = self.strokes[:, :, 0:2]
        return points_from_sdf(sdf, positions)


    def canvas(self, raw_sdf: bool = False):

        strokes = self.get_strokes()

        return render_sdf_batched(
            strokes=strokes[:, :, :], 
            height=self.height, 
            width=self.width, 
            raw_sdf=raw_sdf,
        )

    def render(self, other_image=None):

        if other_image is None:

            canvas = self.canvas()
            canvas = canvas[-1, :, :]
            canvas = canvas.squeeze(0).detach().cpu()

            plt.imshow(canvas, cmap='grey')
            plt.show()

        else:

            canvas = self.canvas(raw_sdf=True)
            canvas = canvas[-1, :, :].squeeze(0)
            other_image = other_image[-1, :, : ,:].squeeze(0)

            canvas_drawing = self.canvas()
            canvas_drawing = canvas_drawing[-1, :, :]
            canvas_drawing = canvas_drawing.squeeze(0).detach().cpu()

            canvas = canvas.detach().cpu()
            other_image = other_image.detach().cpu()

            fig, axes = plt.subplots(1, 3, figsize=(30, 10))

            axes[0].imshow(canvas, cmap='grey')
            axes[0].set_title("AI Raw SDF")
            axes[0].axis('off')

            axes[1].imshow(canvas_drawing, cmap='grey', vmin=0, vmax=1)
            axes[1].set_title("AI Drawing")
            axes[1].axis('off')

            axes[2].imshow(other_image, cmap='grey')
            axes[2].set_title("Image with Filter")
            axes[2].axis('off')

            plt.tight_layout()
            plt.show()


    def __str__(self):

        return self.strokes.__str__()


    def get_distance(self, index):

        stroke = self.strokes[:, index, :].clone()
        prior_stroke = self.strokes[:, index-1, :].clone()
        x_dist = prior_stroke[:, 0] - stroke[:,0]
        y_dist = prior_stroke[:, 1] - stroke[:,1]

        distance = torch.sqrt(x_dist**2 + y_dist**2 + 1e-8) 

        return distance   

    def avg_distance(self):

        stroke = self.strokes[:, 1:, :].clone()
        prior_stroke = self.strokes[:, :-1, :].clone()
        x_dist = prior_stroke[:, :, 0] - stroke[:, :,0]
        y_dist = prior_stroke[:, :, 1] - stroke[:, :,1]

        distance = torch.sqrt(x_dist**2 + y_dist**2 + 1e-8) 

        return distance.mean()    


    def _line_loss(self, prefered_ld, index):

        criterion = nn.MSELoss()

        if self.strokes.shape[1] <= index.__abs__():
            
            zero_loss = self.strokes.clone().sum() * 0
            return zero_loss

        distance = self.get_distance(index)
        loss = criterion(distance, prefered_ld)

        return loss


    def get_simga(self, index):

        return self.strokes[:, index, 2]


    def _sigma_loss(self, prefered_sigma, index):

        criterion = nn.MSELoss()
        simga = self.get_simga(index)

        return criterion(simga, prefered_sigma)


    def get_radius(self, index):

        return self.strokes[:, index, 3]

    def _radius_loss(self, prefered_radius, index):

        criterion = nn.MSELoss()
        radius = self.get_radius(index)

        return criterion(radius, prefered_radius)


    def _get_angle(self, origin_x, origin_y, xa, ya, xb, yb):
        """
        Finds the angle of the vector A and vector B and normalized
        with the origin point
        θ = arccos( (A→ · B→) / ||A→||*||B→|| ) / π, θ ∈ [0, 1] 
        """

        xa = xa - origin_x
        xb = xb - origin_x
        ya = ya - origin_y
        yb = yb - origin_y

        dot_product = xa * xb + ya * yb
        length_a = torch.sqrt(xa**2 + ya**2 + 1e-8).clamp(min=1e-6)
        length_b = torch.sqrt(xb**2 + yb**2 + 1e-8).clamp(min=1e-6)

        valid = (length_a > 1e-3) & (length_b > 1e-3)

        theda_cos = dot_product / (length_a * length_b)
        theda_cos = torch.clamp(theda_cos, -1 + 1e-6, 1 - 1e-6)

        theda = torch.acos(theda_cos)
        theda = theda * valid.float()

        return theda / torch.pi


    def get_angle(self, index):

        xa = self.strokes[:, index-2, 0].clone()
        ya = self.strokes[:, index-2, 1].clone()
        xb = self.strokes[:, index-1, 0].clone()
        yb = self.strokes[:, index-1, 1].clone()
        xc = self.strokes[:, index, 0].clone()
        yc = self.strokes[:, index, 1].clone()

        angle = self._get_angle(
            xa=xa,
            ya=ya,
            origin_x=xb,
            origin_y=yb,
            xb=xc,
            yb=yc,
        )

        return angle


    def _angle_loss(self, index):

        if self.strokes.shape[1] <= index.__abs__()+1:
            
            zero_loss = self.strokes.clone().sum() * 0
            return zero_loss

        criterion = nn.BCELoss()

        angle = self.get_angle(index)
        weight_angle = nested_smoothstep(angle, 1)

        angle_loss = criterion(
            weight_angle, 
            torch.ones_like(weight_angle)*0.95
        )

        return angle_loss


    def loss(self):

        prefered_distance_vec = torch.ones(
            self.batch_size, 
            device=self.device
        ) * self.prefered_distance

        prefered_sigma_vec = torch.ones(
            self.batch_size, 
            device=self.device
        ) * self.prefered_sigma

        prefered_radius_vec = torch.ones(
            self.batch_size, 
            device=self.device
        ) * self.prefered_radius

        loss = torch.stack([
            self._line_loss(prefered_distance_vec, index=-1),
            self._sigma_loss(prefered_sigma_vec, index=-1),
            self._radius_loss(prefered_radius_vec, index=-1),
            self._angle_loss(index=-1),
        ])

        return loss.sum()


    def forget_graph(self):

        self.strokes = self.strokes.detach()

    def debug_printout(self):

        print(f"Largest x: {self.strokes[-1, :, 0].max():.4f} < {self.width}\n"
            f"Smallest x: {self.strokes[-1, :, 0].min():.4f} > {0}\n"
            f"Largest y: {self.strokes[-1, :, 1].max():.4f} < {self.height}\n"
            f"Smallest y: {self.strokes[-1, :, 1].min():.4f} > {0}\n"
            f"Largest sigma: {self.strokes[-1, :, 2].max():.4f} ~= {self.prefered_sigma}\n"
            f"Smallest sigma: {self.strokes[-1, :, 2].min():.4f} ~= {self.prefered_sigma}\n"
            f"Largest radius: {self.strokes[-1, :, 3].max():.4f} ~= {self.prefered_radius}\n"
            f"Smallest radius: {self.strokes[-1, :, 3].min():.4f} ~= {self.prefered_radius}\n"
            f"Average Distance: {self.avg_distance().item():.4f} ~= {self.prefered_distance}\n"
        )
        
    def shape(self):
        
        return self.strokes.shape


if __name__ == "__main__":

    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    strokes = Stroke(64, 300, 350, device="cuda")

    stroke = torch.zeros(64, 4).to("cuda")
    stroke[:, 0] = 50
    stroke[:, 1] = 50
    stroke[:, 2] = 0.0006
    stroke[:, 3] = 0.01
    stroke.requires_grad_(True)

    strokes.draw(stroke)

    stroke = torch.zeros(64, 4).to("cuda")
    stroke[:, 0] = 200
    stroke[:, 1] = 200
    stroke[:, 2] = 0.016
    stroke[:, 3] = 0.016
    stroke.requires_grad_(True)

    strokes.draw(stroke)

    stroke = torch.zeros(64, 4).to("cuda")
    stroke[:, 0] = 50
    stroke[:, 1] = 200
    stroke[:, 2] = 0.003
    stroke[:, 3] = 0.003
    stroke.requires_grad_(True)

    strokes.draw(stroke)

    stroke = torch.zeros(64, 4).to("cuda")
    stroke[:, 0] = 240
    stroke[:, 1] = 240
    stroke[:, 2] = 0.006
    stroke[:, 3] = 0.006
    stroke.requires_grad_(True)

    strokes.draw(stroke)

    stroke = torch.zeros(64, 4).to("cuda")
    stroke[:, 0] = 290
    stroke[:, 1] = 0
    stroke[:, 2] = 0.006
    stroke[:, 3] = 0.006
    stroke.requires_grad_(True)

    strokes.draw(stroke)


    dummy_strokes = torch.rand(10, 4).to('cuda')
    render_lines_sdf(dummy_strokes, 300, 300, raw_sdf=True)
    torch.cuda.synchronize()

    start = time.time()

    canvas = strokes.canvas()

    prefered_distance = 25
    prefered_sigma = 0.005
    prefered_radius = 0.005
    
    loss = strokes.loss(
        prefered_distance=prefered_distance,
        prefered_sigma=prefered_sigma,
        prefered_radius=prefered_radius,
    )

    end = time.time()

    print(f"Time to generate canvas: {(end - start):.4f} seconds.")
    print(f"Check does grads work with 'draw.py' pipe line: {canvas.requires_grad}, {loss.requires_grad}")
    print(f"Check does cuda work with 'draw.py' pipe line: {canvas.device.__str__()[:-2] == 'cuda'} || device used: {canvas.device}")

    print("Visual Test: ")

    strokes.render(other_image=canvas)

    print("Exit")
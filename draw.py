import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from differentiable_rasterizer import render_sdf, image_to_sdf
from utils import points_from_sdf, nested_smoothstep, points_from_image, faster_point_from_image
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import filter
import yaml
import random
import time


#TODO rewrite everything in taichi, c or c++


class Drawer():

    def __init__(self, edge_image, interpolation_mode: str = 'bicubic'):

        self.edge_image = edge_image
        self.device = edge_image.device
        self.height = edge_image.shape[0]
        self.width = edge_image.shape[1]
        self.interpolation_mode = interpolation_mode

        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        line_preference = config["prefered_line"]

        self.prefered_distance = line_preference["distance"]
        self.prefered_sigma = float(line_preference["sigma"])
        self.prefered_radius = float(line_preference["radius"])

        self.stroke = torch.zeros(0, 4, device=self.device)

    def trace_edge(self) -> bool:

        points_in_stroke = self.stroke.shape[0]

        if points_in_stroke < 1:
            new_point = torch.tensor(
                [
                    self.width / 2, 
                    self.height / 2, 
                    self.prefered_sigma, 
                    self.prefered_radius,
                ]
                , device=self.device
            ).unsqueeze(0)
            self.draw(new_point)
        else:
            last_point = self.stroke[-1, :].unsqueeze(0)
            self.draw(last_point)

        point_dist, point = self.all_points_from_image()

        raw_grad = torch.autograd.grad(point_dist.sum(), point)[0]
        direction = raw_grad / (raw_grad.norm(dim=-1, keepdim=True) + 1e-8)

        line_distance = self.prefered_distance
        if points_in_stroke < 1:
            line_distance = point_dist[-1]
        scaled_grad = direction * line_distance

        with torch.no_grad():
            point -= scaled_grad

        self.stroke[-1, 0:2] = point[-1, :]
        self.update_edge_image(new_point=point[-1, :].detach())
        is_complete = False

        return is_complete

    def update_edge_image(self, new_point):

        H, W = self.edge_image.shape
        x, y = torch.unbind(new_point, dim=0)

        standard_devation = 1e-6
        brownian_motion = torch.randn(2, device=self.device) * standard_devation
        x_epsilon = brownian_motion[0]
        y_epsilon = brownian_motion[1]

        x, y = x + x_epsilon, y + y_epsilon

        pixel_y, pixel_x = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing='ij'
        )
        sdf = torch.sqrt((pixel_x - x)**2 + (pixel_y - y)**2 + 1e-8)
        mask = sdf > self.prefered_distance

        self.edge_image = torch.where(mask, self.edge_image, 1)

    def get_stroke(self, use_empty_buffer: bool = False, new_graph: bool = True):

        new_stroke = self.stroke.clone()

        scale = new_stroke.new_tensor([self.width, self.height, 1, 1])
        new_stroke = new_stroke / scale

        if use_empty_buffer and new_stroke.shape[1] == 0:
            new_stroke = torch.ones((1, 4), device=self.device)

        new_stroke = new_stroke.detach() if new_graph else new_stroke

        return new_stroke


    def get_point(self, index, new_graph: bool = True):

        point = self.stroke[index, :].unsqueeze(0).clone()
        point = point.detach() if new_graph else point

        return point

    def reverse_order(self):

        self.stroke = torch.flip(self.stroke, dims=[0])


    def draw(self, new_stroke):

        self.stroke = torch.concat([self.stroke, new_stroke], dim=0)
        

    def point_from_sdf(self, sdf, stroke_index):

        point = self.stroke[stroke_index, 0:2].unsqueeze(0).clone()
        point = point.detach().requires_grad_(True)
        distance = points_from_sdf(image_sdf=sdf, positions=point)

        return (distance, point)


    def all_points_from_sdf(self, sdf):

        point = self.stroke[:, 0:2].clone()
        point = point.detach().requires_grad_(True)
        distance = points_from_sdf(image_sdf=sdf, positions=point)

        return (distance, point)

    
    def point_from_image(self, index: int = -1):

        point = self.stroke[index, 0:2].clone()
        point = point.detach().requires_grad_(True)
        distance = faster_point_from_image(
            edge_image=self.edge_image, 
            position=point,
        )

        return (distance, point)

    
    def all_points_from_image(self):

        point = self.stroke[:, 0:2].clone()
        point = point.detach().requires_grad_(True)
        distance = points_from_image(
            edge_image=self.edge_image, 
            positions=point,
            interpolation_mode=self.interpolation_mode
        )

        return (distance, point)


    def canvas(self, raw_sdf: bool = False):

        stroke = self.get_stroke()

        return render_sdf(
            stroke=stroke, 
            height=self.height, 
            width=self.width, 
            raw_sdf=raw_sdf,
        )

    def render(self, other_image=None):

        other_image=self.edge_image

        if other_image is None:

            canvas = self.canvas()
            canvas = canvas.squeeze(0).squeeze(0).detach().cpu()

            plt.imshow(canvas, cmap='grey')
            plt.show()

        else:

            canvas = self.canvas(raw_sdf=True)
            canvas = canvas.squeeze(0)

            canvas_drawing = self.canvas()
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

        return self.stroke.__str__()


    def get_distance(self, index):

        stroke = self.stroke[index, :].clone()
        prior_stroke = self.stroke[index-1, :].clone()
        x_dist = prior_stroke[:, 0] - stroke[:,0]
        y_dist = prior_stroke[:, 1] - stroke[:,1]

        distance = torch.sqrt(x_dist**2 + y_dist**2 + 1e-8) 

        return distance   

    def avg_distance(self):

        stroke = self.stroke[1:, :].clone()
        prior_stroke = self.stroke[:-1, :].clone()
        x_dist = prior_stroke[:, 0] - stroke[:, :,0]
        y_dist = prior_stroke[:, 1] - stroke[:, :,1]

        distance = torch.sqrt(x_dist**2 + y_dist**2 + 1e-8) 

        return distance.mean()    


    def get_simga(self, index):

        return self.stroke[index, 2]


    def get_radius(self, index):

        return self.stroke[index, 3]


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

        xa = self.stroke[index-2, 0].clone()
        ya = self.stroke[index-2, 1].clone()
        xb = self.stroke[index-1, 0].clone()
        yb = self.stroke[index-1, 1].clone()
        xc = self.stroke[index, 0].clone()
        yc = self.stroke[index, 1].clone()

        angle = self._get_angle(
            xa=xa,
            ya=ya,
            origin_x=xb,
            origin_y=yb,
            xb=xc,
            yb=yc,
        )

        return angle


    def forget_graph(self):

        self.stroke = self.stroke.detach()

    def debug_printout(self):

        print(f"Largest x: {self.stroke[:, 0].max():.4f} < {self.width}\n"
            f"Smallest x: {self.stroke[:, 0].min():.4f} > {0}\n"
            f"Largest y: {self.stroke[:, 1].max():.4f} < {self.height}\n"
            f"Smallest y: {self.stroke[:, 1].min():.4f} > {0}\n"
            f"Largest sigma: {self.stroke[:, 2].max():.4f} ~= {self.prefered_sigma}\n"
            f"Smallest sigma: {self.stroke[:, 2].min():.4f} ~= {self.prefered_sigma}\n"
            f"Largest radius: {self.stroke[:, 3].max():.4f} ~= {self.prefered_radius}\n"
            f"Smallest radius: {self.stroke[:, 3].min():.4f} ~= {self.prefered_radius}\n"
            f"Average Distance: {self.avg_distance().item():.4f} ~= {self.prefered_distance}\n"
        )
        
    def shape(self):
        
        return self.stroke.shape


if __name__ == "__main__":

    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms = transforms.Compose([transforms.ToTensor()])
    dataset = ImageFolder(root='dataset_images/', transform=transforms)

    image,_ = dataset[0]
    _, height, width = image.shape
    image = image.mean(0).unsqueeze(0).unsqueeze(0).to(device)
    canny = filter.canny(image).squeeze(0).squeeze(0)
    sdf = image_to_sdf(canny).squeeze(0)

    drawer = Drawer(edge_image=canny)
    
    for _ in range(300):
        drawer.trace_edge()
        
    drawer.render(other_image=canny)


    #distance, point = drawer.all_points_from_sdf(sdf)
    
    
    
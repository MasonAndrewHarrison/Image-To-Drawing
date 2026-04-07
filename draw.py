import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from differentiable_rasterizer import render_sdf, image_to_sdf
from utils import *
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import filter
import yaml
import random
import time

#TODO rewrite everything in taichi, c or c++


class Canvas():

    def __init__(self, image):

        _, height, width = image.shape
        image = image.mean(0).unsqueeze(0).unsqueeze(0).to(device)
        
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        filter_config = config["filter"]

        self.size_threshold = filter_config["size_threshold"]
        self.density_threshold = float(filter_config["density_threshold"])
        self.scaler = filter_config["interpolation_scaler"]

        ex_dog_config = filter_config["ex_dog"]
        ex_dog = filter.ex_difference_of_gaussians(
            image=image,
            tau=float(ex_dog_config["tau"]),
            epsilon=float(ex_dog_config["epsilon"]),
            phi=float(ex_dog_config["phi"]),
            sigma=float(ex_dog_config["sigma"]),
            threshold=float(ex_dog_config["threshold"]),
            k=float(ex_dog_config["k"]),
        )
        
        ex_dog = F.interpolate(
            ex_dog.float(),
            size=[height*self.scaler, width*self.scaler],
            mode="bicubic",
            align_corners=False
        ).squeeze(0).squeeze(0)
        binary = (ex_dog < 0.5).to(torch.uint8)

        data = separate_pixels(binary)
        data = filter_noise(
            *data, 
            size_threshold=self.size_threshold
        )
        num_labels, labels,_ = density_filter(
            *data, 
            density_threshold=self.density_threshold
        )

        self.layers = separate_to_layers(num_labels, labels)



class Drawer():

    def __init__(self, edge_image, interpolation_mode: str = 'bicubic'):
        
        self.device = edge_image.device
        self.height = edge_image.shape[0]
        self.width = edge_image.shape[1]

        binary = edge_image < 0.5
        self.negative_sdf = image_to_negative_sdf(binary.unsqueeze(0)).squeeze(0)
        self.interpolation_mode = interpolation_mode

        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        line_preference = config["prefered_line"]

        self.search_r_min = line_preference["search_radius_min"]
        self.search_r_max = line_preference["search_radius_max"]
        self.prefered_sigma = float(line_preference["sigma"])
        self.prefered_radius = float(line_preference["radius"])
        self.stroke = torch.zeros(0, 4, device=self.device)

    def trace_edge(self):

        points_in_stroke = self.stroke.shape[0]
        is_complete = False
        first_stroke = points_in_stroke < 1

        if first_stroke:
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


        search_radius = self.search_r_max if not first_stroke else -1
        print(search_radius)

        last_point = self.stroke[-1, 0:2]
        vector = vec_to_lowest_point(
            self.negative_sdf, 
            position=last_point, 
            magnatude=-1,
            search_radius=search_radius
        )
        if vector[0] == 0 and vector[1] == 0:
            is_complete = True
        else:
            new_point = last_point + vector
            self.update_negative_sdf(new_point)
            self.stroke[-1, 0:2] = new_point
            

        
        return is_complete
        

    def update_negative_sdf(self, new_point):

        H, W = self.negative_sdf.shape
        x, y = torch.unbind(new_point, dim=0)
        radius = self.search_r_min
        padding = self.search_r_max + 1

        print(x,y, H, W)    
        x = torch.clip(x, padding, W - padding)
        y = torch.clip(y, padding, H - padding)
        print(x,y, H, W)

        x_start = torch.clip(x - radius, 0, W).__int__()
        x_end = torch.clip(x + radius, 0, W).__int__()
        y_start = torch.clip(y - radius, 0, H).__int__()
        y_end = torch.clip(y + radius, 0, H).__int__()

        mask = ~create_circle_mask(radius, device=self.device)

        print(mask.shape, self.negative_sdf[y_start:y_end, x_start:x_end].shape)
        updated_patch = torch.where(mask, self.negative_sdf[y_start:y_end, x_start:x_end], 0)
        self.negative_sdf[y_start:y_end, x_start:x_end] = updated_patch

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


    def canvas(self, raw_sdf: bool = False):

        stroke = self.get_stroke()

        return render_sdf(
            stroke=stroke, 
            height=self.height, 
            width=self.width, 
            raw_sdf=raw_sdf,
        )

    def render(self, other_image=None):

        other_image=self.negative_sdf

        if other_image is None:

            canvas = self.canvas()
            canvas = canvas.squeeze(0).squeeze(0).detach().cpu()

            plt.imshow(canvas, cmap='grey')
            plt.show()

        else:

            canvas_drawing = self.canvas()
            canvas_drawing = canvas_drawing.squeeze(0).detach().cpu()

            other_image = other_image.detach().cpu()
            fig, axes = plt.subplots(1, 3, figsize=(30, 10))

            canvas = ~( (other_image<0) + (canvas_drawing<0.5) )
            canvas = canvas.detach().cpu()

            axes[0].imshow(canvas, cmap='grey')
            axes[0].scatter(self.stroke[-1, 0].detach().cpu(), self.stroke[-1, 1].detach().cpu())
            axes[1].imshow(canvas_drawing, cmap='grey')
            axes[2].imshow(other_image, cmap='grey')

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
        
    def shape(self):
        
        return self.stroke.shape


if __name__ == "__main__":


    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms = transforms.Compose([transforms.ToTensor()])
    dataset = ImageFolder(root='dataset_images/', transform=transforms)

    image,_ = dataset[0]
    canny = Canvas(image).layers[1, :, :]
    canny = torch.tensor(canny, device=device).squeeze(0)

    drawer = Drawer(edge_image=canny)
    
    for _ in range(200):
        complete = drawer.trace_edge()
        if complete:
            break
        
        
    drawer.render(other_image=canny)
    
    
    
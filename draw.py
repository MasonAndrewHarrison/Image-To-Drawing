import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from differentiable_rasterizer import render_lines_sdf, render_lines_sdf_batched
import random
import time

class Strokes():

    def __init__(self, batch_size, height, width, device):

        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.device = device

        self.strokes = torch.zeros(batch_size, 0, 5).to(device)

    def get_strokes(self):

        new_stroke = self.strokes.clone()

        scale = new_stroke.new_tensor([self.width, self.height, 1, 1, 1])
        new_stroke = new_stroke / scale

        return new_stroke


    def draw(self, new_stroke):

        new_stroke = new_stroke.unsqueeze(1)
        self.strokes = torch.concat([self.strokes, new_stroke], dim=1)

    def canvas(self, for_model: bool = True):

        strokes = self.get_strokes()

        return render_lines_sdf_batched(
            strokes=strokes[:, :, :], 
            height=self.height, 
            width=self.width, 
            for_model=for_model,
        )

    def render(self, other_image=None):

        if other_image is None:

            canvas = self.canvas()
            canvas = canvas[-1, :, :]
            canvas = canvas.squeeze(0).detach().cpu()

            plt.imshow(canvas, cmap='grey')
            plt.show()

        else:

            canvas = self.canvas()
            canvas = canvas[-1, :, :].squeeze(0)
            other_image = other_image[-1, :, : ,:].squeeze(0)

            canvas_drawing = self.canvas(for_model=False)
            canvas_drawing = canvas_drawing[-1, :, :]
            canvas_drawing = canvas_drawing.squeeze(0).detach().cpu()

            overlap_image = other_image + canvas
            overlap_image = 5*(overlap_image > .5) + overlap_image
            overlap_image = overlap_image.detach().cpu()

            canvas = canvas.detach().cpu()
            other_image = other_image.detach().cpu()

            fig, axes = plt.subplots(1, 4, figsize=(30, 10))

            axes[0].imshow(canvas, cmap='grey')
            axes[0].set_title("By SDF AI Drawing")
            axes[0].axis('off')

            axes[1].imshow(canvas_drawing, cmap='grey', vmin=0, vmax=1)
            axes[1].set_title("SDF and Gaussian AI Drawing")
            axes[1].axis('off')

            axes[2].imshow(overlap_image)
            axes[2].set_title("Overlap Image")
            axes[2].axis('off')

            axes[3].imshow(other_image, cmap='grey')
            axes[3].set_title("Image with Filter")
            axes[3].axis('off')

            

            plt.tight_layout()
            plt.show()


    def __str__(self):

        return self.strokes.__str__()


    @staticmethod
    def _smoothstep(a, b, t):
        """
        Smooth interpolation from a to b
        f: ℝ²×[0,1] → [a,b],  f(a,b,α) = (b-a)·α²(3-2α) + a
        """

        smooth_funct = (t**2)*(3-2*t)
        ab_lerp = (b-a)*smooth_funct + a

        return ab_lerp

    @staticmethod
    def _nested_smoothstep(t, iterations=1):  
        """
        Applies f(t) = t²(3-2t) recursively for n iterations
        out = f∘f∘...∘f(t), t ∈ [0, 1]
        """

        t = torch.clamp(t, 0, 1)
        smooth_funct = lambda t : (t**2)*(3-2*t)
        out = t

        for _ in range(iterations):
            out = smooth_funct(out)

        return out


    def get_distance(self, index):

        stroke = self.strokes[:, index, :].clone()
        prior_stroke = self.strokes[:, index-1, :].clone()
        x_dist = prior_stroke[:, 0] - stroke[:,0]
        y_dist = prior_stroke[:, 1] - stroke[:,1]

        distance = torch.sqrt(x_dist**2 + y_dist**2 + 1e-8) 

        return distance      


    def get_alpha(self, index):

        return self.strokes[:, index ,4].clip(0, 1)


    def _line_loss(self, prefered_ld, index):

        criterion = nn.MSELoss()

        if self.strokes.shape[1] <= index.__abs__():
            
            zero_loss = self.strokes.clone().sum() * 0
            return zero_loss

        distance = self.get_distance(index)

        alpha = self.get_alpha(index)
        weight_dist = self._smoothstep(prefered_ld, distance, alpha)

        loss = criterion(weight_dist, prefered_ld)

        return loss


    def get_simga(self, index):

        return self.strokes[:, index, 2]


    def _sigma_loss(self, prefered_sigma, index):

        criterion = nn.MSELoss()
        simga = self.get_simga(index)

        return criterion(simga, prefered_sigma)


    def get_radius(self, index):

        return self.strokes[:, index, 3]

    def get_alpha_avg(self):

        batch_size, line_count,_ = self.strokes.shape

        return self.strokes[:, :, 4].sum() / (batch_size * line_count)

    def _alpha_loss(self):

        criterion = nn.BCELoss()

        return criterion(self.get_alpha_avg(), torch.ones_like(self.get_alpha_avg()))


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
        weight_angle = self._nested_smoothstep(angle, 1)

        angle_loss = criterion(
            weight_angle, 
            torch.ones_like(weight_angle)*0.95
        )

        return angle_loss


    def loss(self,* , prefered_distance, prefered_sigma, prefered_radius):

        prefered_distance_vec = torch.ones(
            self.batch_size, 
            device=self.device
        ) * prefered_distance

        prefered_sigma_vec = torch.ones(
            self.batch_size, 
            device=self.device
        ) * prefered_sigma

        prefered_radius_vec = torch.ones(
            self.batch_size, 
            device=self.device
        ) * prefered_radius

        loss = torch.stack([
            self._line_loss(prefered_distance_vec, -1),
            self._sigma_loss(prefered_sigma_vec, -1),
            self._radius_loss(prefered_radius_vec, -1),
            self._angle_loss(-1),
            self._alpha_loss(),
        ])

        return loss.sum()


    def forget_grads(self):

        self.strokes = self.strokes.detach()

    def debug_printout(self):

        prefered_sigma = 0.05
        prefered_radius = 0.05

        print(f"Largest x: {self.strokes[-1, :, 0].max():.4f} < {self.width}\n"
            f"Smallest x: {self.strokes[-1, :, 0].min():.4f} > {0}\n"
            f"Largest y: {self.strokes[-1, :, 1].max():.4f} < {self.height}\n"
            f"Smallest y: {self.strokes[-1, :, 1].min():.4f} > {0}\n"
            f"Largest sigma: {self.strokes[-1, :, 2].max():.4f} ~= {prefered_sigma}\n"
            f"Smallest sigma: {self.strokes[-1, :, 2].min():.4f} ~= {prefered_sigma}\n"
            f"Largest radius: {self.strokes[-1, :, 3].max():.4f} ~= {prefered_radius}\n"
            f"Smallest radius: {self.strokes[-1, :, 3].min():.4f} ~= {prefered_radius}\n"
            f"Largest alpha: {self.strokes[-1, :, 4].max():.4f} ~= {1}\n"
            f"Smallest alpha: {self.strokes[-1, :, 4].min():.4f} ~= {0}\n")



if __name__ == "__main__":

    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    strokes = Strokes(64, 300, 300, device="cuda")

    stroke = torch.zeros(64, 5).to("cuda")
    stroke[:, 0] = 50
    stroke[:, 1] = 50
    stroke[:, 2] = 0.0006
    stroke[:, 3] = 0.0006
    stroke[:, 4] = 1
    stroke.requires_grad_(True)

    strokes.draw(stroke)

    stroke = torch.zeros(64, 5).to("cuda")
    stroke[:, 0] = 200
    stroke[:, 1] = 200
    stroke[:, 2] = 0.016
    stroke[:, 3] = 0.016
    stroke[:, 4] = 1
    stroke.requires_grad_(True)

    strokes.draw(stroke)

    stroke = torch.zeros(64, 5).to("cuda")
    stroke[:, 0] = 50
    stroke[:, 1] = 200
    stroke[:, 2] = 0.003
    stroke[:, 3] = 0.003
    stroke[:, 4] = 0.33
    stroke.requires_grad_(True)

    strokes.draw(stroke)

    stroke = torch.zeros(64, 5).to("cuda")
    stroke[:, 0] = 240
    stroke[:, 1] = 240
    stroke[:, 2] = 0.006
    stroke[:, 3] = 0.006
    stroke[:, 4] = 1
    stroke.requires_grad_(True)

    strokes.draw(stroke)

    stroke = torch.zeros(64, 5).to("cuda")
    stroke[:, 0] = 290
    stroke[:, 1] = 0
    stroke[:, 2] = 0.006
    stroke[:, 3] = 0.006
    stroke[:, 4] = 1
    stroke.requires_grad_(True)

    strokes.draw(stroke)


    dummy_strokes = torch.rand(10, 5).to('cuda')
    render_lines_sdf(dummy_strokes, 300, 300, for_model=True)
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
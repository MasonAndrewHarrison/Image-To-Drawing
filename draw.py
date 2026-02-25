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

        self.strokes = torch.zeros(batch_size, 0, 7).to(device)

    def get_strokes(self):

        new_stroke = self.strokes.clone()

        scale = new_stroke.new_tensor([self.width, self.height, self.width, self.height, 1, 1, 1])
        new_stroke = new_stroke / scale

        return new_stroke


    def _draw_new_line(self, new_stroke):

        new_stroke = new_stroke.unsqueeze(1)
        self.strokes = torch.concat([self.strokes, new_stroke], dim=1)

    def draw(self, new_stroke, new_line: bool = False):

        _,last_index,_ = self.strokes.shape

        if new_line or last_index == 0: 

            opacity_adjusted = torch.cat([
                new_stroke[:, :6],
                (new_stroke[:, 6] + 1).unsqueeze(1)
            ], dim=1)

            self._draw_new_line(opacity_adjusted)

        else:

            last_stroke = self.strokes[:, last_index-1, :].clone()

            last_xy2 = last_stroke[:, 2:4]
            current_xy1 = new_stroke[:, 0:2]
            avg_sig_r = (last_stroke[:, 4:6] + new_stroke[:, 4:6]) / 2
            opacity = new_stroke[:, 6].unsqueeze(1)

            connecting_stroke = torch.cat([last_xy2, current_xy1, avg_sig_r, opacity], dim=1)

            new_stroke = torch.cat([new_stroke[:, :6], new_stroke[:, 6].unsqueeze(1) + 2], dim=1)

            self._draw_new_line(connecting_stroke)
            self._draw_new_line(new_stroke)


    def canvas(self):

        strokes = self.get_strokes()

        return render_lines_sdf_batched(
            strokes=strokes[:, :, :], 
            height=self.height, 
            width=self.width, 
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

            overlap_image = other_image + canvas
            overlap_image = 5*(overlap_image > .5) + overlap_image
            overlap_image = overlap_image.detach().cpu()

            canvas = canvas.detach().cpu()
            other_image = other_image.detach().cpu()

            fig, axes = plt.subplots(1, 3, figsize=(30, 10))

            axes[0].imshow(canvas, cmap='grey')
            axes[0].set_title("AI Drawing")
            axes[0].axis('off')

            axes[1].imshow(overlap_image)
            axes[1].set_title("Overlap Image")
            axes[1].axis('off')

            axes[2].imshow(other_image, cmap='grey')
            axes[2].set_title("Image with Filter")
            axes[2].axis('off')

            

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
        x_dist = stroke[:,2] - stroke[:,0]
        y_dist = stroke[:,3] - stroke[:,1]

        distance = torch.sqrt(x_dist**2 + y_dist**2 + 1e-8) 

        return distance      


    def get_alpha(self, index):

        return self.strokes[:, index ,6].clip(0, 1)


    def _line_loss(self, prefered_ld, index):

        criterion = nn.MSELoss()

        if self.strokes.shape[1] < index.__abs__():
            
            zero_loss = self.strokes.clone().sum() * 0
            return zero_loss

        distance = self.get_distance(index)

        alpha = self.get_alpha(index)
        weight_dist = self._smoothstep(prefered_ld, distance, alpha)

        loss = criterion(weight_dist, prefered_ld)

        return loss


    def get_simga(self, index):

        return self.strokes[:, index, 4]


    def _sigma_loss(self, prefered_sigma, index):

        criterion = nn.MSELoss()
        simga = self.get_simga(index)

        return criterion(simga, prefered_sigma)


    def get_radius(self, index):

        return self.strokes[:, index, 5]


    def _radius_loss(self, prefered_radius, index):

        criterion = nn.MSELoss()
        radius = self.get_radius(index)

        return criterion(radius, prefered_radius)


    def get_angle(self, origin_x, origin_y, xa, ya, xb, yb):
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
        theda_cos = torch.clamp(theda_cos, -1, 1)

        theda = torch.acos(theda_cos)
        theda = theda * valid.float()

        return theda / torch.pi


    def get_line_angle(self, index):

        xa = self.strokes[:, index-2, 0].clone()
        ya = self.strokes[:, index-2, 1].clone()
        xb = self.strokes[:, index-2, 2].clone()
        yb = self.strokes[:, index-2, 3].clone()
        xc = self.strokes[:, index, 0].clone()
        yc = self.strokes[:, index, 1].clone()
        xd = self.strokes[:, index, 2].clone()
        yd = self.strokes[:, index, 3].clone()

        angle1 = self.get_angle(
            xa=xa,
            ya=ya,
            origin_x=xb,
            origin_y=yb,
            xb=xc,
            yb=yc,
        )

        angle2 = self.get_angle(
            xa=xb,
            ya=yb,
            origin_x=xc,
            origin_y=yc,
            xb=xd,
            yb=yd,
        )

        return angle1, angle2


    def _angle_loss(self, index):

        if self.strokes.shape[1] < index.__abs__()+1:
            
            zero_loss = self.strokes.clone().sum() * 0
            return zero_loss

        criterion = nn.BCELoss()

        angle1, angle2 = self.get_line_angle(index)

        weight_angle1 = self._nested_smoothstep(angle1, 1)
        weight_angle2 = self._nested_smoothstep(angle2, 1)

        valid = (angle1 > 1e-7) & (angle2 > 1e-7)
        if not valid.any():
            return self.strokes.clone().sum() * 0

        angle_loss1 = criterion(
            weight_angle1, 
            torch.ones_like(weight_angle1)
        )

        angle_loss2 = criterion(
            weight_angle2, 
            torch.ones_like(weight_angle2)
        )

        avg_angle = (angle_loss1 + angle_loss2) / 2


        test_loss = 2 - (angle1 + angle2)
        #print(test_loss)
        return test_loss


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
            #self._line_loss(prefered_distance_vec, -2),
            #self._line_loss(prefered_distance_vec, -1),
            #self._sigma_loss(prefered_sigma_vec, -1),
            #self._radius_loss(prefered_radius_vec, -1),
            self._angle_loss(-1)*5,
        ])

        return loss.sum()/4


    def forget_grads(self):

        self.strokes = self.strokes.detach()



if __name__ == "__main__":

    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    strokes = Strokes(64, 300, 300, device="cuda")

    stroke = torch.zeros(64, 7).to("cuda")
    stroke[:, 0] = 50
    stroke[:, 1] = 50
    stroke[:, 2] = 200
    stroke[:, 3] = 200
    stroke[:, 4] = 0.006
    stroke[:, 5] = 0.006
    stroke[:, 6] = -0.5
    stroke.requires_grad_(True)

    strokes.draw(stroke)

    stroke = torch.zeros(64, 7).to("cuda")
    stroke[:, 0] = 200
    stroke[:, 1] = 50
    stroke[:, 2] = 50
    stroke[:, 3] = 200
    stroke[:, 4] = 0.016
    stroke[:, 5] = 0.016
    stroke[:, 6] = 0.5
    stroke.requires_grad_(True)

    strokes.draw(stroke)

    dummy_strokes = torch.rand(10, 7).to('cuda')
    render_lines_sdf(dummy_strokes, 300, 300)
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

    strokes._angle_loss(-1)

    end = time.time()

    print(f"Time to generate canvas: {(end - start):.4f} seconds.")
    print(f"Check does grads work with 'draw.py' pipe line: {canvas.requires_grad}, {loss.requires_grad}")
    print(f"Check does cuda work with 'draw.py' pipe line: {canvas.device.__str__()[:-2] == 'cuda'} || device used: {canvas.device}")

    print("Visual Test: ")

    strokes.render()

    print("Exit")
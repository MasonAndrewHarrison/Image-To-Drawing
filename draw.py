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

        #new_stroke[:, :, 0] /= self.width
        #new_stroke[:, :, 1] /= self.height
        #new_stroke[:, :, 2] /= self.width
        #new_stroke[:, :, 3] /= self.height

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

    def render(self):

        canvas = self.canvas()
        canvas = canvas[-1, :, :]
        canvas = canvas.squeeze(0).detach().cpu()

        plt.imshow(canvas, cmap='grey')
        plt.show()

    def __str__(self):

        return self.strokes.__str__()

    # f: ℝ²×[0,1] → [a,b],  f(a,b,α) = (b-a)·α²(3-2α) + a
    @staticmethod
    def _lerp(a, b, t):

        lerp_funct = (t**2)*(3-2*t)
        ab_lerp = (b-a)*lerp_funct + a

        return ab_lerp

    def _line_loss(self, prefered_ld, index):

        criterion = nn.MSELoss()

        if self.strokes.shape[1] < index.__abs__():
            
            zero_loss = self.strokes.clone().sum() * 0
            return zero_loss

        stroke = self.strokes[:, index, :].clone()
        x_dist = stroke[:,2] - stroke[:,0]
        y_dist = stroke[:,3] - stroke[:,1]

        distance = torch.sqrt(x_dist**2 + y_dist**2 + 1e-8)

        alpha = stroke[:,6].clip(0, 1)

        weight_dist = self._lerp(prefered_ld, distance, alpha)

        loss = criterion(weight_dist, prefered_ld)

        return loss

    def _sigma_loss(self, prefered_sigma, index):

        criterion = nn.MSELoss()

        simga = self.strokes[:, index, 4]

        return criterion(simga, prefered_sigma)

    def _radius_loss(self, prefered_radius, index):

        criterion = nn.MSELoss()

        radius = self.strokes[:, index, 5]

        return criterion(radius, prefered_radius)

    def loss(self, prefered_distance, prefered_sigma, prefered_radius):

        loss_distance2 = self._line_loss(prefered_distance, -2)
        loss_distance1 = self._line_loss(prefered_distance, -1)
        loss_simga = self._sigma_loss(prefered_sigma, -1)
        loss_radius = self._radius_loss(prefered_radius, -1)

        loss = torch.stack([
            loss_distance1,
            loss_distance2,
            loss_simga,
            loss_radius,
        ])

        return loss.sum()/4

    def forget_grads(self):

        self.strokes = self.strokes.detach()

if __name__ == "__main__":

    strokes = Strokes(64, 300, 300, device="cuda")

    stroke1 = torch.zeros(32, 7).to("cuda")
    stroke1[:, 0] = 50
    stroke1[:, 1] = 50
    stroke1[:, 2] = 200
    stroke1[:, 3] = 200
    stroke1[:, 4] = 0.006
    stroke1[:, 5] = 0.006
    stroke1[:, 6] = -0.5
    stroke1.requires_grad_(True)

    stroke2 = torch.zeros(32, 7).to("cuda")
    stroke2[:, 0] = 50
    stroke2[:, 1] = 50
    stroke2[:, 2] = 100
    stroke2[:, 3] = 100
    stroke2[:, 4] = 0.006
    stroke2[:, 5] = 0.006
    stroke2[:, 6] = -0.5
    stroke2.requires_grad_(True)

    stroke = torch.concat([stroke1, stroke2], dim=0)

    strokes.draw(stroke)

    stroke = torch.zeros(64, 7).to("cuda")
    stroke[:, 0] = 50
    stroke[:, 1] = 200
    stroke[:, 2] = 200
    stroke[:, 3] = 50
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

    end = time.time()

    print(f"Time to generate canvas: {(end - start):.4f} seconds.")

    print(f"Check does grads work with 'draw.py' pipe line: {canvas.requires_grad}")
    print(f"Check does cuda work with 'draw.py' pipe line: {canvas.device.__str__()[:-2] == 'cuda'} || device used: {canvas.device}")

    print("Visual Test: ")

    strokes.render()

    print("Exit")
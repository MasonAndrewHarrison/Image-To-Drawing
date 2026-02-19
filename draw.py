import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from differentiable_rasterizer import render_lines_sdf, render_lines_sdf_batched
import random

class Strokes():

    def __init__(self, batch_size, height, width, device):

        self.width = width
        self.height = height
        self.batch_size = batch_size

        self.strokes = torch.zeros(batch_size, 0, 6).to(device)

    def get_strokes(self):

        new_stroke = self.strokes.clone()

        new_stroke[:, :, 0] /= self.width
        new_stroke[:, :, 1] /= self.height
        new_stroke[:, :, 2] /= self.width
        new_stroke[:, :, 3] /= self.height

        return new_stroke

    def draw_new_line(self, new_stroke):

        new_stroke = new_stroke.unsqueeze(1)
        print(new_stroke.shape, self.strokes.shape)
        self.strokes = torch.concat([self.strokes, new_stroke], dim=1)

    def draw(self, new_stroke, new_line: bool = False):

        _,last_index,_ = self.strokes.shape

        if new_line or last_index == 0: 
            self.draw_new_line(new_stroke)

        else:

            last_stroke = self.strokes[:, last_index-1, :]

            last_xy2 = last_stroke[:, 2:4]
            current_xy1 = new_stroke[:, 0:2]
            avg_sig_r = (last_stroke[:, 4:6] + new_stroke[:, 4:6]) / 2

            connecting_stroke = torch.concat([last_xy2, current_xy1, avg_sig_r], dim=1)

            self.draw_new_line(connecting_stroke)
            self.draw_new_line(new_stroke)

    def canvas(self):

        strokes = self.get_strokes()

        return render_lines_sdf_batched(
            strokes=strokes[:, :, :], 
            height=self.height, 
            width=self.width, 
        )

    def render(self):

        canvas = self.canvas()
        canvas = canvas[36, :, :, :]
        canvas = canvas.squeeze(0).detach().cpu()

        plt.imshow(canvas, cmap='grey')
        plt.show()

    def __str__(self):

        return self.strokes.__str__()


if __name__ == "__main__":

    strokes = Strokes(64, 300, 300, device="cuda")

    stroke1 = torch.zeros(32, 6).to("cuda")
    stroke1[:, 0] = 50
    stroke1[:, 1] = 50
    stroke1[:, 2] = 200
    stroke1[:, 3] = 200
    stroke1[:, 4] = 0.006
    stroke1[:, 5] = 0.006
    stroke1.requires_grad_(True)

    stroke2 = torch.zeros(32, 6).to("cuda")
    stroke2[:, 0] = 50
    stroke2[:, 1] = 50
    stroke2[:, 2] = 100
    stroke2[:, 3] = 100
    stroke2[:, 4] = 0.006
    stroke2[:, 5] = 0.006
    stroke2.requires_grad_(True)

    stroke = torch.concat([stroke1, stroke2], dim=0)

    strokes.draw(stroke)

    stroke = torch.zeros(64, 6).to("cuda")
    stroke[:, 0] = 50
    stroke[:, 1] = 200
    stroke[:, 2] = 200
    stroke[:, 3] = 50
    stroke[:, 4] = 0.016
    stroke[:, 5] = 0.016
    stroke.requires_grad_(True)

    strokes.draw(stroke)

    canvas = strokes.canvas()

    print(canvas.shape)

    print(f"Check does grads work with 'draw.py' pipe line: {canvas.requires_grad}")
    print(f"Check does cuda work with 'draw.py' pipe line: {canvas.device.__str__()[:-2] == 'cuda'} || device used: {canvas.device}")

    print("Visual Test: ")

    strokes.render()

    print("Exit")
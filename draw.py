import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from differentiable_rasterizer import render_lines
import random

class Strokes():

    def __init__(self, height, width, device):

        self.width = width
        self.height = height

        self.strokes = torch.zeros(0, 6).to(device)

    def get_strokes(self):

        new_stroke = self.strokes.clone()

        new_stroke[:, 0] /= self.width
        new_stroke[:, 1] /= self.height
        new_stroke[:, 2] /= self.width
        new_stroke[:, 3] /= self.height

        return new_stroke

    def draw_new_line(self, new_stroke):

        new_stroke = new_stroke.unsqueeze(0)
        self.strokes = torch.concat([self.strokes, new_stroke], dim=0)

        #print(self.strokes.requires_grad)

    def draw(self, new_stroke, new_line: bool = False):

        last_index,_ = self.strokes.shape

        if new_line or last_index == 0: 
            self.draw_new_line(new_stroke)

        else:

            last_stroke = self.strokes[last_index-1, :]

            last_xy2 = last_stroke[2:4]
            current_xy1 = new_stroke[0:2]
            avg_sig_r = (last_stroke[4:6] + new_stroke[4:6]) / 2

            connecting_stroke = torch.concat([last_xy2, current_xy1, avg_sig_r], dim=0)

            self.draw_new_line(connecting_stroke)
            self.draw_new_line(new_stroke)

    def canvas(self):

        strokes = self.get_strokes()

        return render_lines(
            strokes=strokes, 
            height=self.height, 
            width=self.width, 
        )

    def render(self):

        canvas = self.canvas()
        canvas = canvas.squeeze(0).detach().cpu()

        plt.imshow(canvas, cmap='grey')
        plt.show()

    def __str__(self):

        return self.strokes.__str__()


if __name__ == "__main__":

    strokes = Strokes(300, 300, device="cuda")

    stroke = torch.zeros(6).to("cuda")
    stroke[0] = 50
    stroke[1] = 50
    stroke[2] = 200
    stroke[3] = 200
    stroke[4] = 0.006
    stroke[5] = 0.006
    stroke.requires_grad_(True)

    strokes.draw(stroke)

    stroke = torch.zeros(6).to("cuda")
    stroke[0] = 50
    stroke[1] = 200
    stroke[2] = 200
    stroke[3] = 50
    stroke[4] = 0.016
    stroke[5] = 0.016
    stroke.requires_grad_(True)

    strokes.draw(stroke)

    canvas = strokes.canvas()

    print(f"Check does grads work with 'draw.py' pipe line: {canvas.requires_grad}")
    print(f"Check does cuda work with 'draw.py' pipe line: {canvas.device.__str__()[:-2] == 'cuda'} || device used: {canvas.device}")

    print("Visual Test: ")

    strokes.render()

    print("Exit")
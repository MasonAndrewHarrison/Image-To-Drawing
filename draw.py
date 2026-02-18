import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from differentiable_rasterizer import render_lines

class Strokes():

    def __init__(self, height, width):

        self.width = width
        self.height = height

        self.strokes = torch.zeros(0, 4)

    def get_strokes(self):

        new_stroke = self.strokes.detach()

        new_stroke[:, 0] /= self.width
        new_stroke[:, 1] /= self.height
        new_stroke[:, 2] /= self.width
        new_stroke[:, 3] /= self.height

        return new_stroke

    def draw_new_line(self, new_stroke):

        new_stroke = new_stroke.unsqueeze(0)
        self.strokes = torch.concat([self.strokes, new_stroke], dim=0)

    def draw(self, new_stroke, new_line: bool = False):

        last_index,_ = self.strokes.shape

        if new_line or last_index == 0: 
            self.draw_new_line(new_stroke)

        else:
            last_x, last_y = self.strokes[last_index-1, 2:4]

            last_x = last_x.item()
            last_y = last_y.item()

            start_x = new_stroke[0].item()
            start_y = new_stroke[1].item()

            connecting_line = torch.tensor([last_x, last_y, start_x, start_y])

            self.draw_new_line(connecting_line)
            self.draw_new_line(new_stroke)

    def canvas(self, thickness=0.005):

        strokes = self.get_strokes()

        return render_lines(
            strokes=strokes, 
            height=self.height, 
            width=self.width, 
            brush_thickness=thickness,
        )

    def render(self, thickness=0.005):

        canvas = self.canvas(thickness=thickness)
        canvas = canvas.squeeze(0).detach().cpu()

        plt.imshow(canvas, cmap='grey')
        plt.show()
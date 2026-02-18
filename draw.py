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

        return self.strokes

    def draw_new_line(self, x1, y1, x2, y2):

        x1 = x1 / self.width
        x2 = x2 / self.width
        y1 = y1 / self.height
        y2 = y2 / self.height

        new_stroke = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
        self.strokes = torch.concat([self.strokes, new_stroke], dim=0)

    def draw(self, x1, y1, x2, y2, new_line: bool = False):

        last_index,_ = self.strokes.shape

        if new_line or last_index == 0: 
            self.draw_new_line(x1, y1, x2, y2)

        else:
            last_x, last_y = self.strokes[last_index-1, 2:4]

            last_x = last_x.item()
            last_y = last_y.item()

            last_x *= self.width
            last_y *= self.height

            self.draw_new_line(last_x, last_y, x1, y1)
            self.draw_new_line(x1, y1, x2, y2)

    def canvas(self, thickness=0.01):

        return render_lines(
            strokes=self.strokes, 
            height=self.height, 
            width=self.width, 
            brush_thickness=thickness,
        )

    def render(self, thickness=0.005):

        canvas = self.canvas(thickness=thickness)
        canvas = canvas.squeeze(0).detach().cpu()

        plt.imshow(canvas, cmap='grey')
        plt.show()
import torch
import matplotlib.pyplot as plt
import numpy as np
import pygame
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random
from model import Model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(in_channels=1, out_features=6).to(device)
model.eval()
model.load_state_dict(torch.load("Model_Weights.pth", map_location=device))

pygame.init()
font = pygame.font.Font(None, 74)

SCALE = 5
SCREEN_WIDTH = 320 * SCALE
SCREEN_HEIGHT = 240 * SCALE

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

current_stroke = []
strokes_list = []

clock = pygame.time.Clock()

running = True
mouse_x = 0
mouse_y = 0
left_clicked = False
right_clicked = False

def get_28x28_matrix():

    string_image = pygame.image.tostring(screen, 'RGB')
    temp_image = Image.frombytes('RGB', (SCREEN_WIDTH, SCREEN_HEIGHT), string_image)
    
    temp_image = temp_image.convert('L')
    temp_image = temp_image.resize((320, 240), Image.LANCZOS)

    matrix = (np.array(temp_image) / 255.0)*2 - 1
    matrix = torch.tensor(matrix, dtype=torch.float32)
    matrix = matrix.unsqueeze(0).unsqueeze(0)

    return matrix


def drawStroke(stroke, diameter):

    for i, point in enumerate(stroke):
        if len(stroke) > 1 and not i == 0:
            pygame.draw.line(screen, BLACK, stroke[i-1], stroke[i], diameter)
            pygame.draw.circle(screen, BLACK, stroke[i], diameter // 2 - 1.5)

def drawAllStrokes(current_stroke, strokes_list, diameter):

    for stroke in strokes_list:
        drawStroke(stroke, diameter)

    drawStroke(current_stroke, diameter)  

while running:

    mouse_moved = False
    if_unleft_clicked = False
    space_pressed = False
    ai_new_line = False

    for event in pygame.event.get():

        if event.type == pygame.MOUSEMOTION:
            mouse_moved = True
            mouse_x, mouse_y = event.pos
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                left_clicked = True
            elif event.button == 3:
                right_clicked = True
        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                space_pressed = True
                matrix = get_28x28_matrix()
                out = model(matrix.to(device))
                mouse_x, mouse_y = out[:, 2], out[:, 3]
                mouse_x_end, mouse_y_end = out[:, 4], out[:, 5]
                ai_new_line = (out[:, 0].item()+0.1) * random.randint(-1, 1) <= 0
                print(ai_new_line, out[:, 0])

                mouse_x = mouse_x.item()
                mouse_y = mouse_y.item()
                mouse_x_end = mouse_x_end.item()
                mouse_y_end = mouse_y_end.item()

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                left_clicked = False
                if_unleft_clicked = True
            elif event.button == 3:
                right_clicked = False

        if event.type == pygame.QUIT:
            running = False

    if (left_clicked and mouse_moved): 
        current_stroke.append((mouse_x, mouse_y))

    elif space_pressed:

        if ai_new_line:
            strokes_list.append(current_stroke)
            current_stroke = []

        ry = random.randint(1, SCREEN_HEIGHT)
        rx = random.randint(1, SCREEN_WIDTH)

        current_stroke.append((mouse_x * rx * 4, mouse_y * ry * 4))

        ry = random.randint(1, SCREEN_HEIGHT)
        rx = random.randint(1, SCREEN_WIDTH)

        current_stroke.append((mouse_x_end * rx * 4, mouse_y_end * ry * 4))
 
    elif if_unleft_clicked:
        strokes_list.append(current_stroke)
        current_stroke = []

    if right_clicked:
        strokes_list = []

    drawAllStrokes(current_stroke, strokes_list, 20)
    screen.fill(WHITE)
    drawAllStrokes(current_stroke, strokes_list, 10)
        
    pygame.display.update()

    clock.tick(60)

pygame.quit()
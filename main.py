import torch
import matplotlib.pyplot as plt
import numpy as np
import pygame
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    temp_image = temp_image.resize((28, 28), Image.LANCZOS)

    matrix = (np.array(temp_image) / 255.0)*2 - 1
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

    elif if_unleft_clicked:
        strokes_list.append(current_stroke)
        current_stroke = []

    if right_clicked:
        strokes_list = []

    drawAllStrokes(current_stroke, strokes_list, 20)
    matrix = get_28x28_matrix()
    screen.fill(WHITE)
    drawAllStrokes(current_stroke, strokes_list, 10)
        
    pygame.display.update()

    clock.tick(60)

pygame.quit()
# Image to Drawing

> **Work in Progress** — This is a personal research project and is actively being developed.

A LSTM is used to create vector line art of an image and can be saved as an .SVG file. 

## Steps to Train Network

1. Setup Repo and Venv

2. Install PyTorch:\
   (For CUDA)`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128`\
   (For CPU)`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

3. Install dependencies:\
   `pip install -r requirements.txt`

## Convert an Image

> **add later bash to colorize png image** \
    `download weights from hugging face`
    `python main --...`

## Train Network

> **add yaml config file later** \
  `python create_dataset.py`\
  `python train.py`

## Overview

   This project take in an image it then uses a Flow Extented or an Extented Difference of Gaussians filter on that image. After this step the LSTM recurrent neural network takes the filtered image and traces that image which is converted into a .SVG file which will be the line art of that iamge.

## Architecture

- Flow Extented Difference of Gaussians
- LSTM Recurrent Neural Network



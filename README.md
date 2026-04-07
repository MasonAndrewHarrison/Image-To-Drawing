# Image to Drawing

> **Work in Progress** — This is a personal research project and is actively being developed.

1. Setup Repo and Venv

2. Install PyTorch:\
   (For CUDA)`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128`\
   (For CPU)`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

3. Install dependencies:\
   `pip install -r requirements.txt`

## Convert an Image

    `python main --...`


## Overview

   This project take in an image it then uses a Flow Extented Difference of Gaussians filter on that image. After this it goes throught and traces the outline.

   
## Architecture

- Flow Extented Difference of Gaussians
- Sign Distance Function



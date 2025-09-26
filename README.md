# Real-Time Style Transfer Webcam App

This project applies real-time style transfer to your webcam using PyTorch and OpenCV. Users can switch between multiple artistic styles and save snapshots.

## Features
- Real-time webcam style transfer
- Pre-trained styles: candy, mosaic, udnie
- Save snapshots
- Keyboard controls: 1/2/3 to switch styles, s to save, q to quit

## Requirements
- Python 3.7+
- torch, torchvision, opencv-python, pillow
Install with: pip install torch torchvision opencv-python pillow

## Usage
Run the app:
python style_transfer_app.py
Snapshots are saved in the project folder.

## Project Structure
project/
├─ fast_neural_style/        # transformer_net.py, utils.py
├─ models/                   # candy.pth, mosaic.pth, udnie.pth
└─ style_transfer_app.py" > README.md

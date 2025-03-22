# Medical Segmentation API with Uncertainty Estimation

This repository contains a production-ready FastAPI service for performing semantic segmentation on medical images using a Swin-Tiny UPerNet architecture. The model includes Monte Carlo (MC) Dropout at inference time to estimate predictive uncertainty.

## Features

- FastAPI backend with a RESTful endpoint for image segmentation
- Swin-Tiny transformer encoder with custom UPerNet decoder
- Monte Carlo Dropout for uncertainty quantification
- Uncertainty map visualization using OpenCV colormaps
- Side-by-side output of predicted mask and uncertainty heatmap
- Modular and well-documented codebase for easy extension and deployment

## Inference Workflow

1. The API accepts an image uploaded via a POST request to the `/predict/` endpoint.
2. The image is preprocessed and passed through the segmentation model.
3. The model performs multiple stochastic forward passes using MC Dropout.
4. The mean prediction is thresholded to produce a binary mask.
5. The standard deviation across predictions is visualized as an uncertainty heatmap.
6. The binary mask and heatmap are returned as a combined PNG image.

## File Structure

| File              | Description                                                  |
|-------------------|--------------------------------------------------------------|
| `main.py`         | FastAPI application and inference logic                      |
| `model.py`        | Swin-Tiny + UPerNet model architecture and utility functions |
| `requirements.txt`| List of required Python packages                             |
| `.gitignore`      | Specifies files and directories to be ignored by Git         |

## Requirements

- Python 3.9+
- PyTorch
- FastAPI
- OpenCV
- Pillow
- timm
- torchvision
- uvicorn

To install dependencies:

pip install -r requirements.txt

## Author

Developed by Samyak Shrestha

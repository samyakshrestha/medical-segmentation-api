# Skin Lesion Segmentation API with Uncertainty Estimation

This repository contains a production-ready FastAPI service for performing semantic segmentation of skin lesions, trained on the ISIC 2018 dataset. The segmentation model uses a Swin-Tiny transformer encoder with a custom UPerNet decoder, and incorporates Monte Carlo (MC) Dropout at inference time to estimate predictive uncertainty.

## Features

- RESTful FastAPI endpoint for real-time image segmentation
- Swin-Tiny + UPerNet architecture for accurate lesion boundary detection
- Monte Carlo Dropout-based uncertainty quantification
- Heatmap visualization of model uncertainty using OpenCV colormaps
- Side-by-side output of the predicted segmentation mask and uncertainty map
- Clean, modular codebase suitable for further extension and deployment

## Inference Workflow

1. The API accepts a dermoscopic image uploaded via a POST request to the `/predict/` endpoint.
2. The image is resized, normalized, and passed through the segmentation model.
3. The model performs multiple stochastic forward passes using MC Dropout.
4. The mean prediction is binarized to produce a segmentation mask.
5. The standard deviation across predictions is visualized as a heatmap to express uncertainty.
6. The API returns both the binary mask and uncertainty heatmap as a combined PNG image.

## File Structure

| File              | Description                                                  |
|-------------------|--------------------------------------------------------------|
| `main.py`         | FastAPI application and endpoint logic                       |
| `model.py`        | Model architecture and MC Dropout inference utilities        |
| `requirements.txt`| Required Python packages                                     |
| `.gitignore`      | Specifies ignored files (e.g., `venv/`, model weights)       |

## Requirements

- Python 3.9+
- PyTorch
- FastAPI
- OpenCV
- Pillow
- timm
- torchvision
- uvicorn

Install dependencies:

pip install -r requirements.txt

## Model Weights

To run the API, you will need to download the pretrained model weights and place the file in the root directory of this repository.

**Download link:**

[best_swin_upernet_main.pth](https://huggingface.co/samyakshrestha/swin-medical-segmentation/resolve/main/best_swin_upernet_main.pth?download=true)

After downloading, ensure the file is named exactly "best_swin_upernet_main.pth" and saved in the same directory as `main.py` and `model.py`.

This weight file contains the Swin-Tiny UPerNet model trained on the ISIC 2018 skin lesion segmentation dataset.


## Dataset

This project uses the ISIC 2018 Challenge Dataset for training, which contains dermoscopic images of skin lesions annotated for binary segmentation.

Dataset access: https://challenge.isic-archive.com/data/

## Docker Support

This project includes a `Dockerfile` for containerized deployment. Run the API with Docker:

### Build the Docker image
docker build -t medical-segmentation-api .

### Run the container
docker run -p 8000:8000 medical-segmentation-api

The API will then be available at:
http://localhost:8000/predict/

## Author

Samyak Shrestha 

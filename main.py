from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from model import load_model, predict_with_uncertainty
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import torch

app = FastAPI()

# Load the model when the API starts
model = load_model()
model.eval()

def convert_to_image(array, colormap=None):
    array = (array * 255).astype(np.uint8)  # Normalize to 0–255 range
    if colormap is not None:
        array = cv2.applyColorMap(array, colormap)
    return Image.fromarray(array)

@app.post("/predict/")
async def predict_mask(file: UploadFile = File(...)):
    # Read and preprocess the image
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    image = image.resize((224, 224))
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0)

    # Perform MC Dropout Inference
    preds_mean, preds_uncertainty = predict_with_uncertainty(image_tensor)

    # Binary mask (0 or 255)
    pred_binary = (preds_mean > 0.5).astype(np.uint8) * 255
    mask_image = Image.fromarray(pred_binary).convert("L")

    # Normalize and apply colormap to uncertainty
    uncertainty = (preds_uncertainty - preds_uncertainty.min()) / (preds_uncertainty.max() - preds_uncertainty.min() + 1e-8)
    uncertainty_colormap = cv2.applyColorMap((uncertainty * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    uncertainty_image = Image.fromarray(uncertainty_colormap).convert("RGB")

    # Combine side by side
    combined = Image.new("RGB", (mask_image.width + uncertainty_image.width, mask_image.height))
    combined.paste(mask_image.convert("RGB"), (0, 0))
    combined.paste(uncertainty_image, (mask_image.width, 0))

    # Save to buffer and return
    img_io = BytesIO()
    combined.save(img_io, format="PNG")
    img_io.seek(0)

    return Response(content=img_io.getvalue(), media_type="image/png")
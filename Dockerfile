# Use the official Python image as a base
FROM python:3.9-slim

# Install system dependencies for OpenCV and wget
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 wget && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model weights from Hugging Face
RUN wget -O best_swin_upernet_main.pth "https://huggingface.co/samyakshrestha/swin-medical-segmentation/resolve/main/best_swin_upernet_main.pth?download=true"

# Copy the rest of the app code
COPY . .

# Expose the port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
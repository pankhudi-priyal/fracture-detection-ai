from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import json
import base64
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

app = FastAPI()

# Allow your frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local dev; restrict this before real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load config
with open("model/model_config.json") as f:
    config = json.load(f)
THRESHOLD = config["threshold"]

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.densenet121(weights=None)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 1)
model.load_state_dict(torch.load("model/best_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Grad-CAM setup
target_layers = [model.features.denseblock4]
cam = GradCAM(model=model, target_layers=target_layers)

# Same transform used during evaluation in Colab
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/")
def root():
    return {"status": "Fracture detection API is running", "threshold": THRESHOLD}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    raw_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(raw_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()

    prediction = "Fractured" if prob >= THRESHOLD else "Not Fractured"

    # Generate Grad-CAM overlay
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    rgb_img = np.array(raw_image.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Convert overlay image to base64 so it can be sent as JSON
    overlay_img = Image.fromarray(visualization)
    buffer = io.BytesIO()
    overlay_img.save(buffer, format="PNG")
    overlay_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "prediction": prediction,
        "confidence": round(prob, 4),
        "threshold_used": THRESHOLD,
        "heatmap_image": f"data:image/png;base64,{overlay_base64}"
    }
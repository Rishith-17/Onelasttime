from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import io

app = FastAPI()

# Allow CORS for frontend or Lovable AI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and processor ONCE during startup
model_name = "dima806/facial_emotions_image_detection"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)
model.eval()  # no gradients

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_idx = logits.argmax(-1).item()
            emotion = model.config.id2label[predicted_idx]

        return {"emotion": emotion}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
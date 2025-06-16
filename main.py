from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import io

# Load processor and model (faster and more memory efficient than pipeline)
processor = AutoProcessor.from_pretrained("dima806/facial_emotions_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")
model.eval()

app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess image
        inputs = processor(images=img, return_tensors="pt")

        with torch.no_grad():  # No gradients = lower memory
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()

        predicted_label = model.config.id2label[predicted_class_idx]

        # Free up memory
        del inputs, outputs, logits
        torch.cuda.empty_cache()

        return {"emotion": predicted_label}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
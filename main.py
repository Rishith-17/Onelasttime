from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import pipeline
from PIL import Image
import io

# Load the model from Hugging Face
emotion_model = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

app = FastAPI()

# CORS to allow frontend access
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

        # Use model to predict
        predictions = emotion_model(img)
        top_emotion = predictions[0]["label"]

        return {"emotion": top_emotion}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
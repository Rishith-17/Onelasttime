from fastapi import FastAPI, File, UploadFile
from transformers import pipeline
from PIL import Image
import io

app = FastAPI()
emotion_pipe = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    results = emotion_pipe(image)
    if results:
        top = results[0]
        return {"label": top["label"], "score": top["score"]}
    return {"label": "No face detected", "score": 0}

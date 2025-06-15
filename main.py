from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

app = FastAPI()

# CORS settings (unchanged)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_api(image: UploadFile = File(...)):  # ✅ Fix is here
    img = Image.open(image.file)  # ✅ Updated for UploadFile

    # Your model prediction logic here
    # Example stub:
    prediction = model.predict(img)  # Make sure 'model' is defined elsewhere
    return {"emotion": prediction}
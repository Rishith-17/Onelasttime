from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

app = FastAPI()

# CORS (keep this)
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
        img = Image.open(image.file)
        # 🧪 Just for testing — return fixed response
        return {"emotion": "happy"}
    except Exception as e:
        # ✅ Return error as JSON (never crash silently)
        return JSONResponse(status_code=500, content={"error": str(e)})
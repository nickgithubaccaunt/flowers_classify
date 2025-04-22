from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image
import os

app = FastAPI(
    title="Flower Classification API",
    description="API for classifying flower images into 5 categories",
    version="1.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Загрузка модели
model = load_model("app/model.h5")

# Классы цветов (порядок должен соответствовать обучению модели)
CLASS_NAMES = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']


def preprocess_image(file, target_size=(128, 128)):
    """Подготовка изображения для модели"""
    try:
        image = Image.open(BytesIO(file)).convert("RGB")
        image = image.resize(target_size)
        image = np.array(image) / 255.0
        return np.expand_dims(image, axis=0)
    except Exception as e:
        raise HTTPException(400, detail=f"Invalid image: {str(e)}")


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Эндпоинт для классификации изображений цветов"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, detail="Only image files are accepted")

    try:
        image = preprocess_image(await file.read())
        predictions = model.predict(image)[0]

        return {
            "predicted_class": CLASS_NAMES[np.argmax(predictions)],
            "probabilities": {
                cls: float(prob) for cls, prob in zip(CLASS_NAMES, predictions)
            }
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/")
async def health_check():
    return {"status": "OK", "message": "Flower Classification API is running"}
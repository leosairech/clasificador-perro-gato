from fastapi import FastAPI, UploadFile, File
import uuid
from pathlib import Path
import shutil
import tensorflow as tf

from app.inferencia import predict

app = FastAPI()

model = tf.keras.models.load_model(
    "models/model.keras",
    compile=False
)

TEMP = Path("temp")
TEMP.mkdir(exist_ok=True)

CLASSES = ["gato", "perro"]

@app.post("/predict")
def predict_api(file: UploadFile = File(...)):
    img_id = str(uuid.uuid4())

    img_path = TEMP / f"{img_id}.jpg"
    txt_path = TEMP / f"{img_id}.txt"

    with img_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    pred = predict(model, str(img_path), str(txt_path))

    return {
        "prediccion": CLASSES[pred]
    }

from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import torch
from ultralytics import YOLO

app = FastAPI()

model_name = "yolov8n.pt"
model = YOLO(model_name)

@app.get("/")
async def home():
    return f"yolo-api running with {model_name}"

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    results = model(image)

    objects = []
    for r in results:
        for box, cls, score in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            objects.append({
                "label": model.names[int(cls)],
                "score": float(score),
                "box": [float(x) for x in box]
            })

    response = {
        "model_name": model_name,
        "objects_num": len(objects),
        "objects": objects
    }

    return response


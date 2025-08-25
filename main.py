from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()
model = YOLO("yolo11n.pt")

@app.get("/")
async def home():
    return "yolov11-api running"

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes))

    results = model.predict(source=image)

    objects = []
    for r in results[0].boxes.data.tolist():
        objects.append({
            "box": r[:4],
            "score": r[4],
            "class": int(r[5])
        })

    return {"objects_num": len(objects), "objects": objects}

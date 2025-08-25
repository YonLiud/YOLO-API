# YOLO API

YOLO API is a FastAPI server for object detection using the YOLOv8 model.  
Send an image via POST `/detect` to get JSON with detected objects, bounding boxes, labels, and scores.

## Features
- FastAPI server, ready to deploy
- Structured JSON response
- Powered by YOLOv8 for fast and accurate detection
- Easy integration with OpenCV or other clients

## Example of usage
```bash
docker run -p 8000:8000 yonliud/yolo-api:latest
curl -X POST -F "file=@example.jpg" http://localhost:8000/detect
```

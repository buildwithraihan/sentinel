from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = FastAPI(title="SENTINEL API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path relative to this file's location, works both locally and on Render
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best.pt")
model = YOLO(MODEL_PATH)

HIGH_CONF = 0.7
MEDIUM_CONF = 0.45
HIGH_RISK_CLASSES = ['ship', 'aircraft']


def classify_natural_vs_artificial(detections):
    classified = []
    for det in detections:
        conf = det['confidence']
        if conf >= HIGH_CONF:
            category, quality = "Artificial", "HIGH"
        elif conf >= MEDIUM_CONF:
            category, quality = "Artificial", "MEDIUM"
        else:
            category, quality = "Uncertain — Review Required", "LOW"
        det['category'] = category
        det['detection_quality'] = quality
        classified.append(det)
    return classified


def assign_priority(classified_detections):
    for det in classified_detections:
        if det['detection_quality'] == "HIGH" and det['class'] in HIGH_RISK_CLASSES:
            det['priority'] = "HIGH"
        elif det['detection_quality'] in ["HIGH", "MEDIUM"]:
            det['priority'] = "MEDIUM"
        else:
            det['priority'] = "LOW"
    return classified_detections


def preprocess_image(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    clahe =

# SENTINEL — AI-Powered Underwater Intelligence

SIH 2026 | Problem Statement 26057 | MoES / NIOT

## Backend
FastAPI service exposing `/analyze` — takes a sonar image, returns detections
with class, confidence, bounding box, natural/artificial classification, and priority.

## Run locally
cd backend
pip install -r requirements.txt
uvicorn sentinel_api:app --host 0.0.0.0 --port 8000

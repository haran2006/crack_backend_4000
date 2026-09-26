"""
FastAPI Crack Detection Backend for Render.com
===============================================
Loads lightweight ONNX/YOLO crack segmentation model (best.onnx / best.pt)
and serves POST /detect endpoint.
"""

import base64
import io
import math
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

# Prefer best.onnx (lightweight, low RAM), fallback to best.pt
MODEL_PATH = Path("best.onnx") if Path("best.onnx").exists() else Path("best.pt")
CONF_THRESHOLD = 0.05
IMAGE_SIZE = 640
PORT = int(os.environ.get("PORT", 8000))

app = FastAPI(title="Smart Crack Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print(f"Loading model from {MODEL_PATH.resolve()} …")
model = YOLO(str(MODEL_PATH))
MODEL_NAME = f"YOLO Segmentation ({MODEL_PATH.name} Cloud)"
print(f"Model loaded successfully! Task={model.task} Classes={model.names}")


def severity_for(region_count: int, confidence: float) -> str:
    if region_count == 0:
        return "Low"
    if region_count >= 3 or confidence > 0.92:
        return "High"
    if region_count == 2 or confidence > 0.80:
        return "Moderate"
    return "Low"


def highest_severity(results: list) -> str:
    order = ["Low", "Moderate", "High"]
    best = "Low"
    for r in results:
        if order.index(r["severity"]) > order.index(best):
            best = r["severity"]
    return best


def pil_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def file_size_label(n_bytes: int) -> str:
    if n_bytes < 1024:
        return f"{n_bytes} B"
    if n_bytes < 1024 ** 2:
        return f"{n_bytes / 1024:.1f} KB"
    return f"{n_bytes / (1024 ** 2):.1f} MB"


@app.post("/detect")
async def detect(images: List[UploadFile] = File(...)):
    t_start = time.perf_counter()
    results_out = []

    for idx, upload in enumerate(images):
        raw = await upload.read()
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        img_array = np.array(pil_img)

        yolo_results = model.predict(
            source=img_array,
            conf=CONF_THRESHOLD,
            imgsz=IMAGE_SIZE,
            retina_masks=True,
            save=False,
            verbose=False,
        )
        result = yolo_results[0]

        annotated_bgr = result.plot(boxes=True, labels=True, conf=True, masks=True)
        annotated_rgb = annotated_bgr[:, :, ::-1]
        annotated_pil = Image.fromarray(annotated_rgb)
        annotated_url = pil_to_data_url(annotated_pil)
        original_url = pil_to_data_url(pil_img)

        boxes = result.boxes
        img_w, img_h = pil_img.size
        regions = []

        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                conf_val = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                regions.append({
                    "id": f"{upload.filename}-region-{i}",
                    "x": (x1 / img_w) * 100,
                    "y": (y1 / img_h) * 100,
                    "width": ((x2 - x1) / img_w) * 100,
                    "height": ((y2 - y1) / img_h) * 100,
                    "confidence": conf_val,
                    "className": class_name,
                })

        has_crack = len(regions) > 0
        max_conf = max((r["confidence"] for r in regions), default=0.0)
        avg_conf = sum(r["confidence"] for r in regions) / len(regions) if regions else 0.0
        region_count = len(regions)
        severity = severity_for(region_count, max_conf)

        crack_m = 0.0
        if has_crack:
            total_area_pct = sum(r["width"] * r["height"] for r in regions) / 10000
            crack_m = round(math.sqrt(total_area_pct) * 5.0, 2)

        results_out.append({
            "id": f"{upload.filename}-{idx}-{int(t_start)}",
            "fileName": upload.filename,
            "fileSizeLabel": file_size_label(len(raw)),
            "imageUrl": original_url,
            "annotatedUrl": annotated_url,
            "hasCrack": has_crack,
            "regions": regions,
            "confidence": round(avg_conf * 1000) / 10,
            "severity": severity,
            "crackLengthMeters": crack_m,
        })

    elapsed = round((time.perf_counter() - t_start) * 10) / 10

    summary = {
        "imagesAnalyzed": len(results_out),
        "imagesWithCracks": sum(1 for r in results_out if r["hasCrack"]),
        "totalRegions": sum(len(r["regions"]) for r in results_out),
        "averageConfidence": round(
            sum(r["confidence"] for r in results_out) / len(results_out) * 10
        ) / 10 if results_out else 0,
        "highestSeverity": highest_severity(results_out),
        "processingTimeSeconds": elapsed,
        "model": MODEL_NAME,
    }

    return JSONResponse({"results": results_out, "summary": summary})


@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_NAME}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)

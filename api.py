"""
FastAPI Crack Detection Backend for Render.com (Ultra-Lightweight ONNX)
======================================================================
Pure onnxruntime + numpy + pillow implementation.
Zero PyTorch / Ultralytics dependencies to keep memory < 200 MB
and prevent Render 512 MB Out-Of-Memory (OOM) crashes.
"""

import base64
import io
import math
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.onnx"
CONF_THRESHOLD = 0.05
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640
PORT = int(os.environ.get("PORT", 8000))

CLASSES = {
    0: "crack-dedection-2",
    1: "damaged",
    2: "n",
}

CLASS_COLORS = {
    0: (0, 229, 255),    # Cyan for cracks
    1: (255, 179, 0),    # Amber for damaged
    2: (186, 104, 200),  # Purple for normal / other
}

app = FastAPI(title="Smart Crack Detection API (Cloud)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print(f"Initializing pure ONNX Runtime session with {MODEL_PATH} ...")
ort_session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)
input_name = ort_session.get_inputs()[0].name
print("ONNX Runtime initialized successfully without PyTorch! Ready for inference.")


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.45) -> List[int]:
    """Pure NumPy Non-Maximum Suppression."""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


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


def pil_to_data_url(img: Image.Image, quality: int = 80) -> str:
    out_img = img
    if max(img.size) > 1280:
        out_img = img.copy()
        out_img.thumbnail((1280, 1280), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    out_img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


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
        orig_w, orig_h = pil_img.size

        # Preprocess for ONNX (resize to 640x640, normalize to 0..1, CHW)
        resized = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        img_np = np.array(resized, dtype=np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))  # (3, 640, 640)
        inp = np.expand_dims(img_np, axis=0)      # (1, 3, 640, 640)

        # Run ONNX inference
        outputs = ort_session.run(None, {input_name: inp})
        pred = outputs[0][0].T  # shape: (8400, 39)

        boxes_xywh = pred[:, :4]
        scores = pred[:, 4:7]

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        # Filter by confidence threshold
        conf_mask = confidences >= CONF_THRESHOLD
        boxes_xywh = boxes_xywh[conf_mask]
        confidences = confidences[conf_mask]
        class_ids = class_ids[conf_mask]

        regions = []
        annotated_pil = pil_img.copy()
        draw = ImageDraw.Draw(annotated_pil)

        if len(confidences) > 0:
            # Convert xywh in 640 space to xyxy
            x1 = np.clip(boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0, 0, IMAGE_SIZE)
            y1 = np.clip(boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0, 0, IMAGE_SIZE)
            x2 = np.clip(boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0, 0, IMAGE_SIZE)
            y2 = np.clip(boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0, 0, IMAGE_SIZE)
            boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

            # Non-Maximum Suppression
            keep_idx = nms(boxes_xyxy, confidences, IOU_THRESHOLD)

            for i, k in enumerate(keep_idx):
                conf_val = float(confidences[k])
                cls_id = int(class_ids[k])
                class_name = CLASSES.get(cls_id, f"class_{cls_id}")

                bx = boxes_xyxy[k]
                # Scale percentages relative to image dimensions
                pct_x = float((bx[0] / IMAGE_SIZE) * 100)
                pct_y = float((bx[1] / IMAGE_SIZE) * 100)
                pct_w = float(((bx[2] - bx[0]) / IMAGE_SIZE) * 100)
                pct_h = float(((bx[3] - bx[1]) / IMAGE_SIZE) * 100)

                regions.append({
                    "id": f"{upload.filename}-region-{i}",
                    "x": pct_x,
                    "y": pct_y,
                    "width": pct_w,
                    "height": pct_h,
                    "confidence": conf_val,
                    "className": class_name,
                })

                # Draw bounding box on annotated copy
                draw_x1 = (pct_x / 100.0) * orig_w
                draw_y1 = (pct_y / 100.0) * orig_h
                draw_x2 = ((pct_x + pct_w) / 100.0) * orig_w
                draw_y2 = ((pct_y + pct_h) / 100.0) * orig_h

                color = CLASS_COLORS.get(cls_id, (0, 229, 255))
                draw.rectangle([draw_x1, draw_y1, draw_x2, draw_y2], outline=color, width=3)
                label_txt = f"{class_name} {int(conf_val * 100)}%"
                draw.text((draw_x1 + 4, max(0, draw_y1 - 14)), label_txt, fill=color)

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
            "imageUrl": pil_to_data_url(pil_img),
            "annotatedUrl": pil_to_data_url(annotated_pil),
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
        "model": "YOLO Segmentation (best.onnx Cloud)",
    }

    return JSONResponse({"results": results_out, "summary": summary})


@app.get("/")
def health():
    return {"status": "ok", "model": "YOLO Segmentation (best.onnx Cloud)"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)

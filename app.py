import os
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


MODEL_PATH = "runs/detect/models/yolov8/weights/best.pt"
DATA_PATH = "SKU110K_small/images/val"

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5


run_id = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")

BASE_DIR = Path("demo_results")
RUN_DIR = BASE_DIR / run_id
IMG_DIR = RUN_DIR / "images"

RUN_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)


print("\nLoading model...")
model = YOLO(MODEL_PATH)
print("Model loaded")


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    return inter / (area_a + area_b - inter + 1e-6)


def nms(dets, thr=0.5):
    dets = sorted(dets, key=lambda x: x["conf"], reverse=True)
    keep = []

    for d in dets:
        if all(iou(d["bbox"], k["bbox"]) < thr for k in keep):
            keep.append(d)

    return keep



image_paths = list(Path(DATA_PATH).glob("*.jpg"))

print(f"Images: {len(image_paths)}")

results = []

start_all = time.time()



for i, img_path in enumerate(image_paths):

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    t0 = time.time()

    res = model(img, verbose=False)[0]

    raw = []

    if res.boxes is not None:
        for b in res.boxes:
            conf = float(b.conf)

            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, b.xyxy[0])

            raw.append({
                "bbox": [x1, y1, x2, y2],
                "conf": conf
            })

    final = nms(raw, IOU_THRESHOLD)

    count = len(final)


    for d in final:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(
        img,
        f"COUNT: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3
    )


    out_img = IMG_DIR / img_path.name
    cv2.imwrite(str(out_img), img)

    dt = time.time() - t0

    results.append({
        "image": img_path.name,
        "count": count,
        "raw": len(raw),
        "final": len(final),
        "latency_ms": dt * 1000
    })

    if i % 10 == 0:
        print(f"[{i}/{len(image_paths)}]")


total_time = time.time() - start_all



metrics = {
    "run_id": run_id,
    "total_images": len(image_paths),
    "avg_latency_ms": (total_time / len(image_paths)) * 1000,
    "total_time_sec": total_time,
    "results": results
}

json_path = RUN_DIR / "report.json"

with open(json_path, "w") as f:
    json.dump(metrics, f, indent=2)


pdf_path = RUN_DIR / "report.pdf"

styles = getSampleStyleSheet()
doc = SimpleDocTemplate(str(pdf_path))

content = []

content.append(Paragraph("Shelf Detection Report", styles["Title"]))
content.append(Spacer(1, 12))

content.append(Paragraph(f"Run ID: {run_id}", styles["Normal"]))
content.append(Paragraph(f"Images: {len(image_paths)}", styles["Normal"]))
content.append(Paragraph(f"Avg latency (ms): {metrics['avg_latency_ms']:.2f}", styles["Normal"]))
content.append(Spacer(1, 12))


avg_count = sum(r["count"] for r in results) / len(results)

content.append(Paragraph(f"Average detected objects: {avg_count:.2f}", styles["Normal"]))
content.append(Spacer(1, 12))


content.append(Paragraph("Per-image results:", styles["Heading2"]))

for r in results[:30]:
    content.append(
        Paragraph(
            f"{r['image']} | count={r['count']} | latency={r['latency_ms']:.1f} ms",
            styles["Normal"]
        )
    )

doc.build(content)


print("\n========================")
print("DONE")
print("========================")
print(f"Run folder: {RUN_DIR}")
print(f"JSON: {json_path}")
print(f"PDF: {pdf_path}")
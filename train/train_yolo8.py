from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=20,
    imgsz=640,
    batch=4,
    workers=2,
    device="cpu",
    optimizer="AdamW",
    lr0=0.001,
    project="models",
    name="yolov8"
)
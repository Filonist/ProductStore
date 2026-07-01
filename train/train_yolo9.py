from ultralytics import YOLO

model = YOLO("yolov9c.pt")

model.train(
    data="data.yaml",
    epochs=10,
    imgsz=640,
    batch=2,
    workers=2,
    device="cpu",
    project="models",
    name="yolov9",
    patience=5
)
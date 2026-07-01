from ultralytics import RTDETR

model = RTDETR("rtdetr-l.pt")

model.train(
    data="data.yaml",
    epochs=10,
    imgsz=320,
    batch=1,
    workers=2,
    device="cpu",
    project="models",
    name="rtdetr"
)
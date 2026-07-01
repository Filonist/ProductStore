import time
import torch

from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn

model = fasterrcnn_resnet50_fpn(weights=None)

model.load_state_dict(
    torch.load(
        "fasterrcnn_model.pth",
        map_location="cpu"
    )
)

model.eval()

img = Image.open(
    "SKU110K_small/images/val/val_100.jpg"
).convert("RGB")

img = F.to_tensor(img)

with torch.no_grad():

    start = time.time()

    _ = model([img])

    end = time.time()

print(
    f"Inference: {(end-start)*1000:.2f} ms"
)
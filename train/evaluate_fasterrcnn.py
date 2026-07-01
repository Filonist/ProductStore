import torch

from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from torchmetrics.detection.mean_ap import MeanAveragePrecision

DEVICE = "cpu"

class CocoToFasterRCNN(torch.utils.data.Dataset):

    def __init__(self, coco_dataset):
        self.ds = coco_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):

        img, anns = self.ds[idx]

        img = F.to_tensor(img)

        boxes = []
        labels = []

        for ann in anns:

            x, y, w, h = ann["bbox"]

            boxes.append([
                x,
                y,
                x + w,
                y + h
            ])

            labels.append(1)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


val_ds_raw = CocoDetection(
    root="SKU110K_small/images/val",
    annFile="SKU110K_small/annotations_val.json"
)

val_ds = CocoToFasterRCNN(val_ds_raw)

val_loader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0
)

print(f"Validation images: {len(val_ds)}")

model = fasterrcnn_resnet50_fpn(weights=None)

model.load_state_dict(
    torch.load(
        "fasterrcnn_model.pth",
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

metric = MeanAveragePrecision()

print("Running evaluation...")

with torch.no_grad():

    for images, targets in val_loader:

        images = [
            img.to(DEVICE)
            for img in images
        ]

        predictions = model(images)

        preds = []
        gts = []

        for pred, target in zip(predictions, targets):

            preds.append({
                "boxes": pred["boxes"].cpu(),
                "scores": pred["scores"].cpu(),
                "labels": pred["labels"].cpu()
            })

            gts.append({
                "boxes": target["boxes"],
                "labels": target["labels"]
            })

        metric.update(preds, gts)

results = metric.compute()

print("\n========== RESULTS ==========")

print(
    f"mAP50       : {results['map_50']:.4f}"
)

print(
    f"mAP50-95    : {results['map']:.4f}"
)

print(
    f"Recall      : {results['mar_100']:.4f}"
)

print("=============================")
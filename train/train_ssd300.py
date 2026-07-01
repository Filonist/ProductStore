import time
from pathlib import Path

import pandas as pd
import torch

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoDetection
from torchvision.models.detection import ssd300_vgg16
from torchvision.transforms import functional as F

DEVICE = "cpu"

EPOCHS = 5
TRAIN_IMAGES = 100
BATCH_SIZE = 2


class CocoToSSD(torch.utils.data.Dataset):

    def __init__(self, ds):
        self.ds = ds

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

            boxes = torch.tensor(
                boxes,
                dtype=torch.float32
            )

            labels = torch.tensor(
                labels,
                dtype=torch.int64
            )

        target = {
            "boxes": boxes,
            "labels": labels
        }

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


print("Loading dataset...")

train_ds_raw = CocoDetection(
    root="SKU110K_small/images/train",
    annFile="SKU110K_small/annotations_train.json"
)

train_ds = CocoToSSD(train_ds_raw)

train_ds = Subset(
    train_ds,
    range(TRAIN_IMAGES)
)

loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)

print("Dataset loaded")
print("Images:", len(train_ds))

print("Loading SSD300...")

model = ssd300_vgg16(
    weights="DEFAULT"
)

model.to(DEVICE)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001,
    momentum=0.9
)

history = []

print("Training started")

for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0

    start_epoch = time.time()

    for batch_idx, (images, targets) in enumerate(loader):

        images = [
            img.to(DEVICE)
            for img in images
        ]

        targets = [
            {
                k: v.to(DEVICE)
                for k, v in t.items()
            }
            for t in targets
        ]

        loss_dict = model(
            images,
            targets
        )

        loss = sum(
            loss_dict.values()
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        if batch_idx % 5 == 0:

            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Batch {batch_idx}/{len(loader)} | "
                f"Loss={loss.item():.4f}"
            )

    avg_loss = epoch_loss / len(loader)

    epoch_time = time.time() - start_epoch

    print(
        f"Epoch {epoch+1} finished | "
        f"Loss={avg_loss:.4f} | "
        f"Time={epoch_time:.2f}s"
    )

    history.append({
        "epoch": epoch + 1,
        "loss": avg_loss
    })

Path("runs/detect/models/ssd300").mkdir(
    parents=True,
    exist_ok=True
)

torch.save(
    model.state_dict(),
    "runs/detect/models/ssd300/best.pth"
)

pd.DataFrame(history).to_csv(
    "runs/detect/models/ssd300/results.csv",
    index=False
)

print("Training finished")
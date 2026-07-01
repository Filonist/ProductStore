import time
import torch

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F


DEVICE = torch.device("cpu")

TRAIN_IMAGES = 100
EPOCHS = 3
BATCH_SIZE = 1



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



print("\nLoading COCO dataset...")

train_ds_raw = CocoDetection(
    root="SKU110K_small/images/train",
    annFile="SKU110K_small/annotations_train.json"
)

print(f"Original dataset size: {len(train_ds_raw)} images")

train_ds = CocoToFasterRCNN(train_ds_raw)

subset_size = min(TRAIN_IMAGES, len(train_ds))

train_ds = Subset(
    train_ds,
    range(subset_size)
)

print(f"Using subset: {subset_size} images")

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)

print(f"Batches per epoch: {len(train_loader)}")


print("\nLoading Faster R-CNN...")

model = fasterrcnn_resnet50_fpn(
    weights="DEFAULT"
)

model.to(DEVICE)

print("Model loaded successfully")


optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)


print("\nTraining started...\n")

training_start = time.time()

for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0.0
    epoch_start = time.time()

    print("=" * 60)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("=" * 60)

    for batch_idx, (images, targets) in enumerate(train_loader):

        batch_start = time.time()

        images = [
            img.to(DEVICE)
            for img in images
        ]

        targets = [
            {
                k: v.to(DEVICE)
                for k, v in target.items()
            }
            for target in targets
        ]

        loss_dict = model(
            images,
            targets
        )

        loss = sum(
            loss_item
            for loss_item in loss_dict.values()
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()

        epoch_loss += loss_value

        batch_time = time.time() - batch_start

        if batch_idx % 5 == 0:

            print(
                f"[Epoch {epoch+1}] "
                f"Batch {batch_idx+1}/{len(train_loader)} | "
                f"Loss={loss_value:.4f} | "
                f"Time={batch_time:.2f}s"
            )

    avg_loss = epoch_loss / len(train_loader)

    epoch_time = time.time() - epoch_start

    print("\nEpoch summary")
    print(f"Average loss : {avg_loss:.4f}")
    print(f"Epoch time   : {epoch_time:.2f} sec")
    print()


torch.save(
    model.state_dict(),
    "fasterrcnn_model.pth"
)

total_time = time.time() - training_start

print("=" * 60)
print("Training completed")
print("=" * 60)

print(f"Total training time: {total_time:.2f} sec")
print("Saved: fasterrcnn_model.pth")
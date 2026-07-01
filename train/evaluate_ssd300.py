import time
import torch

from torchvision.models.detection import ssd300_vgg16
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Subset



DEVICE = "cpu"
IOU_THR = 0.5
CONF_THR = 0.3
MAX_IMAGES = 100


def collate_fn(batch):
    return tuple(zip(*batch))



def iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter + 1e-6

    return inter / union



print("Loading model...")

model = ssd300_vgg16(weights=None)

model.load_state_dict(
    torch.load("runs/detect/models/ssd300/best.pth", map_location=DEVICE)
)

model.to(DEVICE)
model.eval()

print("Model loaded")



print("Loading dataset...")

ds = CocoDetection(
    root="SKU110K_small/images/val",
    annFile="SKU110K_small/annotations_val.json"
)

ds = Subset(ds, range(min(MAX_IMAGES, len(ds))))

loader = DataLoader(
    ds,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn
)

print(f"Images: {len(ds)}")



tp = 0
fp = 0
fn = 0
latencies = []



with torch.no_grad():

    for images, targets in loader:

        img = F.to_tensor(images[0]).to(DEVICE)

        start = time.time()
        outputs = model([img])
        end = time.time()

        latencies.append(end - start)

        pred = outputs[0]

        boxes = pred["boxes"]
        scores = pred["scores"]

        keep = scores > CONF_THR
        boxes = boxes[keep].cpu().numpy()

        gt_boxes = []

        for ann in targets[0]:

            x, y, w, h = ann["bbox"]

            gt_boxes.append([x, y, x + w, y + h])

        matched = set()

        for pb in boxes:

            best_iou = 0
            best_idx = -1

            for i, gb in enumerate(gt_boxes):

                if i in matched:
                    continue

                score = iou(pb, gb)

                if score > best_iou:
                    best_iou = score
                    best_idx = i

            if best_iou >= IOU_THR:

                tp += 1
                matched.add(best_idx)

            else:

                fp += 1

        fn += len(gt_boxes) - len(matched)



precision = tp / (tp + fp + 1e-6)
recall = tp / (tp + fn + 1e-6)
avg_latency = sum(latencies) / len(latencies) * 1000

print("\n========== RESULTS ==========")
print(f"TP: {tp} | FP: {fp} | FN: {fn}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"mAP-like    : {(precision + recall)/2:.4f}")
print(f"Avg latency : {avg_latency:.2f} ms")
print("=============================")
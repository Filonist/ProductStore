import json
from pathlib import Path
from PIL import Image

ROOT = Path("SKU110K_small")


def convert(split):
    images_dir = ROOT / "images" / split
    labels_dir = ROOT / "labels" / split

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "product"
            }
        ]
    }

    ann_id = 0

    image_files = sorted(images_dir.glob("*.jpg"))

    print(f"{split}: found {len(image_files)} images")

    for img_id, img_path in enumerate(image_files):

        img = Image.open(img_path)
        width, height = img.size

        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": width,
            "height": height
        })

        label_path = labels_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            continue

        with open(label_path) as f:

            for line in f:

                cls, x, y, bw, bh = map(
                    float,
                    line.strip().split()
                )

                x_min = (x - bw / 2) * width
                y_min = (y - bh / 2) * height

                bw_px = bw * width
                bh_px = bh * height

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [
                        x_min,
                        y_min,
                        bw_px,
                        bh_px
                    ],
                    "area": bw_px * bh_px,
                    "iscrowd": 0
                })

                ann_id += 1

    output_file = ROOT / f"annotations_{split}.json"

    with open(output_file, "w") as f:
        json.dump(coco, f)

    print(
        f"{split}: "
        f"{len(coco['images'])} images, "
        f"{len(coco['annotations'])} annotations"
    )


convert("train")
convert("val")
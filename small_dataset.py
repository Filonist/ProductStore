from pathlib import Path
import random
import shutil

random.seed(42)

SOURCE = Path("SKU110K_fixed")
DEST = Path("SKU110K_small")

sizes = {
    "train": 1000,
    "val": 200,
    "test": 200
}

for split in ["train", "val", "test"]:

    image_dir = SOURCE / "images" / split
    label_dir = SOURCE / "labels" / split

    target_image_dir = DEST / "images" / split
    target_label_dir = DEST / "labels" / split

    target_image_dir.mkdir(parents=True, exist_ok=True)
    target_label_dir.mkdir(parents=True, exist_ok=True)

    images = list(image_dir.glob("*"))

    selected = random.sample(images, sizes[split])

    for image_path in selected:

        label_path = label_dir / (image_path.stem + ".txt")

        shutil.copy2(image_path, target_image_dir / image_path.name)

        if label_path.exists():
            shutil.copy2(label_path,
                         target_label_dir / label_path.name)

print("Done")
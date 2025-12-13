import multiprocessing
from itertools import repeat
import h5py
import json
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from utils.data import HatefulMemeDataset

DATA_DIR = "data/img"


def process_single_image(img_path, img_size):
    try:
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (img_size, img_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                )
            ]
        )

        img = Image.open(img_path).convert("RGB")
        img_resized = transform(img)

        img_array = np.array(img_resized, dtype=np.uint8)
        img_array = img_array.transpose(2, 0, 1)

        return int(img_path.stem), img_array

    except Exception:
        return None


def preprocess(json_path, save_path, img_size):
    valid_ids = set()
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                valid_ids.add(int(json.loads(line)["id"]))

    all_paths = Path(DATA_DIR).glob("*.png")
    image_paths = [p for p in all_paths if int(p.stem) in valid_ids]
    image_paths = sorted(image_paths)

    total_imgs = len(image_paths)
    num_workers = max(1, multiprocessing.cpu_count() - 2)

    print(
        f"Processing {total_imgs} images (filtered by JSONL) with {num_workers} workers..."
    )

    with h5py.File(save_path, "w") as f:
        dset_imgs = f.create_dataset(
            "images", (total_imgs, 3, img_size, img_size), dtype="uint8"
        )
        dset_ids = f.create_dataset("ids", (total_imgs,), dtype=int)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(
                process_single_image, image_paths, repeat(img_size), chunksize=32
            )

            valid_idx = 0
            for result in tqdm(results, total=total_imgs):
                if result is not None:
                    file_id, img_array = result
                    dset_imgs[valid_idx] = img_array
                    dset_ids[valid_idx] = file_id
                    valid_idx += 1

            if valid_idx < total_imgs:
                dset_imgs.resize((valid_idx, 3, img_size, img_size))
                dset_ids.resize((valid_idx,))

    print(f"Saved {valid_idx} images to {save_path}")


def run_preprocess(img_size):
    for split in ["train", "val", "test"]:
        preprocess(f"data/{split}.jsonl", f"data/resized/{split}.h5", img_size)


if __name__ == "__main__":
    # Preprocess
    run_preprocess(img_size=336)

    # Visualization
    import matplotlib.pyplot as plt

    H5_PATH = "data/resized/test.h5"
    JSON_PATH = "data/test.jsonl"

    dataset = HatefulMemeDataset(H5_PATH, JSON_PATH, do_augment=False)

    print(f"Dataset Size: {len(dataset)}")
    idx = 0
    sample = dataset[idx]

    pixel_values = sample["pixel_values"]  # Tensor (3, w, h)
    text = sample["text"]
    label = sample["label"]

    print(f"\n--- Sample {idx} ---")
    print(f"Text: {text}")
    print(f"Label: {label.item()} (0: Non-hateful, 1: Hateful)")
    print(f"Tensor Shape: {pixel_values.shape}")

    plt.figure(figsize=(6, 6))

    image_numpy = pixel_values.permute(1, 2, 0).numpy()
    plt.imshow(image_numpy)
    plt.title(f"Label: {label.item()}")
    plt.axis("off")
    plt.show()

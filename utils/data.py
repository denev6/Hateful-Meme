import multiprocessing
import h5py
import json
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

DATA_DIR = "data/img"
IMG_SIZE = 224


def process_single_image(img_path):
    try:
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (IMG_SIZE, IMG_SIZE),
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


def preprocess(json_path, save_path):
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
            "images", (total_imgs, 3, IMG_SIZE, IMG_SIZE), dtype="uint8"
        )
        dset_ids = f.create_dataset("ids", (total_imgs,), dtype=int)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(process_single_image, image_paths, chunksize=32)

            valid_idx = 0
            for result in tqdm(results, total=total_imgs):
                if result is not None:
                    file_id, img_array = result
                    dset_imgs[valid_idx] = img_array
                    dset_ids[valid_idx] = file_id
                    valid_idx += 1

            if valid_idx < total_imgs:
                dset_imgs.resize((valid_idx, 3, IMG_SIZE, IMG_SIZE))
                dset_ids.resize((valid_idx,))

    print(f"Saved {valid_idx} images to {save_path}")


def run_preprocess():
    for split in ["train", "val", "test"]:
        preprocess(f"data/{split}.jsonl", f"data/resized/{split}.h5")


class HatefulMemeDataset(Dataset):
    def __init__(self, h5_path, json_path, do_augment=True):
        self.h5_path = h5_path

        self.data = []
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    item["id"] = str(item["id"])
                    self.data.append(item)

        with h5py.File(h5_path, "r") as f:
            h5_ids = f["ids"][:]
            self.id_to_h5_idx = {str(x): i for i, x in enumerate(h5_ids)}

        self.h5_file = None
        self.images = None

        if do_augment:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.TrivialAugmentWide(),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
            self.images = self.h5_file["images"]

        item = self.data[idx]
        img_id = item["id"]
        text = item["text"]
        label = item["label"]

        h5_idx = self.id_to_h5_idx.get(img_id)
        if h5_idx is None:
            raise ValueError(f"Image ID {img_id} not found in HDF5.")

        img_array = self.images[h5_idx]
        img_array = img_array.transpose(1, 2, 0)  # CHW -> HWC for PIL

        pixel_values = self.transform(img_array)

        return {
            "pixel_values": pixel_values,
            "text": text,
            "label": torch.tensor(label, dtype=torch.long),
        }


if __name__ == "__main__":
    # Preprocess
    # run_preprocess()

    # Visualization
    import matplotlib.pyplot as plt

    H5_PATH = "data/resized/test.h5"
    JSON_PATH = "data/test.jsonl"

    dataset = HatefulMemeDataset(H5_PATH, JSON_PATH, do_augment=False)

    print(f"Dataset Size: {len(dataset)}")
    idx = 0
    sample = dataset[idx]

    pixel_values = sample["pixel_values"]  # Tensor (3, 224, 224)
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

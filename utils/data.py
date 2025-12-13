import h5py
import json
from torchvision import transforms
import torch
from torch.utils.data import Dataset


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

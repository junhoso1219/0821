from __future__ import annotations
import os
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

TINY_MEAN = (0.4802, 0.4481, 0.3975)
TINY_STD  = (0.2302, 0.2265, 0.2262)

class _TinyImageNetTrain(Dataset):
    def __init__(self, root: str, transform):
        self.root = os.path.join(root, "tiny-imagenet-200", "train")
        self.transform = transform
        # build class list
        self.classes = sorted([d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        items: List[Tuple[str, int]] = []
        for c in self.classes:
            img_dir = os.path.join(self.root, c, "images")
            for fn in os.listdir(img_dir):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    items.append((os.path.join(img_dir, fn), self.class_to_idx[c]))
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx: int):
        fp, y = self.items[idx]
        img = Image.open(fp).convert("RGB")
        return self.transform(img), y

class _TinyImageNetVal(Dataset):
    def __init__(self, root: str, transform):
        self.root = os.path.join(root, "tiny-imagenet-200", "val")
        self.transform = transform
        ann_path = os.path.join(self.root, "val_annotations.txt")
        # read mapping filename -> wnid
        mapping = {}
        with open(ann_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    mapping[parts[0]] = parts[1]
        classes = sorted(list(set(mapping.values())))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        items: List[Tuple[str, int]] = []
        img_dir = os.path.join(self.root, "images")
        for fn, wnid in mapping.items():
            fp = os.path.join(img_dir, fn)
            if os.path.isfile(fp):
                items.append((fp, self.class_to_idx[wnid]))
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx: int):
        fp, y = self.items[idx]
        img = Image.open(fp).convert("RGB")
        return self.transform(img), y

def get_tiny_imagenet_loaders(root: str = "./data",
                              batch_size: int = 128,
                              num_workers: int = 4,
                              aug: bool = True):
    if aug:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(TINY_MEAN, TINY_STD),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(TINY_MEAN, TINY_STD),
        ])
    test_tf = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(TINY_MEAN, TINY_STD),
    ])

    train_ds = _TinyImageNetTrain(root, train_tf)
    val_ds   = _TinyImageNetVal(root, test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader



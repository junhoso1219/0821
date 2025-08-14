from __future__ import annotations
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

class _DummyCIFAR(Dataset):
    def __init__(self, n: int = 1024, num_classes: int = 10, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 3, 32, 32, generator=g)
        self.y = torch.randint(0, num_classes, (n,), generator=g)
        self.tf = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, i):
        return self.tf(self.x[i]), self.y[i]

def get_cifar10_loaders(root: str = "./data",
                        batch_size: int = 256,
                        num_workers: int = 4,
                        aug: bool = True,
                        dummy: bool = False,
                        dummy_size: int = 1024,
                        seed: int = 0):
    if dummy:
        train_ds = _DummyCIFAR(n=dummy_size, seed=seed)
        test_ds  = _DummyCIFAR(n=max(256, dummy_size//4), seed=seed+1)
    else:
        if aug:
            train_tf = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ])
        else:
            train_tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

        train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=train_tf)
        test_ds  = datasets.CIFAR10(root=root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

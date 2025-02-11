import os
import numpy as np

from PIL import Image

from torch.utils.data import Dataset


class CIFAR10(Dataset):
    def __init__(self, root: str, train: bool, transform=None) -> None:

        self.root = root

        self.train = train

        self.transform = transform

        if train:
            self.data = np.load(os.path.join(self.root, "train_data.npy"))
            self.targets = np.load(os.path.join(self.root, "train_targets.npy"))

        else:
            self.data = np.load(os.path.join(self.root, "test_data.npy"))
            self.targets = np.load(os.path.join(self.root, "test_targets.npy"))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img, mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target
    
    

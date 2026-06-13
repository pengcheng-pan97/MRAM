"""ICONIP — data loaders for MNIST, FashionMNIST, FER2013.

FER2013 should be unpacked under `data/fer2013/{train,test}/<class>/*.jpg`
(use any standard FER2013 distribution; the directory layout is the only
contract).
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

from utils import plot_images


def _transforms():
    mnist_norm = transforms.Normalize((0.1307,), (0.3081,))
    fer_norm = transforms.Normalize((0.485,), (0.229,))
    return {
        "MNIST":        transforms.Compose([transforms.ToTensor(), mnist_norm]),
        "FashionMNIST": transforms.Compose([transforms.ToTensor(), mnist_norm]),
        "FER":          transforms.Compose([transforms.ToTensor(), fer_norm]),
    }


def _make_dataset(data_choose, data_dir, train):
    tfs = _transforms()
    if data_choose == "MNIST":
        return datasets.MNIST(data_dir, train=train, download=True, transform=tfs["MNIST"])
    if data_choose == "FashionMNIST":
        return datasets.FashionMNIST(data_dir, train=train, download=True, transform=tfs["FashionMNIST"])
    if data_choose == "FER":
        split = "train" if train else "test"
        return datasets.ImageFolder(root=f"data/fer2013/{split}", transform=tfs["FER"])
    raise ValueError(f"Unsupported dataset {data_choose!r}; this release supports MNIST / FashionMNIST / FER.")


def get_train_valid_loader(
    data_choose,
    data_dir,
    batch_size,
    random_seed,
    valid_size=0.1,
    shuffle=True,
    show_sample=False,
    num_workers=4,
    pin_memory=False,
):
    """Train + validation loaders. `valid_size` slices the train set."""
    assert 0.0 <= valid_size <= 1.0, "valid_size must be in [0, 1]"

    dataset = _make_dataset(data_choose, data_dir, train=True)

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_idx),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(valid_idx),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if show_sample:
        sample_loader = DataLoader(
            dataset, batch_size=9, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        images, labels = next(iter(sample_loader))
        X = np.transpose(images.numpy(), [0, 2, 3, 1])
        plot_images(X, labels)

    return train_loader, valid_loader


def get_test_loader(data_choose, data_dir, batch_size, num_workers=4, pin_memory=False):
    """Test loader."""
    dataset = _make_dataset(data_choose, data_dir, train=False)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

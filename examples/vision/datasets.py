"""Functions for getting computer vision datasets."""

from __future__ import annotations

import os
from typing import Any
from typing import Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
from torchvision import transforms

T = Tuple[torch.Tensor, torch.Tensor]


def get_cifar(
    args: Any,
) -> tuple[
    DistributedSampler[T],
    DataLoader[T],
    DistributedSampler[T],
    DataLoader[T],
]:
    """Get cifar dataset."""
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ],
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ],
    )

    os.makedirs(args.data_dir, exist_ok=True)

    download = True if args.local_rank == 0 else False
    if not download:
        dist.barrier()
    train_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=download,
        transform=transform_train,
    )
    test_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=download,
        transform=transform_test,
    )
    if download:
        dist.barrier()

    return make_sampler_and_loader(args, train_dataset, test_dataset)


def get_imagenet(
    args: Any,
) -> tuple[
    DistributedSampler[T],
    DataLoader[T],
    DistributedSampler[T],
    DataLoader[T],
]:
    """Get imagenet dataset."""
    train_dataset = datasets.ImageFolder(
        args.train_dir,
        transform=transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ],
        ),
    )
    val_dataset = datasets.ImageFolder(
        args.val_dir,
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ],
        ),
    )

    return make_sampler_and_loader(args, train_dataset, val_dataset)


def make_sampler_and_loader(
    args: Any,
    train_dataset: Dataset[T],
    val_dataset: Dataset[T],
) -> tuple[
    DistributedSampler[T],
    DataLoader[T],
    DistributedSampler[T],
    DataLoader[T],
]:
    """Create sampler and dataloader for train and val datasets."""
    torch.set_num_threads(4)
    kwargs: dict[str, Any] = (
        {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    )
    kwargs['prefetch_factor'] = 8
    kwargs['persistent_workers'] = True

    train_sampler: DistributedSampler[T] = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
    )
    train_loader: DataLoader[T] = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        **kwargs,
    )
    val_sampler: DistributedSampler[T] = DistributedSampler(
        val_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
    )
    val_loader: DataLoader[T] = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        sampler=val_sampler,
        **kwargs,
    )

    return train_sampler, train_loader, val_sampler, val_loader

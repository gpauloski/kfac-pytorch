import os
import kfac
import torch
from torchvision import datasets, transforms

def get_cifar(args):
    torch.set_num_threads(4)

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    os.makedirs(args.data_dir, exist_ok=True)

    download = True if args.backend.local_rank() == 0 else False
    if not download: args.backend.barrier()
    train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, 
                                     download=download, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=args.data_dir, train=False,
                                    download=download, transform=transform_test)
    if download: args.backend.barrier()

    train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.backend.size(), rank=args.backend.rank())
    train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=args.batch_size, sampler=train_sampler, **kwargs)

    test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=args.backend.size(), rank=args.backend.rank())
    test_loader = torch.utils.data.DataLoader(test_dataset,
            batch_size=args.val_batch_size, sampler=test_sampler, **kwargs)

    return train_sampler, train_loader, test_sampler, test_loader

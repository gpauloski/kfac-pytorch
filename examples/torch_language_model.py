"""Language modeling with Transformers Example.

Based on the PyTorch example and modified for distributed training with KFAC:
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

import torch
from torch.utils import collect_env

from examples.language.dataset import get_dataset
from examples.language.engine import evaluate
from examples.language.engine import train
from examples.language.transformer import TransformerModel


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description='Language Modeling Example',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument(
        '--embedding-dim',
        default=200,
        type=int,
        help='embedding dimension size',
    )
    model_group.add_argument(
        '--hidden-dim',
        default=200,
        type=int,
        help='hidden dimension size',
    )
    model_group.add_argument(
        '--attention-heads',
        default=2,
        type=int,
        help='number of attention heads',
    )
    model_group.add_argument(
        '--layers',
        default=2,
        type=int,
        help='number of layers',
    )
    model_group.add_argument(
        '--dropout',
        default=0.2,
        type=float,
        help='dropout probability',
    )

    data_group = parser.add_argument_group('Data Parameters')
    data_group.add_argument(
        '--dataset',
        choices=['penntreebank', 'wikitext2', 'wikitext103'],
        help='dataset to train language model on',
    )
    data_group.add_argument(
        '--download-dir',
        default='/tmp/torchtext-data',
        help='directory to download dataset to',
    )
    data_group.add_argument(
        '--seq-len',
        default=35,
        type=int,
        help='number of tokens in each training sample',
    )
    data_group.add_argument(
        '--batch-size',
        default=20,
        type=int,
        help='batch size',
    )

    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument(
        '--epochs',
        default=3,
        type=int,
        help='training epochs',
    )
    training_group.add_argument(
        '--lr',
        default=5.0,
        type=float,
        help='initial learning rate',
    )
    training_group.add_argument(
        '--backend',
        choices=['gloo', 'mpi', 'nccl'],
        default='nccl',
        help='distributed training backend',
    )
    training_group.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disable CUDA training',
    )
    training_group.add_argument(
        '--seed',
        default=42,
        type=int,
        help='training seed',
    )

    args = parser.parse_args(argv)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = 'cuda' if args.cuda else 'cpu'
    args.local_rank = int(os.environ['LOCAL_RANK'])

    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Train and validate a language model."""
    args = parse_args(argv)

    torch.distributed.init_process_group(
        backend=args.backend,
        init_method='env://',
    )

    if args.cuda:
        torch.cuda.set_device(args.local_rank)
        torch.cuda.manual_seed(args.seed)

    if torch.distributed.get_rank() == 0:
        print('Collecting env info...')
        print(collect_env.get_pretty_env_info())

    datasets, vocab = get_dataset(
        args.dataset,
        args.download_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        cuda=args.cuda,
        rank=torch.distributed.get_rank(),
        world_size=torch.distributed.get_world_size(),
    )

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=args.embedding_dim,
        nhead=args.attention_heads,
        d_hid=args.hidden_dim,
        nlayers=args.layers,
        dropout=args.dropout,
    )
    model.to(args.device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank] if args.cuda else None,
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    for epoch in range(args.epochs):
        datasets.train.sampler.set_epoch(epoch)
        train(
            model,
            criterion=criterion,
            optimizer=optimizer,
            dataloader=datasets.train.loader,
            epoch=epoch,
            epochs=args.epochs,
        )
        evaluate(
            model,
            criterion=criterion,
            dataloader=datasets.val.loader,
            prefix='Validation',
        )
        scheduler.step()

    evaluate(
        model,
        criterion=criterion,
        dataloader=datasets.test.loader,
        prefix='Test',
    )

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

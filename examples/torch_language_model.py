"""Language modeling with Transformers Example.

Based on the PyTorch example and modified for distributed training with KFAC:
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Sequence

import torch
from torch.utils import collect_env

import kfac
from examples.language.dataset import get_dataset
from examples.language.engine import evaluate
from examples.language.engine import train
from examples.language.transformer import TransformerModel

logger = logging.getLogger(__name__)


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
        default=256,
        type=int,
        help='embedding dimension size',
    )
    model_group.add_argument(
        '--hidden-dim',
        default=256,
        type=int,
        help='hidden dimension size',
    )
    model_group.add_argument(
        '--attention-heads',
        default=4,
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
        default='penntreebank',
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
        default=64,
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
        default=20,
        type=int,
        help='training epochs',
    )
    training_group.add_argument(
        '--lr',
        default=1.0,
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

    kfac_group = parser.add_argument_group('KFAC Parameters')
    kfac_group.add_argument(
        '--kfac',
        action='store_true',
        default=False,
        help='enable KFAC preconditioning',
    )
    kfac_group.add_argument(
        '--inv-update-steps',
        type=int,
        default=10,
        help='iters between updating second-order information',
    )
    kfac_group.add_argument(
        '--factor-update-steps',
        type=int,
        default=1,
        help='iters between update kronecker factors',
    )
    kfac_group.add_argument(
        '--factor-decay',
        type=float,
        default=0.95,
        help='alpha value for factor accumulation',
    )
    kfac_group.add_argument(
        '--damping',
        type=float,
        default=0.003,
        help='damping factor',
    )
    kfac_group.add_argument(
        '--kl-clip',
        type=float,
        default=0.001,
        help='KL clip',
    )
    kfac_group.add_argument(
        '--skip-layers',
        nargs='+',
        type=str,
        default=['embedding', 'decoder', 'self_attn'],
        help='layers to skip KFAC registration for',
    )
    kfac_group.add_argument(
        '--strategy',
        choices=['MEM_OPT', 'HYBRID_OPT', 'COMM_OPT'],
        default='COMM_OPT',
        help='distribution strategy for KFAC computations',
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

    logging.basicConfig(
        format='[%(asctime)s] %(levelname)-5s (%(name)s): %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
        if torch.distributed.get_rank() == 0
        else logging.ERROR,
        stream=sys.stdout,
    )

    if args.cuda:
        torch.cuda.set_device(args.local_rank)
        torch.cuda.manual_seed(args.seed)

    if torch.distributed.get_rank() == 0:
        logger.info('Collecting env info...')
        logger.info(collect_env.get_pretty_env_info())
        logger.info(f'Training arguments:\n{args}')

    datasets, vocab = get_dataset(
        args.dataset,
        args.download_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        cuda=args.cuda,
        rank=torch.distributed.get_rank(),
        world_size=torch.distributed.get_world_size(),
    )

    model: torch.nn.Module = TransformerModel(
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.1,
        patience=2,
        min_lr=1e-4,
    )

    logger.info(f'Transformer model:\n{model}')
    preconditioner: kfac.preconditioner.KFACPreconditioner | None = None
    if args.kfac:
        strategy = kfac.enums.DistributedStrategy[args.strategy.upper()]
        preconditioner = kfac.preconditioner.KFACPreconditioner(
            model,
            factor_update_steps=args.factor_update_steps,
            inv_update_steps=args.inv_update_steps,
            damping=args.damping,
            factor_decay=args.factor_decay,
            kl_clip=args.kl_clip,
            lr=lambda x: optimizer.param_groups[0]['lr'],
            grad_worker_fraction=strategy,
            skip_layers=args.skip_layers,
            loglevel=logging.INFO,
        )
        if torch.distributed.get_rank() == 0:
            logger.info(f'Preconditioner config:\n{preconditioner}')

    start = time.perf_counter()
    for epoch in range(args.epochs):
        datasets.train.sampler.set_epoch(epoch)
        train(
            model,
            criterion=criterion,
            optimizer=optimizer,
            preconditioner=preconditioner,
            dataloader=datasets.train.loader,
            epoch=epoch + 1,
            epochs=args.epochs,
        )
        eval_loss = evaluate(
            model,
            criterion=criterion,
            dataloader=datasets.val.loader,
            prefix='Validation',
        )
        scheduler.step(eval_loss)
    end = time.perf_counter()
    logger.info(f'Training completed in {end - start:.2f} seconds.')

    evaluate(
        model,
        criterion=criterion,
        dataloader=datasets.test.loader,
        prefix='Test',
    )

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

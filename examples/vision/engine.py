"""Train and Eval functions for computer vision examples."""

from __future__ import annotations

import argparse
import math
from typing import Tuple

import torch
from tqdm import tqdm

import kfac
from examples.utils import accuracy
from examples.utils import Metric

SampleT = Tuple[torch.Tensor, torch.Tensor]


def train(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    preconditioner: kfac.preconditioner.KFACPreconditioner | None,
    loss_func: torch.nn.Module,
    train_sampler: torch.utils.data.distributed.DistributedSampler[SampleT],
    train_loader: torch.utils.data.DataLoader[SampleT],
    args: argparse.Namespace,
) -> None:
    """Train model."""
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    scaler = args.grad_scaler if 'grad_scaler' in args else None
    mini_step = 0
    step_loss = torch.tensor(0.0).to('cuda' if args.cuda else 'cpu')
    step_accuracy = torch.tensor(0.0).to('cuda' if args.cuda else 'cpu')

    with tqdm(
        total=math.ceil(len(train_loader) / args.batches_per_allreduce),
        bar_format='{l_bar}{bar:10}{r_bar}',
        desc=f'Epoch {epoch:3d}/{args.epochs:3d}',
        disable=not args.verbose,
    ) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            mini_step += 1
            if args.cuda:
                data, target = data.cuda(), target.cuda()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = loss_func(output, target)
            else:
                output = model(data)
                loss = loss_func(output, target)

            with torch.no_grad():
                step_loss += loss
                step_accuracy += accuracy(output, target)

            loss = loss / args.batches_per_allreduce

            if mini_step % args.batches_per_allreduce == 0 or (
                batch_idx + 1 == len(train_loader)
            ):
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                with model.no_sync():  # type: ignore
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

            if mini_step % args.batches_per_allreduce == 0 or (
                batch_idx + 1 == len(train_loader)
            ):
                if preconditioner is not None:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    preconditioner.step()
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                train_loss.update(step_loss / mini_step)
                train_accuracy.update(step_accuracy / mini_step)
                step_loss.zero_()
                step_accuracy.zero_()

                t.set_postfix_str(
                    'loss: {:.4f}, acc: {:.2f}%, lr: {:.4f}'.format(
                        train_loss.avg,
                        100 * train_accuracy.avg,
                        optimizer.param_groups[0]['lr'],
                    ),
                )
                t.update(1)
                mini_step = 0

    if args.log_writer is not None:
        args.log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        args.log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)
        args.log_writer.add_scalar(
            'train/lr',
            optimizer.param_groups[0]['lr'],
            epoch,
        )


def test(
    epoch: int,
    model: torch.nn.Module,
    loss_func: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader[SampleT],
    args: argparse.Namespace,
) -> None:
    """Test the model."""
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    with tqdm(
        total=len(val_loader),
        bar_format='{l_bar}{bar:10}|{postfix}',
        desc='             ',
        disable=not args.verbose,
    ) as t:
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                val_loss.update(loss_func(output, target))
                val_accuracy.update(accuracy(output, target))

                t.update(1)
                if i + 1 == len(val_loader):
                    t.set_postfix_str(
                        '\b\b val_loss: {:.4f}, val_acc: {:.2f}%'.format(
                            val_loss.avg,
                            100 * val_accuracy.avg,
                        ),
                        refresh=False,
                    )

    if args.log_writer is not None:
        args.log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        args.log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)

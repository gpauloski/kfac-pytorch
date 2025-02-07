"""Training and eval functions for the language modeling example."""

from __future__ import annotations

from typing import Tuple

import torch
from tqdm import tqdm

import kfac
from examples.language.transformer import gen_square_subsequent_mask
from examples.utils import Metric

DType = Tuple[torch.Tensor, torch.Tensor]


def train(
    model: torch.nn.Module,
    *,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    preconditioner: kfac.base_preconditioner.BaseKFACPreconditioner | None,
    dataloader: torch.utils.data.DataLoader[DType],
    epoch: int,
    epochs: int,
) -> float:
    """Perform one training epoch."""
    model.train()
    train_loss = Metric('train_loss')
    src_mask: torch.Tensor | None = None

    with tqdm(
        total=len(dataloader),
        bar_format='{l_bar}{bar:8}{r_bar}',
        desc=f'Epoch {epoch:2d}/{epochs:2d}',
        disable=torch.distributed.get_rank() > 0,
    ) as t:
        for data, target in dataloader:
            if src_mask is None:
                seq_len = data.size(0)
                device = next(model.parameters()).device
                src_mask = gen_square_subsequent_mask(seq_len).to(device)

            optimizer.zero_grad()

            data = data.to(model.device)
            target = target.to(model.device).reshape(-1)

            output = model(data, src_mask)
            ntokens = output.size(-1)
            output_flat = output.view(-1, ntokens)

            loss = criterion(output_flat, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            if preconditioner is not None:
                preconditioner.step()
            optimizer.step()

            loss = loss.detach()
            train_loss.update(loss)

            t.set_postfix_str(
                'loss: {:.2f}, ppl: {:.2f}, lr: {:.1E}'.format(
                    train_loss.avg,
                    torch.exp(train_loss.avg),
                    optimizer.param_groups[0]['lr'],
                ),
            )
            t.update(1)

    return train_loss.avg.item()


def evaluate(
    model: torch.nn.Module,
    *,
    criterion: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader[DType],
    prefix: str,
) -> float:
    """Evaluate model."""
    model.eval()
    eval_loss = Metric('eval_loss')
    src_mask: torch.Tensor | None = None

    with (
        torch.no_grad(),
        tqdm(
            total=len(dataloader),
            bar_format='{l_bar}{bar:8}{r_bar}',
            desc=prefix[:11].ljust(11, ' '),
            disable=torch.distributed.get_rank() > 0,
        ) as t,
    ):
        for data, target in dataloader:
            if src_mask is None:
                seq_len = data.size(0)
                device = next(model.parameters()).device
                src_mask = gen_square_subsequent_mask(seq_len).to(device)

            data = data.to(model.device)
            target = target.to(model.device).reshape(-1)

            output = model(data, src_mask)
            ntokens = output.size(-1)
            output_flat = output.view(-1, ntokens)

            loss = criterion(output_flat, target)
            loss = loss.detach()
            eval_loss.update(loss)

            t.update(1)
            t.set_postfix_str(
                'loss: {:.2f}, ppl: {:.2f}'.format(
                    eval_loss.avg,
                    torch.exp(eval_loss.avg),
                ),
                refresh=False,
            )

    return eval_loss.avg.item()

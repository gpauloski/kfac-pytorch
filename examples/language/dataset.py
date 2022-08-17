"""Language modeling datasets."""
from __future__ import annotations

from typing import Callable
from typing import Literal
from typing import NamedTuple

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataset import IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import PennTreebank
from torchtext.datasets import WikiText103
from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import Vocab


class LoaderSampler(NamedTuple):
    """Tuple of a dataloader and the corresponding datasampler."""

    loader: DataLoader
    sampler: DistributedSampler


class Datasets(NamedTuple):
    """Train/val/test tuple of LoaderSamplers."""

    train: LoaderSampler
    val: LoaderSampler
    test: LoaderSampler


class _Dataset(Dataset):
    def __init__(self, data: torch.Tensor, seq_len: int) -> None:
        self._data = data
        self._seq_len = seq_len

    def __len__(self) -> int:
        return len(self._data) // self._seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self._seq_len * idx
        end = self._seq_len * (idx + 1)
        data = self._data[start:end]
        target = self._data[start + 1 : end + 1]
        return data, target


def download_dataset(
    dataset: Literal['penntreebank', 'wikitext2', 'wikitext103'],
    data_dir: str,
) -> tuple[IterableDataset, IterableDataset, IterableDataset]:
    """Get a torchtext language modeling dataset.

    Args:
        dataset (str): one of 'penntreebank', 'wikitext2', or 'wikitext102'.
        data_dir (str): directory to download datasets to.

    Returns:
        tuple of train, validation, and testing sets for the specified
        dataset.
    """
    datasets = {}
    for split in ('train', 'valid', 'test'):
        if dataset.lower() == 'penntreebank':
            datasets[split] = PennTreebank(root=data_dir, split=split)
        elif dataset.lower() == 'wikitext2':
            datasets[split] = WikiText2(root=data_dir, split=split)
        elif dataset.lower() == 'wikitext103':
            datasets[split] = WikiText103(root=data_dir, split=split)
        else:
            raise AssertionError(f'Unsupported dataset {dataset}.')

    return (datasets['train'], datasets['valid'], datasets['test'])


def encode_and_flatten(
    raw_text_iter: IterableDataset,
    tokenizer: Callable[[str], list[str]],
    vocab: Vocab,
) -> torch.Tensor:
    """Tokenizes, encodes, and flattens a dataset."""
    data = [
        torch.tensor(vocab(tokenizer(item)), dtype=torch.long)
        for item in raw_text_iter
    ]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def get_dataset(
    dataset: Literal['penntreebank', 'wikitext2', 'wikitext103'],
    data_dir: str,
    seq_len: int,
    batch_size: int,
    *,
    cuda: bool = False,
    rank: int | None = None,
    world_size: int | None = None,
) -> tuple[Datasets, Vocab]:
    """Get language modeling datasets.

    Args:
        dataset (str): one of 'penntreebank', 'wikitext2', or 'wikitext102'.
        data_dir (str): directory to download datasets to.
        seq_len (int): number of tokens in a training sequence.
        batch_size (int): batch size.
        cuda (bool): set as True if training with CUDA.
        rank (int): optional rank of this worker for initalizing the
            distributed sampler.
        world_size (int): optional world size if using distributed training.

    Returns:
        Datasets, a named tuple with attributes train, val, and test, each
        corresponding to another tuple with the dataloader and sampler for
        that training data split. Also returns the vocab used to encode
        the datasets.
    """
    train_iter, val_iter, test_iter = download_dataset(dataset, data_dir)

    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(
        map(tokenizer, train_iter),
        specials=['<unk>'],
    )
    vocab.set_default_index(vocab['<unk>'])

    train_data = encode_and_flatten(train_iter, tokenizer, vocab)
    val_data = encode_and_flatten(val_iter, tokenizer, vocab)
    test_data = encode_and_flatten(test_iter, tokenizer, vocab)

    train_dataset = _Dataset(train_data, seq_len=seq_len)
    val_dataset = _Dataset(val_data, seq_len=seq_len)
    test_dataset = _Dataset(test_data, seq_len=seq_len)

    num_replicas = (
        torch.distributed.get_world_size()
        if world_size is None
        else world_size
    )
    rank = torch.distributed.get_rank() if rank is None else rank

    train_sampler, val_sampler, test_sampler = (
        DistributedSampler(dataset, num_replicas=num_replicas, rank=rank)
        for dataset in (train_dataset, val_dataset, test_dataset)
    )

    train_loader, val_loader, test_loader = (
        DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=True,
            sampler=sampler,
            num_workers=4 if cuda else 0,
            pin_memory=cuda,
        )
        for dataset, sampler in zip(
            (train_dataset, val_dataset, test_dataset),
            (train_sampler, val_sampler, test_sampler),
        )
    )

    return (
        Datasets(
            train=LoaderSampler(train_loader, train_sampler),
            val=LoaderSampler(val_loader, val_sampler),
            test=LoaderSampler(test_loader, test_sampler),
        ),
        vocab,
    )


if __name__ == '__main__':
    datasets, vocab = get_dataset(
        'penntreebank',
        '/tmp/torchtext-data',
        12,
        4,
        world_size=1,
        rank=0,
    )

    datasets.train.sampler.set_epoch(0)
    for batch, (data, target) in enumerate(datasets.train.loader):
        if batch > 2:
            break
        print(f'BATCH {batch}')
        for sample in range(len(data)):
            print(f'SAMPLE {sample}')
            print(vocab.lookup_tokens(list(data[sample])))
            print(vocab.lookup_tokens(list(target[sample])))

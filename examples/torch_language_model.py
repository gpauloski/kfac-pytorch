# coding: utf-8
import argparse
import time
import math
import os
import sys
import datetime
import kfac
import torch
import torch.distributed as dist
import torch.optim as optim

import rnn_utils.lstm as models
import cnn_utils.optimizers as optimizers

from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchnlp.datasets import wikitext_2_dataset, penn_treebank_dataset
from torchnlp.encoders import LabelEncoder
from torchnlp.samplers import BPTTBatchSampler
from tqdm import tqdm
from utils import Metric, create_lr_schedule
from rnn_utils.utils import DistributedSampler

def parse_args():
    # General Settings
    parser = argparse.ArgumentParser(description='PyTorch Distributed Language Model')
    parser.add_argument('--dataset', type=str, default='penntreebank',
                        help='data corpus to use (wikitext2, wikitext103, penntreebank)')
    parser.add_argument('--data-dir', type=str, default='/tmp/',
                        help='location to download the data corpus')
    parser.add_argument('--log-dir', default='./logs',
                        help='tensorboard/checkpoint log directory')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='do not use CUDA')

    # Model Settings
    # Hyperparams based on :
    #  - https://arxiv.org/pdf/1409.2329.pdf
    #  - https://openreview.net/pdf?id=HyMTkQZAb
    parser.add_argument('--emsize', type=int, default=650,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=650,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--not-tied', action='store_true',
                        help='do not tie the word embedding and softmax weights')

    # Training Settings
    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--base-lr', type=float, default=10.0,
                        help='initial learning rate, scaled by num workers')
    parser.add_argument('--lr-decay', nargs='+', type=int, default=[30, 35],
                        help='[DISABLED] epoch intervals to decay lr')
    parser.add_argument('--lr-decay-epoch', type=int, default=1000,
                        help='epoch to start decaying learning rate')
    parser.add_argument('--lr-decay-rate', type=float, default=1 / 1.2,
                        help='multiplicative factor of learning rate decay')
    parser.add_argument('--warmup-epochs', type=float, default=0,
                        help='number of warmup epochs')
    parser.add_argument('--momentum', type=float, default=0.0,
                        help='SGD momentum (default 0.0)')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='weight decay (default 0.0)')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')

    # KFAC Parameters
    parser.add_argument('--kfac-update-freq', type=int, default=10,
                        help='iters between kfac inv ops (0 = no kfac) (default: 10)')
    parser.add_argument('--kfac-cov-update-freq', type=int, default=1,
                        help='iters between kfac cov ops (default: 1)')
    parser.add_argument('--kfac-update-freq-alpha', type=float, default=10,
                        help='KFAC update freq multiplier (default: 10)')
    parser.add_argument('--kfac-update-freq-decay', nargs='+', type=int, default=None,
                        help='KFAC update freq decay schedule (default None)')
    parser.add_argument('--use-inv-kfac', action='store_true', default=False,
                        help='Use inverse KFAC update instead of eigen (default False)')
    parser.add_argument('--stat-decay', type=float, default=0.95,
                        help='Alpha value for covariance accumulation (default: 0.95)')
    parser.add_argument('--damping', type=float, default=0.002,
                        help='KFAC damping factor (default 0.003)')
    parser.add_argument('--damping-alpha', type=float, default=0.5,
                        help='KFAC damping decay factor (default: 0.5)')
    parser.add_argument('--damping-decay', nargs='+', type=int, default=None,
                        help='KFAC damping decay schedule (default None)')
    parser.add_argument('--kl-clip', type=float, default=0.001,
                        help='KL clip (default: 0.001)')
    parser.add_argument('--skip-layers', nargs='+', type=str, default=['linear', 'embedding'],
                        help='Layer types to ignore registering with KFAC'
                             '(default: [linear, embedding])')
    parser.add_argument('--coallocate-layer-factors', action='store_true', default=False,
                        help='Compute A and G for a single layer on the same worker. ')

    # Set automatically by torch distributed launch
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args


def get_dataset(args):
    args.data_dir = os.path.join(args.data_dir, args.dataset)
    os.makedirs(args.data_dir, exist_ok=True)


    if not args.verbose: 
        args.backend.barrier()
    if args.dataset == 'wikitext2':
        datasets = wikitext_2_dataset(directory=args.data_dir, train=True, dev=True, test=True)
    elif args.dataset == 'penntreebank':
        datasets = penn_treebank_dataset(directory=args.data_dir, train=True, dev=True, test=True)
    if args.verbose:
        args.backend.barrier()

    encoder = LabelEncoder(datasets[0])
    datasets = [encoder.batch_encode(d) for d in datasets]
    if args.verbose:
        print('Loaded {}: train_token_count={}, vocab_size={}'.format(
                args.dataset, len(datasets[0]), encoder.vocab_size))

    def collate(batch):
        batch = torch.stack(batch, dim=1)
        return  batch[0:-1], batch[1:].view(-1).contiguous()

    torch.set_num_threads(4)
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    bptt_samplers = tuple([
        BPTTBatchSampler(d, args.bptt + 1, args.batch_size, drop_last=True, type_='source')
        for d in datasets
    ])
    dist_samplers = tuple([
        DistributedSampler(d, num_replicas=args.backend.size(), rank=args.backend.rank(), shuffle=True)
        for d in (bptt_samplers)
    ])
    data_loaders = tuple([
        DataLoader(data, batch_sampler=sampler, collate_fn=collate, **kwargs)
        for data, sampler in zip(datasets, dist_samplers)
    ])

    return data_loaders, dist_samplers, encoder.vocab_size


def train(epoch, model, optimizer, preconditioner, loss_func, data_loader, args):
    model.train()
    
    train_loss = Metric('train_loss', args.backend)
    hidden = None

    with tqdm(total=len(data_loader),
              bar_format='{l_bar}{bar:10}{r_bar}',
              desc='Epoch {:2d}/{:2d}'.format(epoch + 1, args.epochs),
              disable=not args.verbose) as t:
        for batch_idx, (data, target) in enumerate(data_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            if hidden is not None:
                hidden = model.module.detach(hidden)
            output, hidden = model(data, hidden)
            loss = loss_func(output, target)
            loss.backward()

            with torch.no_grad():
                train_loss.update(loss)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            if preconditioner is not None:
                preconditioner.step()
            optimizer.step() 

            t.set_postfix_str("lr: {:5.4f} loss: {:4.2f}, ppl: {:6.2f}".format(
                     optimizer.param_groups[0]['lr'],
                     train_loss.avg, math.exp(train_loss.avg)))
            t.update(1)

    if args.log_writer is not None:
        args.log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        args.log_writer.add_scalar('train/ppl', math.exp(train_loss.avg), epoch)
        args.log_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)


def evaluate(epoch, model, loss_func, data_loader, args):
    model.eval()
    val_loss = Metric('val_loss', args.backend)

    verbose = args.verbose if epoch is not None else False

    with tqdm(total=len(data_loader),
              bar_format='{l_bar}{bar:10}|{postfix}',
              desc='           ',
              disable=not verbose) as t:
        with torch.no_grad():
            for i, (data, target) in enumerate(data_loader):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output, hidden = model(data)
                loss = loss_func(output, target)
                val_loss.update(loss)

                t.update(1) 
            t.set_postfix_str("\b\b val_loss: {:4.2f}, val_ppl: {:6.2f}".format(
                            val_loss.avg, math.exp(val_loss.avg)), refresh=False)

    if args.log_writer is not None and epoch is not None:
        args.log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        args.log_writer.add_scalar('val/ppl', math.exp(val_loss.avg), epoch)

    return val_loss.avg.item()


if __name__ == '__main__':
    args = parse_args()

    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.cuda:
        torch.cuda.set_device(args.local_rank)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    else:
        torch.manual_seed(args.seed)

    print('rank = {}, world_size = {}, device_ids = {}'.format(
            torch.distributed.get_rank(), torch.distributed.get_world_size(),
            args.local_rank))

    args.backend = kfac.utils.get_comm_backend()
    args.base_lr = args.base_lr * args.backend.size()
    args.verbose = True if args.backend.rank() == 0 else False
    args.horovod = False

    data_loaders, data_samplers, args.ntokens = get_dataset(args)

    model = models.LSTMModel(args.ntokens, args.emsize, args.nhid, args.nlayers, 
            dropout=args.dropout, tie_weights=not args.not_tied)
    model =  model.cuda() if args.cuda else model
    model = torch.nn.parallel.DistributedDataParallel(model, 
            device_ids=[args.local_rank])


    args.log_dir = os.path.join(args.log_dir, 
            "language_model_kfac{}_gpu_{}_{}".format(
            args.kfac_update_freq, args.backend.size(),
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    os.makedirs(args.log_dir, exist_ok=True)
    args.log_writer = SummaryWriter(args.log_dir) if args.verbose else None

    if args.verbose:
        print(args)
        print(model)
  
    optimizer, precon, kfac_schedules, _ = optimizers.get_optimizer(model, args) 
    loss_func = torch.nn.NLLLoss()

    lr_schedules = [StepLR(optimizer, step_size=1, gamma=args.lr_decay_rate)]
    if precon is not None:
        lr_schedules.append(StepLR(precon, step_size=1, gamma=args.lr_decay_rate))

    start = time.time()

    for epoch in range(args.epochs):
        data_samplers[0].set_epoch(0)
        train(epoch, model, optimizer, precon, loss_func, data_loaders[0], args)
        for scheduler in kfac_schedules:
            scheduler.step()
        if epoch + 1 >= args.lr_decay_epoch:
            for scheduler in lr_schedules:
                scheduler.step()
        evaluate(epoch, model, loss_func, data_loaders[1], args)

    if args.verbose:
        print("\nTraining time:", str(datetime.timedelta(seconds=time.time() - start)))

    # Run on test data.
    test_loss = evaluate(None, model, loss_func, data_loaders[2], args)
    if args.verbose:
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
              test_loss, math.exp(test_loss)))
        print('=' * 89)


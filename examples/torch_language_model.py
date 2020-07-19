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

from torch.optim.lr_scheduler import LambdaLR
from torchtext.experimental import datasets
from torchtext.data.utils import get_tokenizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import Metric, create_lr_schedule

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
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--no-tied', action='store_true',
                        help='do not tie the word embedding and softmax weights')

    # Training Settings
    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--base-lr', type=float, default=1.0,
                        help='initial learning rate, scaled by num workers')
    parser.add_argument('--lr-decay', nargs='+', type=int, default=[20, 30, 35],
                        help='epoch intervals to decay lr')
    parser.add_argument('--warmup-epochs', type=float, default=5,
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
    parser.add_argument('--skip-layers', nargs='+', type=str, default=['linear'],
                        help='Layer types to ignore registering with KFAC'
                             '(default: ["linear"])')
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

    tokenizer = get_tokenizer("basic_english")
    if args.dataset == 'wikitext2':
        dataset = datasets.WikiText2
    elif args.dataset == 'wikitext103':
        dataset = datasets.WikiText103
    elif args.dataset == 'penntreebank':
        dataset = datasets.PennTreebank

    if not args.verbose: 
        args.backend.barrier()
    train_data, val_data, test_data = dataset(tokenizer=tokenizer, root=args.data_dir)
    if args.verbose:
        args.backend.barrier()
    ntokens = len(train_data.get_vocab())
    batch_size = args.batch_size * (args.bptt + 1)

    def collate(data):
        data = torch.stack(data)
        source = data.view(args.batch_size, -1).t().contiguous()
        data = source[:-1]
        target = source[1:].contiguous().view(-1)
        return data, target

    torch.set_num_threads(4)
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    size, rank = args.backend.size(), args.backend.rank()

    train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data, num_replicas=size, rank=rank, shuffle=False)
    train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, collate_fn=collate, drop_last=True,
            sampler=train_sampler, shuffle=False, **kwargs)
    val_sampler =torch.utils.data.distributed.DistributedSampler(
            val_data, num_replicas=size, rank=rank, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=batch_size, collate_fn=collate, drop_last=True,
            sampler=val_sampler, shuffle=False, **kwargs)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_data, num_replicas=size, rank=rank, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, collate_fn=collate, drop_last=True,
            sampler=test_sampler, shuffle=False, **kwargs)

    return (train_sampler, train_loader, val_sampler, val_loader,
            test_sampler, test_loader, ntokens)


def train(epoch, model, optimizer, preconditioner, lrs, loss_func,
          train_sampler, train_loader, args):
    model.train()
    train_sampler.set_epoch(epoch)
    
    train_loss = Metric('train_loss', args.backend)

    with tqdm(total=len(train_loader), 
              bar_format='{l_bar}{bar:10}{r_bar}',
              desc='Epoch {:2d}/{:2d}'.format(epoch + 1, args.epochs),
              disable=not args.verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            model.zero_grad()
            output, hidden = model(data)
            loss = loss_func(output, target)
            loss.backward()

            with torch.no_grad():
                train_loss.update(loss)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            if preconditioner is not None:
                preconditioner.step(epoch)
            optimizer.step() 

            t.set_postfix_str("loss: {:4.2f}, ppl: {:6.2f}".format(
                     train_loss.avg, math.exp(train_loss.avg)))
            t.update(1)

    if args.log_writer is not None:
        args.log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        args.log_writer.add_scalar('train/ppl', math.exp(train_loss.avg), epoch)
        args.log_writer.add_scalar('train/lr', args.base_lr * lrs(epoch), epoch)


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

    #torch.multiprocessing.set_start_method('spawn')
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

    ds = get_dataset(args)
    train_sampler, train_loader, _, val_loader, _, test_loader, args.ntokens = ds

    model = models.LSTMModel(args.ntokens, args.emsize, args.nhid, args.nlayers, 
            dropout=args.dropout, tie_weights=not args.no_tied)
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
  
    optimizer, precon, lr_schedules, lrs, = optimizers.get_optimizer(model, args) 
    loss_func = torch.nn.NLLLoss()

    start = time.time()

    try: 
        for epoch in range(args.epochs):
            train(epoch, model, optimizer, precon, lrs, loss_func, 
                    train_sampler, train_loader, args)
            for scheduler in lr_schedules:
                scheduler.step()
            evaluate(epoch, model, loss_func, val_loader, args)
    except KeyboardInterrupt:
        pass

    if args.verbose:
        print("\nTraining time:", str(datetime.timedelta(seconds=time.time() - start)))

    # Run on test data.
    test_loss = evaluate(None, model, loss_func, test_loader, args)
    if args.verbose:
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
              test_loss, math.exp(test_loss)))
        print('=' * 89)


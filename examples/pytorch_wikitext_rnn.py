# coding: utf-8
import argparse
import time
import math
import os
import sys
import torch
import torch.nn as nn
import torch.onnx
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as torch_data
from torch.optim.lr_scheduler import LambdaLR
from torchtext.experimental import datasets
from distutils.version import LooseVersion
from datetime import datetime, timedelta
from tqdm import tqdm

import rnn_language_utils.model as models
from utils import *

import horovod.torch as hvd
sys.path.append("./kfac")
import kfac

def initialize():
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
    parser.add_argument('--data', type=str, default='examples/rnn_language_utils/data/wikitext-2',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--base-lr', type=float, default=20,
                        help='initial learning rate, scaled by num workers')
    parser.add_argument('--lr-decay', nargs='+', type=int, default=[10, 15, 20, 25, 30, 35],
                        help='epoch intervals to decay lr')
    parser.add_argument('--warmup-epochs', type=float, default=5,
                        help='number of warmup epochs')
    parser.add_argument('--momentum', type=float, default=0.0,
                        help='SGD momentum (default 0.0)')
    parser.add_argument('--wd', type=float, default=0.0,
                        help='weight decay (default 0.0)')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--no-cuda', action='store_true',
                        help='do not use CUDA')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='report interval')
    parser.add_argument('--log-dir', default='./logs',
                        help='tensorboard/checkpoint log directory')
    parser.add_argument('--onnx-export', type=str, default='',
                        help='path to export the final model in onnx format')
    parser.add_argument('--nhead', type=int, default=2,
                       help='the number of heads in the encoder/decoder of the transformer model')

    # KFAC Parameters
    parser.add_argument('--kfac-update-freq', type=int, default=10,
                        help='iters between kfac inv ops (0 = no kfac) (default: 10)')
    parser.add_argument('--kfac-cov-update-freq', type=int, default=1,
                        help='iters between kfac cov ops (default: 1)')
    parser.add_argument('--stat-decay', type=float, default=0.95,
                        help='Alpha value for covariance accumulation (default: 0.95)')
    parser.add_argument('--damping', type=float, default=0.002,
                        help='KFAC damping factor (default 0.003)')
    parser.add_argument('--kl-clip', type=float, default=0.001,
                        help='KL clip (default: 0.001)')
    parser.add_argument('--diag-blocks', type=int, default=1,
                        help='Number of blocks to approx layer factor with (default: 1)')
    parser.add_argument('--diag-warmup', type=int, default=5,
                        help='Epoch to start diag block approximation at (default: 5)')


    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    hvd.init()

    args.verbose = 1 if hvd.rank() == 0 else 0
    if args.verbose:
        print(args)

    if args.cuda:
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    args.device = torch.device("cuda" if args.cuda else "cpu")

    args.log_dir = os.path.join(args.log_dir, 
                                "wikitext_{}_kfac{}_gpu_{}_{}".format(
                                args.model, args.kfac_update_freq, hvd.size(),
                                datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    args.save = os.path.join(args.log_dir, 'model_checkpoint.pt')
    os.makedirs(args.log_dir, exist_ok=True)

    # Horovod: write TensorBoard logs on first worker.
    try:
        if LooseVersion(torch.__version__) >= LooseVersion('1.2.0'):
            from torch.utils.tensorboard import SummaryWriter
        else:
            from tensorboardX import SummaryWriter
        args.log_writer = SummaryWriter(args.log_dir) if hvd.rank() == 0 else None
    except ImportError:
        args.log_writer = None

    return args


def get_datasets(args):
    #corpus = data.Corpus(args.data)

    #train_data = data.batchify(corpus.train, args.batch_size)
    #val_data = data.batchify(corpus.valid, args.batch_size)
    #test_data = data.batchify(corpus.test, args.batch_size)

    #train_dataset = data.wiki_dataset(train_data, args.bptt)
    #val_dataset = data.wiki_dataset(val_data, args.bptt)
    #test_dataset = data.wiki_dataset(test_data, args.bptt)

    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

    train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, sampler=train_sampler, **kwargs)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, sampler=val_sampler, **kwargs)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, sampler=test_sampler, **kwargs)

    return (train_sampler, train_loader), (val_sampler, val_loader), \
           (test_sampler, test_loader), len(corpus.dictionary) 


def get_model(args):
    if args.model == 'Transformer':
        model = models.TransformerModel(args.ntokens, args.emsize, args.nhead, args.nhid, 
                                        args.nlayers, args.dropout).to(args.device)
    else:
        model = models.RNNModel(args.model, args.ntokens, args.emsize, args.nhid, 
                                args.nlayers, args.dropout, args.tied).to(args.device)

    # Horovod: scale learning rate by the number of GPUs.
    args.base_lr = args.base_lr * hvd.size()
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr,
                          momentum=args.momentum, weight_decay=args.wd)

    if args.kfac_update_freq > 0:
        preconditioner = kfac.KFAC(
                model, lr=args.base_lr, stat_decay=args.stat_decay,
                damping=args.damping, kl_clip=args.kl_clip,
                TCov=args.kfac_cov_update_freq,
                TInv=args.kfac_update_freq,
                diag_blocks=args.diag_blocks,
                diag_warmup=args.diag_warmup)
    else:
         preconditioner = None

    optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=model.named_parameters(),
            compression=hvd.Compression.none, op=hvd.Average)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    lrs = create_lr_schedule(hvd.size(), args.warmup_epochs, args.lr_decay, alpha=0.25)
    lr_schedules = [LambdaLR(optimizer, lrs)]
    if preconditioner is not None:
        lr_schedules.append(LambdaLR(preconditioner, lrs))

    criterion = nn.NLLLoss()

    return model, optimizer, preconditioner, lr_schedules, lrs, criterion


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate(epoch, model, criterion, data_loader, args):
    model.eval()
    val_loss = Metric('val_loss')

    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    verbose = args.verbose if epoch is not None else False

    with tqdm(total=len(data_loader),
              bar_format='{l_bar}{bar}|{postfix}',
              desc='           ',
              disable=not verbose) as t:
        with torch.no_grad():
            for i, (data, target) in enumerate(data_loader):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = torch.squeeze(data), torch.squeeze(target)
                if args.model == 'Transformer':
                    output = model(data)
                    output = output.view(-1, ntokens)
                else:
                    output, hidden = model(data, hidden)
                    hidden = repackage_hidden(hidden)
                loss = criterion(output, target)
                val_loss.update(loss)
                t.update(1) 

            t.set_postfix_str("\b\b val_loss: {:4.2f}, val_ppl: {:6.2f}".format(
                            val_loss.avg.item(), math.exp(val_loss.avg.item())), 
                            refresh=False)

    if args.log_writer is not None and epoch is not None:
        args.log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        args.log_writer.add_scalar('val/ppl', torch.exp(val_loss.avg), epoch)

    return val_loss.avg.item()


def train(epoch, model, optimizer, preconditioner, lr_schedules, lrs,
          criterion, train_sampler, train_loader, args):
    model.train()
    train_sampler.set_epoch(epoch)
    total_loss = 0.
    train_loss = Metric('train_loss')

    for scheduler in lr_schedules:
        scheduler.step()
    lr = args.base_lr * lrs(epoch)

    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)

    with tqdm(total=len(train_loader), 
              desc='Epoch {:2d}/{:2d}'.format(epoch + 1, args.epochs),
              disable=not args.verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = torch.squeeze(data), torch.squeeze(target)
            # Starting each batch, we detach the hidden state from how it was 
            # previously produced. If we didn't, the model would try backpropagating 
            # all the way to start of the dataset.
            model.zero_grad()
            if args.model == 'Transformer':
                output = model(data)
                output = output.view(-1, args.ntokens)
            else:
                hidden = repackage_hidden(hidden)
                output, hidden = model(data, hidden)
            loss = criterion(output, target)

            loss.backward()
            train_loss.update(loss)
            #total_loss += loss.item()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.synchronize()
            if preconditioner is not None:
                preconditioner.step(epoch)
            with optimizer.skip_synchronize():
                optimizer.step() 

            #if batch_idx % args.log_interval == 0 and batch_idx > 0:
            #train_loss.update(torch.tensor(total_loss / args.log_interval))
            #total_loss = 0.

            t.set_postfix_str("loss: {:4.2f}, ppl: {:6.2f}".format(
                     train_loss.avg.item(), math.exp(train_loss.avg.item())))
            t.update(1)

    if args.log_writer is not None:
        args.log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        args.log_writer.add_scalar('train/ppl', torch.exp(train_loss.avg), epoch)
        args.log_writer.add_scalar('train/lr', lr, epoch)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    args = initialize()

    train_data, val_data, test_data, ntokens = get_datasets(args)
    args.ntokens = ntokens
    model, opt, preconditioner, lr_schedules, lrs, loss_func = get_model(args)

    if args.verbose:
        print(model)

    start = time.time()

    try: 
        for epoch in range(args.epochs):
            train(epoch, model, opt, preconditioner, lr_schedules, lrs,
                  loss_func, train_data[0], train_data[1], args)
            evaluate(epoch, model, loss_func, val_data[1], args)
            if args.verbose:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
    except KeyboardInterrupt:
        pass

    if args.verbose:
        print("\nTraining time:", str(timedelta(seconds=time.time() - start)))

    # Run on test data.
    test_loss = evaluate(None, model, loss_func, test_data[1], args)
    if args.verbose:
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
              test_loss, math.exp(test_loss)))
        print('=' * 89)

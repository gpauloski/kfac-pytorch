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
from torch.optim.lr_scheduler import LambdaLR
from distutils.version import LooseVersion
from datetime import datetime, timedelta
from tqdm import tqdm

import rnn_language_utils.data as data
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
    parser.add_argument('--lr-decay', nargs='+', type=int, default=[10, 15, 25, 30],
                        help='epoch intervals to decay lr')
    parser.add_argument('--warmup-epochs', type=float, default=5,
                        help='number of warmup epochs')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--wd', type=float, default=0.00005,
                        help='weight decay')
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
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
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
    corpus = data.Corpus(args.data)

    # Starting from sequential data, batchify arranges the dataset into columns.
    # For instance, with the alphabet as the sequence and batch size 4, we'd get
    # ┌ a g m s ┐
    # │ b h n t │
    # │ c i o u │
    # │ d j p v │
    # │ e k q w │
    # └ f l r x ┘.
    # These columns are treated as independent by the model, which means that the
    # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    # batch processing.

    def batchify(data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(args.device)

    train_data = batchify(corpus.train, args.batch_size)
    val_data = batchify(corpus.valid, args.batch_size)
    test_data = batchify(corpus.test, args.batch_size)

    return train_data, val_data, test_data, len(corpus.dictionary) 


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


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.
def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(epoch, model, criterion, data_source, args):
    model.eval()
    val_loss = Metric('val_loss')

    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    length = len(range(0, data_source.size(0) - 1, args.bptt))
    verbose = args.verbose if epoch is not None else False

    with tqdm(total=length,
              bar_format='{l_bar}{bar}|{postfix}',
              desc='           '.format(epoch + 1, args.epochs),
              disable=not verbose) as t:
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, args.bptt):
                data, targets = get_batch(data_source, i)
                if args.model == 'Transformer':
                    output = model(data)
                    output = output.view(-1, ntokens)
                else:
                    output, hidden = model(data, hidden)
                    hidden = repackage_hidden(hidden)
                loss = criterion(output, targets)
                val_loss.update(loss)
                t.update(1) 

            t.set_postfix_str("\b\b val_loss: {:4.2f}, val_ppl: {:6.2f}".format(
                            val_loss.avg.item(), math.exp(val_loss.avg.item())), 
                            refresh=False)
     
    if args.log_writer is not None and epoch is not None:
        args.log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        args.log_writer.add_scalar('val/ppl', torch.exp(val_loss.avg), epoch)

    return val_loss


def train(epoch, model, optimizer, preconditioner, lr_schedules, lrs,
          criterion, train_data, args):
    model.train()
    total_loss = 0.
    train_loss = Metric('train_loss')

    for scheduler in lr_schedules:
        scheduler.step()
    lr = args.base_lr * lrs(epoch)

    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    length = len(range(0, train_data.size(0) - 1, args.bptt))

    with tqdm(total=length, 
              desc='Epoch {:2d}/{:2d}'.format(epoch + 1, args.epochs),
              disable=not args.verbose) as t:
        for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
            data, targets = get_batch(train_data, i)
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
            loss = criterion(output, targets)
            loss.backward()
            total_loss += loss.item()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.synchronize()
            if preconditioner is not None:
                preconditioner.step(epoch)
            with optimizer.skip_synchronize():
                optimizer.step() 

            if batch % args.log_interval == 0 and batch > 0:
                loss = torch.tensor(total_loss / args.log_interval)
                train_loss.update(loss)

                total_loss = 0.
            
            t.set_postfix_str("loss: {:4.2f}, ppl: {:6.2f}".format(
                    train_loss.avg.item(), math.exp(train_loss.avg.item()), lr))
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

    start = time.time()

    try: 
        for epoch in range(args.epochs):
            train(epoch, model, opt, preconditioner, lr_schedules, lrs,
                  loss_func, train_data, args)
            evaluate(epoch, model, loss_func, val_data, args)
            with open(args.save, 'wb') as f:
                torch.save(model, f)
    except KeyboardInterrupt:
        pass

    if args.verbose:
        print("\nTraining time:", str(timedelta(seconds=time.time() - start)))

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.
        if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
            model.rnn.flatten_parameters()

    # Run on test data.
    loss = evaluate(None, model, loss_func, test_data, args)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
          test_loss, math.exp(test_loss)))
    print('=' * 89)

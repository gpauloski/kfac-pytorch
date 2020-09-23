import math
import sys
import torch
from tqdm import tqdm

sys.path.append('..')
from utils import Metric, accuracy

def train(epoch,
          model,
          optimizer, 
          preconditioner, 
          loss_func, 
          train_sampler, 
          train_loader, 
          args):

    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss', args.backend) 
    train_accuracy = Metric('train_accuracy', args.backend)
    scaler = args.grad_scaler if args.grad_scaler else None

    with tqdm(total=len(train_loader),
              bar_format='{l_bar}{bar:10}{r_bar}',
              desc='Epoch {:3d}/{:3d}'.format(epoch, args.epochs),
              disable=not args.verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()

            for i in range(0, len(data), args.batch_size):
                data_batch = data[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]

                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        output = model(data_batch)
                        loss = loss_func(output, target_batch)
                else:
                    output = model(data_batch)
                    loss = loss_func(output, target_batch)
                
                loss_ = loss.detach().clone() 
                loss = loss / args.batches_per_allreduce

                if args.horovod:
                    loss.backward()
                else:
                    if i < args.batches_per_allreduce:
                        with model.no_sync():
                            if scaler is not None:
                                scaler.scale(loss).backward()
                            else:
                                loss.backward()
                    else:
                        if scaler is not None:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()

                with torch.no_grad():            
                    train_loss.update(loss_)
                    train_accuracy.update(accuracy(output, target_batch))

            if args.horovod:
                optimizer.synchronize()
                if preconditioner is not None:
                    preconditioner.step()
                with optimizer.skip_synchronize():
                    optimizer.step()
            else:
                if preconditioner is not None:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    preconditioner.step()
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

            t.set_postfix_str("loss: {:.4f}, acc: {:.2f}%, lr: {:.4f}".format(
                    train_loss.avg, 100*train_accuracy.avg,
                    optimizer.param_groups[0]['lr']))
            t.update(1)

    if args.log_writer is not None:
        args.log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        args.log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)
        args.log_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'],
                                    epoch)


def test(epoch, 
         model, 
         loss_func, 
         val_loader, 
         args):
    model.eval()
    val_loss = Metric('val_loss', args.backend)
    val_accuracy = Metric('val_accuracy', args.backend)

    with tqdm(total=len(val_loader),
              bar_format='{l_bar}{bar:10}|{postfix}',
              desc='             '.format(epoch, args.epochs),
              disable=not args.verbose) as t:
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                val_loss.update(loss_func(output, target))
                val_accuracy.update(accuracy(output, target))

                t.update(1)
                if i + 1 == len(val_loader):
                    t.set_postfix_str("\b\b val_loss: {:.4f}, val_acc: {:.2f}%".format(
                            val_loss.avg, 100*val_accuracy.avg),
                            refresh=False)

    if args.log_writer is not None:
        args.log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        args.log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)

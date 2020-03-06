
def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(epoch):
    if hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

def create_lr_schedule(workers, warmup_epochs, decay_schedule, alpha=0.1):
    def lr_schedule(epoch):
        lr_adj = 1.
        if epoch < warmup_epochs:
            lr_adj = 1. / workers * (epoch * (workers - 1) / warmup_epochs + 1)
        else:
            decay_schedule.sort(reverse=True)
            for e in decay_schedule:
                if epoch > e:
                    lr_adj *= alpha
        return lr_adj
    return lr_schedule

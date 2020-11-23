import torch
import torch.nn.functional as F

from kfac import comm

def accuracy(output, target):
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).float().mean()

def save_checkpoint(model, optimizer, preconditioner, schedulers, filepath):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'preconditioner': preconditioner.state_dict()
                if preconditioner is not None else None,
        'schedulers': [s.state_dict() for s in schedulers]
                if isinstance(schedulers, list) else None
    }
    torch.save(state, filepath)

class LabelSmoothLoss(torch.nn.Module):
    
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

class Metric(object):
    def __init__(self, name):
        self.name = name
        self.total = torch.tensor(0.0)
        self.n = torch.tensor(0.0)

    def update(self, val, n=1):
        comm.backend.allreduce(val, async_op=False)
        self.total += val.cpu()
        self.n += n

    @property
    def avg(self):
        return self.total / self.n

def create_lr_schedule(workers, warmup_epochs, decay_schedule, alpha=0.1):
    def lr_schedule(epoch):
        lr_adj = 1.
        if epoch < warmup_epochs:
            lr_adj = 1. / workers * (epoch * (workers - 1) / warmup_epochs + 1)
        else:
            decay_schedule.sort(reverse=True)
            for e in decay_schedule:
                if epoch >= e:
                    lr_adj *= alpha
        return lr_adj
    return lr_schedule

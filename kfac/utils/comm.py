import enum
import os
import torch
import torch.distributed as dist

try:
    import horovod.torch as hvd
    HVD_EXISTS = True
except:
    HVD_EXISTS = False


class Ops(enum.Enum):
    Average = "average"
    Sum = "sum"


def get_comm_backend():
    if _horovod_is_initialized():
        return HorovodBackend()
    elif _torch_distributed_is_initialized():
        return TorchBackend()
    else:
        return CommBackend()


def _horovod_is_initialized():
    if not HVD_EXISTS:
        return False
    try:
        # If hvd.init() has not been called, this will fail
        world_size = hvd.size()
    except:
        return False
    else:
        return True


def _torch_distributed_is_initialized():
    return dist.is_initialized()


class CommBackend(object):
    """Distributed training communication abstraction."""
    def __init__(self):
        self.Average = Ops.Average
        self.Sum = Ops.Sum

    def size(self):
        """Get worker count"""
        return 1

    def local_rank(self):
        """Get workers local rank"""
        return 0

    def rank(self):
        """Get unique worker rank"""
        return 0

    def allgather(self, tensor, tensor_list):
        raise NotImplementedError()

    def allreduce(self, tensors, op=Ops.Average):
        """Allreduce list of tensors inplace.

        Args:
          tensors (list, torch.Tensor): list of tensors to reduce
          op (Op): reduction operation to apply
        """
        pass

    def broadcast(self, tensors, ranks):
        """Broadcasts tensors inplace.

        Args:
          tensors (list, torch.Tensor): list of tensors to broadcast
          ranks (list, int): list of source rank for each tensor in tensors
        """
        pass

    def gather(self, tensors, rank=0):
        raise NotImplementedError()

    def reduce(self, tensors, ranks, op=Ops.Average):
        """Reduce list of tensors inplace.

        Args:
          tensors (list, torch.Tensor): list of tensors to reduce
          ranks (list, int): rank corresponding to each tensor to place final
              results on
          op (Op): reduction operation to apply
        """
        pass

    def reduce_scalar(self, tensor):
        """Reduce (sum) single tensor in place."""
        pass

    def barrier(self):
        return

class HorovodBackend(CommBackend):
    def size(self):
        return hvd.size()
    
    def local_rank(self):
        return hvd.local_rank()

    def rank(self):
        return hvd.rank()

    def allgather(self, tensor, tensor_list):
        raise NotImplementedError()

    def allreduce(self, tensors, op=Ops.Average):
        handles = []
        op = self._get_op(op)
        for tensor in tensors:
            handles.append(hvd.allreduce_async_(tensor, op=op))
        self._sync(handles)

    def broadcast(self, tensors, ranks):
        handles = []
        for tensor, rank in zip(tensors, ranks):
            handles.append(hvd.broadcast_async_(tensor, root_rank=rank))
        self._sync(handles)

    def gather(self, tensors, tensor_list, rank=0):
        raise NotImplementedError()

    def reduce(self, tensors, rank, op=Ops.Average):
        # Horovod only support allreduce
        self.allreduce(tensors, op=op)

    def reduce_scalar(self, tensor):
        hvd.allreduce_(tensor, op=hvd.Sum)

    def barrier(self):
        hvd.allreduce(torch.tensor(1), name='barrier')

    def _get_op(self, op):
        if op == Ops.Average:
            return hvd.Average
        elif op == Ops.Sum:
            return hvd.Sum
        else:
            raise ValueError('Unknown communication operation {}'.format(op))

    def _sync(self, handles):
        for handle in handles:
            hvd.synchronize(handle)


class TorchBackend(CommBackend):
    def size(self):
        return dist.get_world_size()

    def local_rank(self):
        try:
            return os.environ['LOCAL_RANK']
        except:
            raise RuntimeError('LOCAL_RANK must be set in the environment'
                               'when using torch.distributed')

    def rank(self):
        return dist.get_rank()

    def allgather(self, tensor, tensor_list):
        raise NotImplementedError()
    
    def allreduce(self, tensors, op=Ops.Average):
        handles = []
        for tensor in tensors:
            handles.append(dist.all_reduce(tensor, async_op=True))
        self._sync(handles)
        if op == Ops.Average:
            for tensor in tensors:
                tensor /= self.size()

    def broadcast(self, tensors, ranks):
        handles = []
        for tensor, rank in zip(tensors, ranks):
            handles.append(dist.broadcast(tensor, src=rank, async_op=True))
        self._sync(handles)

    def gather(self, tensors, rank=0):
        raise NotImplementedError()

    def reduce(self, tensors, ranks, op=Ops.Average):
        handles = []
        for tensor, rank in zip(tensors, ranks):
            handles.append(dist.reduce(tensor, dst=rank, async_op=True))
        self._sync(handles)
        if op == Ops.Average:
            for tensor, rank in zip(tensors, ranks):
                if self.rank() == rank:
                    tensor /= self.size()

    def reduce_scalar(self, tensor):
        dist.all_reduce(tensor)

    def barrier(self):
        dist.barrier()

    def _sync(self, handles):
        for handle in handles:
            handle.wait()

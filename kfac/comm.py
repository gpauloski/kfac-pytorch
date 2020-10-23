import enum
import os
import torch
import torch.distributed as dist

try:
    import horovod.torch as hvd
    HVD_EXISTS = True
except:
    HVD_EXISTS = False


# The global var containing the current initialized backend object
backend = None


class Ops(enum.Enum):
    Average = "average"
    Sum = "sum"


def init_comm_backend():
    global backend
    if backend is None:
        backend = _get_comm_backend()


def _get_comm_backend():
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
    
    def allreduce(self, tensor, op=Ops.Average, async=True):
        """Allreduce tensor inplace.

        Args:
          tensor (torch.Tensor)
          op (Op): reduction operation to apply (default: Ops.Average)
          async (bool): whether this op should be asynchronous (default: True)

        Returns:
          Async work handle, if async_op is True, else None
        """
        pass

    def broadcast(self, tensors, ranks, async=True):
        """Broadcast tensor.

        Args:
          tensor (torch.Tensor)
          rank (int): source rank for tensor
          async (bool): whether this op should be asynchronous

        Returns:
          Async work handle, if async_op is True, else None
        """
        pass
    
    def reduce(self, tensor, rank, op=Ops.Average, async=True):
        """Reduce tensor inplace.

        Args:
          tensor (torch.Tensor)
          rank (int): dest rank for reduction
          op (Op): reduction operation to apply (default: Ops.Average)
          async (bool): whether this op should be asynchrous (default: True)

        Returns:
          Async work handle, if async_op is True, else None
        """
        pass
    
    def barrier(self):
        return

    def sync(self, handles):
        """Executes asynchonous handles.

        Args:
          handles (handle or list(handles): handles returns by async functions
        """
        return

class HorovodBackend(CommBackend):
    def size(self):
        return hvd.size()
    
    def local_rank(self):
        return hvd.local_rank()

    def rank(self):
        return hvd.rank()

    def allreduce(self, tensor, op=Ops.Average, async=True):
        op = self._get_op(op)
        if async:
            return hvd.allreduce_async_(tensor, op=op)
        else:
            hvd.allreduce_(tensor, op=op)

    def broadcast(self, tensor, rank, async=True):
        if async:
            return hvd.broadcast_async_(tensor, root_rank=rank)
        else:
            hvd.broadcast_(tensor, root_rank=rank)

    def reduce(self, tensor, rank, op=Ops.Average, async=True):
        # Horovod only support allreduce
        self.allreduce(tensor, op=op, async=async)

    def barrier(self):
        hvd.allreduce(torch.tensor(1), name='barrier')

    def _get_op(self, op):
        if op == Ops.Average:
            return hvd.Average
        elif op == Ops.Sum:
            return hvd.Sum
        else:
            raise ValueError('Unknown communication operation {}'.format(op))

    def sync(self, handles):
        if isinstance(handles, list):
            for handle in handles:
                hvd.synchronize(handle)
        else:
            hvd.synchronize(handles)


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
    
    def allreduce(self, tensor, op=Ops.Average, async=True):
        # Note: actually returns tuple(handle, tensor)
        # because there is no average op in Torch distribted so
        # we need to pass the tensor to sync() to be averages
        handle = dist.all_reduce(tensor, async_op=async)
        
        if not async:
            if op == Ops.Average:
                tensor /= self.size()
            return
        else:
            if op == Ops.Average:
                return (handle, tensor)
            return handle
                

    def broadcast(self, tensor, rank, async=True):
        return dist.broadcast(tensor, src=rank, async_op=async)

    def reduce(self, tensor, rank, op=Ops.Average, async=True):
        # Note: actually returns tuple(handle, should_average)
        # because there is no average op in Torch distribted
        handle = dist.reduce(tensor, dst=rank, async_op=async)
        
        if not async:
            if op == Ops.Average:
                tensor /= self.size()
            return
        else:
            if op == Ops.Average:
                return (handle, tensor)
            return handle

    def barrier(self):
        dist.barrier()

    def sync(self, handles):
        if isinstance(handles, list):
            if len(handles) == 0:
               return
            if isinstance(handles[0], tuple):
                for handle, tensor in handles:        
                    handle.wait()
                    tensor /= self.size()
            else:
                for handle in handles:
                    handle.wait()
        else:
            if isinstance(handles, tuple):
                handle, tensor = handles
                handle.wait()
                tensor /= self.size()
            else:
                handles.wait()
 

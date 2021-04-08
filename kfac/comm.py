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


class Ops(enum.Enum):
    Average = "average"
    Sum = "sum"


class CommGroup(object):
    def __init__(self, ranks):
        self.ranks = ranks
        if (_torch_distributed_is_initialized() and 
                self.size > 1 and self.size < backend.size()):
            self.group = dist.new_group(ranks)
        else:
            self.group = None

    @property
    def size(self):
        return len(self.ranks)

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
    
    def allreduce(self, tensor, op=Ops.Average, group=None, async_op=True):
        """Allreduce tensor inplace.

        Args:
          tensor (torch.Tensor)
          op (Op): reduction operation to apply (default: Ops.Average)
          group (CommGroup): CommGroup for collective
              communication. If None, uses default group (default: None).
          async_op (bool): whether this op should be asynchronous (default: True)

        Returns:
          Async work handle, if async_op is True, else None
        """
        return

    def broadcast(self, tensor, src, group=None, async_op=True):
        """Broadcast tensor.

        Args:
          tensor (torch.Tensor)
          src (int): source rank for tensor.
          group (CommGroup): CommGroup for collective
              communication. If None, uses default group (default: None).
          async_op (bool, optional): whether this op should be asynchronous

        Returns:
          Async work handle, if async_op is True, else None
        """
        return
    
    def reduce(self, tensor, dst, op=Ops.Average, async_op=True):
        """Reduce tensor inplace.

        Args:
          tensor (torch.Tensor)
          dst (int): dest rank for reduction
          op (Op): reduction operation to apply (default: Ops.Average)
          async_op (bool): whether this op should be asynchrous (default: True)

        Returns:
          Async work handle, if async_op is True, else None
        """
        return
    
    def barrier(self):
        return

    def sync(self, handles):
        """Executes asynchonous handles.

        Args:
          handles (handle or list(handles): handles returns by async functions
        """
        return

    def wait(self, handle):
        """Execute handle and block until finished"""
        return

class HorovodBackend(CommBackend):
    def size(self):
        return hvd.size()
    
    def local_rank(self):
        return hvd.local_rank()

    def rank(self):
        return hvd.rank()

    def allreduce(self, tensor, op=Ops.Average, group=None, async_op=True):
        op = self._get_op(op)
        if async_op:
            return hvd.allreduce_async_(tensor, op=op)
        else:
            hvd.allreduce_(tensor, op=op)

    def broadcast(self, tensor, src, group=None, async_op=True):
        # HVD does not have broadcast groups so we ignore
        if async_op:
            return hvd.broadcast_async_(tensor, root_rank=src)
        else:
            hvd.broadcast_(tensor, root_rank=rank)

    def reduce(self, tensor, dst, op=Ops.Average, async_op=True):
        # Horovod only support allreduce
        self.allreduce(tensor, op=op, async_op=async_op)

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
                self.wait(handle)
        else:
            self.wait(handles)

    def wait(self, handle):
        if handle is not None:
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
    
    def allreduce(self, tensor, op=Ops.Average, group=None, async_op=True):
        if group is not None:
            if group.size <= 1:
                return
            kwargs = {'group': group.group} if group.group is not None else {}
        else:
            kwargs = {}
        # Note: actually returns tuple(handle, tensor)
        # because there is no average op in Torch distribted so
        # we need to pass the tensor to sync() to be averages
        handle = dist.all_reduce(tensor, async_op=async_op, **kwargs)
 
        if not async_op:
            if op == Ops.Average:
                tensor /= self.size()
            return
        else:
            if op == Ops.Average:
                return (handle, tensor)
            return handle

    def broadcast(self, tensor, src, group=None, async_op=True):
        if group is not None:
            if group.size <= 1:
                return
            kwargs = {'group': group.group} if group.group is not None else {}
            return dist.broadcast(tensor, src=src, async_op=async_op, **kwargs)
        return dist.broadcast(tensor, src=src, async_op=async_op)

    def reduce(self, tensor, dst, op=Ops.Average, async_op=True):
        # Note: actually returns tuple(handle, should_average)
        # because there is no average op in Torch distribted
        handle = dist.reduce(tensor, dst=dst, async_op=async_op)
        
        if not async_op:
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
                    self.wait(handle)
                    tensor /= self.size()
            else:
                for handle in handles:
                    self.wait(handle)
        else:
            if isinstance(handles, tuple):
                handle, tensor = handles
                self.wait(handle)
                tensor /= self.size()
            else:
                self.wait(handles)

    def wait(self, handle):
        if handle is not None:
            handle.wait() 

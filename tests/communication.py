import argparse
import kfac
import torch


@kfac.utils.trace(sync=True)
def broadcast(t, group, src=0):
    #handle = kfac.comm.backend.broadcast(t, src=src, group=group)
    handle = kfac.comm.backend.allreduce(t, op=kfac.comm.Ops.Sum, group=group)
    kfac.comm.backend.sync(handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Communication Test')
    parser.add_argument('--backend', type=str, default='nccl',
                        help='communication backend')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device for tensors')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training')
    args = parser.parse_args()

    torch.distributed.init_process_group(backend=args.backend, init_method='env://')
    kfac.comm.init_comm_backend()

    if args.device == 'cuda':
        torch.cuda.set_device(args.local_rank)
    
    print('rank = {}, world_size = {}, device_ids = {}'.format(
            torch.distributed.get_rank(), torch.distributed.get_world_size(),
            args.local_rank))

    size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    def get_group(groups):
        for group in groups:
            if rank in group.ranks:
                return group

    def factorize(num):
        return [n for n in range(1, round(num / 2) + 1) if num % n == 0]

    counts = factorize(size) + [size]
    tensor = torch.empty(100, 100).to(args.device)

    for count in counts:
        gs = kfac.utils.partition_inv_ranks(size, count)
        gs = [kfac.comm.CommGroup(g) for g in gs]
        g = get_group(gs)

        for _ in range(20):
            broadcast(tensor, g, src=g.ranks[0])

        if rank == 0: print('Group size {}/{}: {}'.format(count, size, kfac.utils.get_trace()))
        kfac.utils.clear_trace()


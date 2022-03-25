# KFAC Training Examples

Cifar-10 and ImageNet distributed training with K-FAC examples.

## Requirements

The provided example training scripts have additional package requirements defined in `examples/requirements.txt`.

```
$ pip install -r examples/requirements.txt
```

## Usage

Note: these examples use the `torchrun` launcher which is only available in PyTorch 1.10 and later.
For PyTorch 1.9, use `python -m torch.distributed.run` and for PyTorch 1.8, use `python -m torch.distributed.launch`.


#### Single Node, Multi-GPU
```
$ torchrun --standalone --nnodes 1 --nproc_per_node=[NGPUS] \
      examples/torch_{cifar10,imagenet}_resnet.py [ARGS]
```

#### Multi-Node, Multi-GPU
On each node, run:
```
$ torchrun --nnodes=[NNODES] --nproc_per_node=[NGPUS] --rdzv_backend=c10d --rdzv_endpoint=[HOSTADDR] \
      examples/torch_{cifar10,imagenet}_resnet.py [ARGS]
```

The full list of arguments can be found with the `--help` argument.
E.g., `python examples/torch_cifar10_resnet.py --help`.

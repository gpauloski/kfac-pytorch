# KFAC Training Examples

Distributed training with K-FAC examples for computer vision (ImageNet and
CIFAR-10) and language modeling (PennTreebank and WikiText) tasks.

## Requirements

The provided example training scripts require KFAC to be installed and the
additional requirements defined in `examples/requirements.txt`.

```
$ pip install -e .
$ pip install -r examples/requirements.txt
```

## Usage

### Single Node, Multi-GPU
```
$ torchrun --standalone --nnodes 1 --nproc_per_node=[NGPUS] \
      examples/torch_{...}.py [ARGS]
```

### Multi-Node, Multi-GPU
On each node, run:
```
$ torchrun --nnodes=[NNODES] --nproc_per_node=[NGPUS] --rdzv_backend=c10d --rdzv_endpoint=[HOSTADDR] \
      examples/torch_{...}.py [ARGS]
```

The full list of arguments can be found with the `--help` argument.
E.g., `python examples/torch_cifar10_resnet.py --help`.

#!/bin/bash

NGPUS=4
NNODES=1
MASTER=""

while getopts ":hcn:N:m:b:i:s:l:" opt; do
  case $opt in
    h) echo "-h         Display this help message"
       echo "-n [ngpus] Number of GPUs to use on each node (default 2)"
       echo "-N [nodes] Number of nodes to use (default 1)"
       echo "-m [mastr] Address of master node. Only used if NNODES>1"
       exit 0
    ;;
    n) NGPUS="$OPTARG"
    ;;
    N) NNODES="$OPTARG"
    ;;
    m) MASTER="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
  esac
done

if [ $NNODES -eq 1 ]; then
  python -m torch.distributed.launch \
      --nproc_per_node=$NGPUS \
    examples/torch_imagenet_resnet.py \
      --kfac-update-freq 100 \
      --kfac-cov-update-freq 10 \
      --damping 0.001 \
      --epochs 55 \
      --lr-decay 25 35 40 45 50 \
      --model resnet50
else
  python -m torch.distributed.launch \
      --nproc_per_node=$NGPUS \
      --nnodes=$NNODES \
      --node_rank=$OMPI_COMM_WORLD_RANK \
      --master_addr="$MASTER" \
    examples/torch_imagenet_resnet.py \
      --kfac-update-freq 100 \
      --kfac-cov-update-freq 10 \
      --damping 0.001 \
      --epochs 55 \
      --lr-decay 25 35 40 45 50 \
      --model resnet50
fi

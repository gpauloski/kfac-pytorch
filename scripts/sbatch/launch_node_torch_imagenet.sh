#!/bin/bash

NGPUS=1
NNODES=1
MASTER=""
MVAPICH=false

while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | sed 's/^[^=]*=//g'`
    if [[ "$VALUE" == "$PARAM" ]]; then
        shift
        VALUE=$1
    fi
    case $PARAM in
        -h|--help)
            echo "USAGE: ./launch_node_torch_imagenet.sh"
            echo "  -h,--help           Display this help message"
            echo "  -N,--ngpus [count]  Number of GPUs per node (default: 1)"
            echo "  -n,--nnodes [count] Number of nodes this script is launched on (default: 1)"
            echo "  -m,--master [addr]  Address of master node (default: \"\")"
            echo "  --mvapich           Use MVAPICH env variables for initialization (default: false)"
            exit 0
        ;;
        -N|--ngpus)
            NGPUS=$VALUE
        ;;
        -n|--nnodes)
            NNODES=$VALUE
        ;;
        -m|--master)
            MASTER=$VALUE
        ;;      
        --mvapich)
            MVAPICH=true
        ;;
        *)
          echo "ERROR: unknown parameter \"$PARAM\""
          exit 1
        ;;
    esac
    shift
done

if [ "$MVAPICH" == true ]; then
  LOCAL_RANK=$MV2_COMM_WORLD_RANK
else
  LOCAL_RANK=$OMPI_COMM_WORLD_RANK
fi

echo Launching torch.distributed: nproc_per_node=$NGPUS, nnodes=$NNODES, master_addr=$MASTER, local_rank=$LOCAL_RANK, using_mvapich=$MVAPICH

KWARGS=""
KWARGS+="--kfac-update-freq 100 "
KWARGS+="--kfac-cov-update-freq 10 "
KWARGS+="--damping 0.001 "
KWARGS+="--epochs 55 "
KWARGS+="--lr-decay 25 35 40 45 50 "
KWARGS+="--model resnet50 "

if [ $NNODES -eq 1 ]; then
  python -m torch.distributed.launch \
      --nproc_per_node=$NGPUS \
    examples/torch_imagenet_resnet.py $KWARGS
else
  python -m torch.distributed.launch \
      --nproc_per_node=$NGPUS \
      --nnodes=$NNODES \
      --node_rank=$LOCAL_RANK \
      --master_addr=$MASTER \
    examples/torch_imagenet_resnet.py $KWARGS
fi

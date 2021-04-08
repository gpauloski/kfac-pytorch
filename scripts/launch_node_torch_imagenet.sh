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
        --help)
            echo "USAGE: ./launch_node_torch_imagenet.sh"
            echo "  --help           Display this help message"
            echo "  --ngpus [count]  Number of GPUs per node (default: 1)"
            echo "  --nnodes [count] Number of nodes this script is launched on (default: 1)"
            echo "  --master [addr]  Address of master node (default: \"\")"
            echo "  --mvapich           Use MVAPICH env variables for initialization (default: false)"
            exit 0
        ;;
        --ngpus)
            NGPUS=$VALUE
        ;;
        --nnodes)
            NNODES=$VALUE
        ;;
        --master)
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

module load conda
conda deactivate
conda activate pytorch
module unload spectrum_mpi
module use /home/01255/siliu/mvapich2-gdr/modulefiles/
module load gcc/7.3.0 
module load mvapich2-gdr/2.3.4

export MV2_USE_CUDA=1
export MV2_ENABLE_AFFINITY=1
export MV2_THREADS_PER_PROCESS=2
export MV2_SHOW_CPU_BINDING=1
export MV2_CPU_BINDING_POLICY=hybrid
export MV2_HYBRID_BINDING_POLICY=spread
export MV2_USE_RDMA_CM=0
export MV2_SUPPORT_DL=1

export OMP_NUM_THREADS=4
which python

echo Launching torch.distributed: nproc_per_node=$NGPUS, nnodes=$NNODES, master_addr=$MASTER, local_rank=$LOCAL_RANK, using_mvapich=$MVAPICH

KWARGS=""
KWARGS+="--base-lr 0.0125 "
KWARGS+="--kfac-update-freq 500 "
KWARGS+="--kfac-cov-update-freq 50 "
KWARGS+="--damping 0.003 "
KWARGS+="--model resnet50 "
KWARGS+="--checkpoint-freq 10 "
KWARGS+="--kfac-comm-method comm-opt "
KWARGS+="--kfac-grad-worker-fraction 0.0 "
KWARGS+="--log-dir logs/kfac_comm_opt_500 "
KWARGS+="--fp16 "

# KFAC Schedule
KWARGS+="--epochs 55 "
KWARGS+="--lr-decay 25 35 40 45 50 "
# SGD Schedule
#KWARGS+="--epochs 90 "
#KWARGS+="--lr-decay 30 40 80 "

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

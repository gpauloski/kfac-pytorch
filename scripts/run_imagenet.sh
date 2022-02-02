#!/bin/bash
# USAGE:
#
#   To launch pretraining with this script, first customize the PRELOAD and
#   CMD variables for your training configuration.
#
#   Run locally on a compute node:
#
#     $ ./run_imagenet.sh
#
#   Submit as a Cobalt or Slurm job:
#
#     $ qsub -q QUEUE -A ALLOC -n NODES -t TIME run_imagenet.sh
#     $ sbatch -p QUEUE -A ALLOC -N NODES -t TIME run_imagenet.sh
#
#   Notes:
#     - training configuration (e.g., # nodes, # gpus / node, etc.) will be
#       automatically inferred from the nodelist
#     - additional arguments to the python script can be specified by passing
#       them as arguments to this script. E.g.,
#
#       $ ./run_imagenet.sh --epochs 55 --batch-size 128
#

PRELOAD="module load conda ; "
PRELOAD="conda activate pytorch ; "
PRELOAD="export OMP_NUM_THREADS=8 ; "

# Arguments to the training script are passed as arguments to this script
CMD="examples/torch_imagenet_resnet.py $@"

# Example: copy imagenet and extract to /tmp on each worker
# ./scripts/copy_and_extract.sh /path/to/imagenet.tar /tmp/imagenet

# Figure out training environment
if [[ -z "${NODEFILE}" ]]; then
    if [[ -n "${SLURM_NODELIST}" ]]; then
        NODEFILE=/tmp/imagenet_slurm_nodelist
        scontrol show hostnames $SLURM_NODELIST > $NODEFILE
    elif [[ -n "${COBALT_NODEFILE}" ]]; then
        NODEFILE=$COBALT_NODEFILE
    fi
fi
if [[ -z "${NODEFILE}" ]]; then
    RANKS=$HOSTNAME
    NNODES=1
else
    MAIN_RANK=$(head -n 1 $NODEFILE)
    RANKS=$(tr '\n' ' ' < $NODEFILE)
    NNODES=$(< $NODEFILE wc -l)
fi

# Torch Distributed Launcher
LAUNCHER="python -m torch.distributed.run "
LAUNCHER+="--nnodes=$NNODES --nproc_per_node=auto --max_restarts 0 "
if [[ "$NNODES" -eq 1 ]]; then
    LAUNCHER+="--standalone "
else
    LAUNCHER+="--rdzv_backend=c10d --rdzv_endpoint=$MAIN_RANK "
fi

# Combine preload, launcher, and script+args into full command
FULL_CMD="$PRELOAD $LAUNCHER $CMD"
echo "Training command: $FULL_CMD"

# Launch the pytorch processes on each worker (use ssh for remote nodes)
RANK=0
for NODE in $RANKS; do
    if [[ "$NODE" == "$HOSTNAME" ]]; then
        echo "Launching rank $RANK on local node $NODE"
        eval $FULL_CMD &
    else
        echo "Launching rank $RANK on remote node $NODE"
        ssh $NODE "cd $PWD; $FULL_CMD" &
    fi
    RANK=$((RANK+1))
done

wait

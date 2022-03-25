#!/bin/bash
# Copy tar file to destination dir and extract on all nodes in environment
#
# Usage:
#   $ ./scripts/copy_and_extract.sh /path/to/source.tar /path/to/dest
#
# Nodes on the environment are inferred from $NODEFILE, $SLURM_NODELIST, or
# $COBALT_NODEFILE in this order. If none of these variables are set, the
# script just executes the copy and extract locally.

SOURCE_TAR=$1
DEST_DIR=$2

mkdir -p $DEST_DIR

FULL_CMD="cp $SOURCE_TAR $DEST_DIR ; "
FULL_CMD+="cd $DEST_DIR ; "
FULL_CMD+="tar -xf $SOURCE_TAR "

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
else
    RANKS=$(tr '\n' ' ' < $NODEFILE)
fi

echo "Command: $FULL_CMD"

# Launch execute the command on each worker (use ssh for remote nodes)
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

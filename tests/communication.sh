#!/bin/bash

HOSTFILE=/tmp/hostfile
cat $HOSTFILE

GPU_PER_NODE=4
NODES=$(wc -l < $HOSTFILE)
MASTER_NODE=$(head -n 1 $HOSTFILE)

mpiexec -hostfile $HOSTFILE -N $NODES \
  ./tests/launch_communication.sh --ngpus $GPU_PER_NODE --nnodes $NODES --master $MASTER_NODE --mvapich


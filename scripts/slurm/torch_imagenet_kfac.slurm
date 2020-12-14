#!/bin/bash

# Sample Slurm job script
#   for TACC Longhorn Nodes
#
#------------------------------------------------------------------------------

#SBATCH -J imgkfc16                     # Job name
#SBATCH -o imgnet_kfc16.o%j             # Name of stdout output file
#SBATCH -N 16                           # Total # of nodes 
#SBATCH -n 64                           # Total # of mpi tasks
#SBATCH -t 12:00:00                     # Run time (hh:mm:ss)
#SBATCH --mail-user=XXX                 # Email address
#SBATCH --mail-type=end                 # Send email at begin and end of job
#SBATCH -p v100                         # Queue
#SBATCH -A XXX                          # Allocation

# Note mpiexec calls may need to be changed depending on MPI type

HOSTFILE=/tmp/hostfile

#scontrol show hostnames $SLURM_NODELIST > $HOSTFILE
cat $HOSTFILE

GPU_PER_NODE=4
NODES=$(wc -l < $HOSTFILE)
MASTER_NODE=$(head -n 1 $HOSTFILE)

mpiexec -hostfile $HOSTFILE -N $NODES ./scripts/cp_imagenet_to_temp.sh

# We use mpiexec to launch one python -m torch.distributed.launch per node
mpirun_rsh -hostfile $HOSTFILE -np $NODES \
  ./scripts/launch_node_torch_imagenet.sh \
    --ngpus $GPU_PER_NODE --nnodes $NODES --master $MASTER_NODE --mvapich

# -----------------------------------------------------------------------------

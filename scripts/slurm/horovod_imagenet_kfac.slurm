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

scontrol show hostnames $SLURM_NODELIST > /tmp/hostfile

cat /tmp/hostfile

mpiexec -hostfile /tmp/hostfile -N 1 ./scripts/cp_imagenet_to_temp.sh

mpiexec -hostfile /tmp/hostfile -N 4 \
   python examples/horovod_imagenet_resnet.py \
     --kfac-update-freq 100 \
     --kfac-cov-update-freq 10 \
     --damping 0.001 \
     --epochs 55 \
     --lr-decay 25 35 40 45 50 \
     --model resnet50

# -----------------------------------------------------------------------------

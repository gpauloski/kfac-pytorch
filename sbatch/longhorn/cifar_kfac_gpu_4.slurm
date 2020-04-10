#!/bin/bash

# Sample Slurm job script
#   for TACC Longhorn Nodes
#
#------------------------------------------------------------------------------

#SBATCH -J cifkfc4                 # Job name
#SBATCH -o sbatch_logs/cif_kfc4.o%j # Name of stdout output file
#SBATCH -N 1                       # Total # of nodes 
#SBATCH -n 4                       # Total # of mpi tasks
#SBATCH -t 4:00:00                # Run time (hh:mm:ss)
#SBATCH --mail-user=jgpauloski@utexas.edu
#SBATCH --mail-type=end            # Send email at begin and end of job
#SBATCH -p v100
#SBATCH -A Deep-Learning-at-Sca    # Allocation

mkdir -p sbatch_logs

module load cuda/10.1

source $SCRATCH/anaconda3/bin/activate pytorch

scontrol show hostnames $SLURM_NODELIST > /tmp/hostfile

cat /tmp/hostfile

mpiexec -hostfile /tmp/hostfile -N 4 \
   python examples/pytorch_cifar10_resnet.py \
     --base-lr 0.1 \
     --epochs 100 \
     --kfac-update-freq 10 \
     --model resnet32 \
     --lr-decay 40 80

# -----------------------------------------------------------------------------

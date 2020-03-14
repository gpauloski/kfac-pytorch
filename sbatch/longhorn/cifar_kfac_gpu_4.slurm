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

# Get nodelist by sorting jobs by latest start to earliest and getting second line
SLURM_NODELIST=$(squeue --sort M --format %N -u $USER | head -2 | tail -1)
scontrol show hostnames $SLURM_NODELIST > /tmp/hostfile

cat /tmp/hostfile

mpiexec -hostfile /tmp/hostfile -N 4 \
   python examples/pytorch_cifar10_resnet.py \
     --lr 0.1 \
     --epochs 100 \
     --kfac-update-freq 10 \
     --model resnet32 \
     --lr-decay 60 80

# -----------------------------------------------------------------------------

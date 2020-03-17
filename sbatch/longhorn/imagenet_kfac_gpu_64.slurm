#!/bin/bash

# Sample Slurm job script
#   for TACC Longhorn Nodes
#
#------------------------------------------------------------------------------

#SBATCH -J imgkfc64                 # Job name
#SBATCH -o sbatch_logs/imgnet_kfc64.o%j # Name of stdout output file
#SBATCH -N 16                       # Total # of nodes 
#SBATCH -n 64                       # Total # of mpi tasks
#SBATCH -t 24:00:00                # Run time (hh:mm:ss)
#SBATCH --mail-user=jgpauloski@utexas.edu
#SBATCH --mail-type=end            # Send email at begin and end of job
#SBATCH -p v100
#SBATCH -A Deep-Learning-at-Sca    # Allocation

mkdir -p sbatch_logs

module load cuda/10.1

source $SCRATCH/anaconda3/bin/activate pytorch

scontrol show hostnames $SLURM_NODELIST > /tmp/hostfile

cat /tmp/hostfile

mpiexec -hostfile /tmp/hostfile -N 1 ./sbatch/cp_imagenet_to_temp.sh

mpiexec -hostfile /tmp/hostfile -N 4 \
   python examples/pytorch_imagenet_resnet50.py \
     --kfac-update-freq 10 \
     --epochs 55 \
     --lr-decay 25 40 45 50 \
     --damping 0.001 

# -----------------------------------------------------------------------------

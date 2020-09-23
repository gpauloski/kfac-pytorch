#!/bin/bash
# To kill hanging Python CUDA processes, run with:
#   `mpiexec -hostfile /path/to/hostfile -N 1 ./sbatch/kill_python_processes.py`

nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9

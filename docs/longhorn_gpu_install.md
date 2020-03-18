## To Install Horovod with MPI on Longhorn

Create a new Conda Python env
```
$ conda create -n yourenvname python=3.6
$ conda activate yourenvname
$ conda config --add channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/
```

Install base dependencies
```
$ conda install tensorflow-gpu==1.15         # I used version 1.15 but 1.14 should work as well
$ conda install gxx_linux-ppc64le=7.3.0 cffi
$ module load cuda/10.1
$ pip install kfac tensorflow-probability pytorch
```

Install Horovod 

```
$ HOROVOD_CUDA_HOME=/opt/apps/cuda/10.1/ HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_HOME=/opt/apps/cuda10_0/nccl/2.4.8/   HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 pip install horovod --no-cache-dir
Collecting horovod
```

Other installs:

Note: I use Horovod==0.19.0 because of this fix: "Fixed hvd.allgather to work with CUDA tensors when building with HOROVOD_GPU_ALLGATHER=MPI (#1480)."

- NCCL backend (NCCL should be faster than MPI but Horovod does not support allgather with NCCL)
  ```
  $ HOROVOD_CUDA_HOME=/usr/local/cuda-10.0/ HOROVOD_GPU_ALLREDUCE=NCCL \
        HOROVOD_NCCL_HOME=/opt/apps/cuda10_0/nccl/2.4.8/ HOROVOD_WITH_TENSORFLOW=1 \
        HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 \
        pip install horovod --no-cache-dir
  ```

- MPI backend (seg faults in Spectrum MPI currently)
  ```
  $ conda install ddl cudatoolkit-dev
  $ HOROVOD_CUDA_HOME=$CONDA_PREFIX HOROVOD_GPU_ALLREDUCE=MPI HOROVOD_GPU_ALLGATHER=MPI \
        HOROVOD_GPU_BROADCAST=MPI pip install horovod==0.19.0 --no-cache-dir
  ```
  
- DDL backend
  ```
  $ conda install ddl cudatoolkit-dev
  $ HOROVOD_CUDA_HOME=$CONDA_PREFIX HOROVOD_GPU_ALLREDUCE=DDL pip install horovod==0.18.2 --no-cache-dir
  ```
  
- Mixed backend (experimental)
  ```
  $ HOROVOD_CUDA_HOME=/usr/local/cuda-10.0/ HOROVOD_GPU_ALLREDUCE=NCCL \
        HOROVOD_NCCL_HOME=/opt/apps/cuda10_0/nccl/2.4.8/ HOROVOD_GPU_BROADCAST=NCCL \
        HOROVOD_GPU_ALLGATHER=MPI HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 \
        HOROVOD_WITHOUT_MXNET=1 HOROVOD_ALLOW_MIXED_GPU_IMPL=1 \
        pip install horovod==0.19.0 --no-cache-dir
  ```
      
See `sbatch/resnet50-longhorn-4nodes.slurm` for an example of running.

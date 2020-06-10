## Install Pytorch 1.6 and TF 1.15 with Horovod
```
$ conda create --name pytorch1.6 --python=3.6
$ conda activate pytorch1.6
$ conda install gxx_linux-ppc64le=7.3.0 cffi tensorflow-gpu==1.15
$ pip install /scratch/00946/zzhang/pytorch/dist/torch-1.6.0a0+c02cb7a-cp36-cp36m-linux_ppc64le.whl
$ pip install /scratch/00946/zzhang/pytorch-wheels/wheels-master/pytorch_wheels/torchvision-0.6.0a0+7ee5a8b-cp36-cp36m-linux_ppc64le.whl
$ HOROVOD_CUDA_HOME=$CONDA_PREFIX HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_TENSORFLOW=1 pip install horovod --no-cache-dir
```

## KFAC Install
```
$ pip install torchsummary tqdm
$ cd /path/to/kfac
$ pip install .
```

## Install TorchText
see `torchtext_longhorn_install.md`

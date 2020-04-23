# Frontera GPU Install

## Install Pytorch
```
$ pip install pytorch==1.14.0 torchvision==0.5.0
```

## Install Horovod
```
$ module load cuda/10.1 cudnn/7.5.0 nccl/2.4.2
$ CPLUS_INCLUDE_PATH=/opt/apps/intel18/boost/1.69/include/ \
    CC=gcc HOROVOD_CUDA_HOME=/opt/apps/cuda/10.1 HOROVOD_GPU_ALLREDUCE=NCCL \
    HOROVOD_NCCL_HOME=/opt/apps/cuda10_1/nccl/2.4.2 HOROVOD_WITHOUT_TENSORFLOW=1 \
    HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 \
    pip3 install --user horovod==0.19.0 --no-cache-dir
```

Note this installs Horovod without TensorFlow. This is because the default
Python on Frontera is 3.7 which TensorFlow 1.14 is incompatible with. Using
Python 3.6 will allow you to change `HOROVOD_WITHOUT_TENSORFLOW=1` to
`HOROVOD_WITH_TENSORFLOW=1`.

## To Install Horovod + PyTorch on Longhorn

Create a new Conda Python env
```
$ module load conda
$ conda config --add channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/
$ conda create -n yourenvname python=3.6
$ conda activate yourenvname
```

Install PyTorch and Horovod
```
$ conda install pytorch=1.1 tqdm
$ module load gcc/7.3.0
$ module load cuda/10.1
$ export CPATH=/usr/local/cuda-10.1/include:$CPATH
$ HOROVOD_CUDA_HOME=/opt/apps/cuda/10.1/ HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_HOME=/opt/apps/cuda10_0/nccl/2.4.8/   HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 pip install horovod==0.19.2 --no-cache-dir
$ module unload gcc/7.3.0
$ module load spectrum_mpi
```

Install KFAC 
```
$ git clone https://github.com/gpauloski/kfac_pytorch.git
$ cd kfac_pytorch
$ pip install .   # use -e to install in develop mode
```

To Train: grab nodes and place data onto `/tmp`.
```
$ cd kfac_pytorch
$ scontrol show hostname c001-[006-007] > hostfile
$ mpiexec -hostfile hostfile -N 4  python examples/pytorch_imagenet_resnet.py --model resnet50 --kfac-update-freq 2000 --kfac-cov-update-freq 200 --epochs 55 --base-lr 0.0125 --lr-decay 25 35 40 45 50  --warmup-epochs 5 --damping 0.001 --train-dir /tmp/ILSVRC2012_img_train/ --val-dir /tmp/ILSVRC2012_img_val
```

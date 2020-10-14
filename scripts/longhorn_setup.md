## To Install Horovod + PyTorch on Longhorn

### Easy Install

1. Load the most recent PyTorch Conda environment provided on Longhorn.
   ```
   $ module load pytorch-py3
   ```
2. Clone and install the KFAC repo.
   ```
   $ git clone https://github.com/gpauloski/kfac_pytorch.git
   $ cd kfac_pytorch
   $ git checkout experimental  # Note: use experimental branch for most up to date features
   $ pip install -e .           # Note: -e installs in development mode
   ```
3. Install dependencies for example ResNet scripts.
   ```
   $ pip install torch-summary tqdm torchvision
   ```
4. Test Cifar10 + ResNet single-node, 4 GPU training (NOTE: do this on a compute node!).
   - **torch.distributed**:
      ```
      $ python -m torch.distributed.launch --nproc_per_node=4 examples/torch_cifar10_resnet.py --epochs 5
      ```
   - **Horovod**:
      ```
      $ mpiexec -hostfile /tmp/hostfile -N 4 python examples/horovod_cifar10_resnet.py --epochs 5
      ```
   Both should complete without any errors in 1-2 minutes.
   Training times can vary depending on the versions of PyTorch and CUDA.
   For multi-node training, the Horovod usage is the same (just add more nodes to the hostfile).
   Multi-node training with torch.distributed is slightly more complicated and requires using `mpiexec` to launch a `torch.distributed.launch` process on each worker.
   See `scripts/slurm/torch_imagenet_kfac.slurm` for an example of how this can be done.
   
### Advanced Install (PyTorch 1.6)

FP16 training with KFAC requires `torch.cuda.amp` which was added PyTorch 1.6.
Longhorn does not have PyTorch 1.6 Conda enviroments so we must build our own.
KFAC with `torch.cuda.amp` and Horovod backend has not been tested so we will just use torch.distributed training.
This advanced install should also have faster training times than the easy install.

1. Create a new Conda Python env.
   ```
   $ module load conda
   $ conda config --add channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/
   $ conda create -n yourenvname python=3.6
   $ conda activate yourenvname
   ```
   Note: sometime the paths get messed up if you are working with multiple Conda environments so I usually exit and launch a new shell.
2. Install packages.
   ```
   $ conda install cudatoolkit-dev==10.2.89 nccl==2.5.6 cudnn
   $ conda install numpy hdf5 tensorboard blas cffi onnx six pillow
   ```
3. Install PyTorch 1.6
   ```
   $ pip install /scratch/00946/zzhang/pytorch/dist/torch-1.6.0a0+168cddf-cp36-cp36m-linux_ppc64le.whl
   ```
4. Continue with steps (2) and (3) from the Easy Install section to install KFAC and the example requirements.
5. Run Cifar10 + ResNet example with torch.distributed and AMP.
   ```
   $ python -m torch.distributed.launch --nproc_per_node=4 examples/torch_cifar10_resnet.py --epochs 5 --fp16
   ```
   This should take about 1 minute to run.

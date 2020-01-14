# PytorchExamples

## Set Up Romeo (RHEL7) For Pytorch Parallel Compatibility

```bash
module load anaconda
module load cuda/10.0
module load cudnn/10.0-v7.4.1
module load gcc/6.4.0
export CUDNN_INCLUDE_DIR=/opt/cudnn-10.0-linux-x64-v7.4.1/cuda/include
export CUDNN_LIB_DIR=/opt/cudnn-10.0-linux-x64-v7.4.1/cuda/lib64
conda create -n py36 python=3.6
source activate py36
export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
export CMAKE_PREFIX_PATH="/opt/gcc-6.4.0;$CMAKE_PREFIX_PATH"
conda install -c anaconda libgcc-ng libstdcxx-ng
conda install -c conda-forge openmpi
conda install numpy ninja pyyaml mkl setuptools cmake cffi
conda install -c pytorch magma-cuda100
conda install -c mingfeima mkldnn
conda uninstall --force mkl
pip install mkl
pip install mkl_include
conda uninstall --force cmake
pip install cmake
chmod +x ~/.conda/envs/py36/lib/python3.6/site-packages/cmake/data/bin/*
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch/
BUILD_TEST=0 python setup.py install
```

### Credits 

Allan Streib of Future Systems, Indiana University Bloomington designed the set up. 

## Setting Up RHEL7 (Romeo r-003)

```bash
module load anaconda
module load cuda/10.0
module load cudnn/10.0-v7.4.1
module load gcc/6.4.0
export CUDNN_INCLUDE_DIR=/opt/cudnn-10.0-linux-x64-v7.4.1/cuda/include
export CUDNN_LIB_DIR=/opt/cudnn-10.0-linux-x64-v7.4.1/cuda/lib64
```

### Note

```text
Private Note: This is for running in r-003 RHEL7
```

```bash
source activate ENV1
```

## Running Examples

### RHEL7 Compatible

```bash
mpirun -n <parallelism> python3 mnist/mnist_dist_rhel7.py
```

import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import mpi4py.rc

mpi4py.rc.initialize = False
from mpi4py import MPI
import numpy as np

from torch.nn.parallel import DistributedDataParallel as DDP

# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

'''
This is an MPI-based DDP program with GLOO backend
MPI spawns the processes and provide the rank for the 
DDP program. Here the MPI program provides unique dataset
from each process corresponding to the GPU process. 

This kind of a programming model can support N:N way parallelism.
MPI program (data engineering program) and Deep learning program
running on equal number of processes. 

This program runs with CPUs.

'''


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size, data):
    print(f"Running basic DDP example on rank {rank} : {data.shape} {data[0]}.")
    setup(rank, world_size)
    data_tensor = torch.from_numpy(data)
    # create model and move it to GPU with id rank
    model = ToyModel()
    ddp_model = DDP(model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(data_tensor)
    labels = torch.randn(20, 5)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


def mpi_program():
    if not MPI.Is_initialized():
        MPI.Init()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    data = np.ones(200, dtype=np.float32).reshape(20, 10) * rank
    print("MPI Program ", rank, data.shape)
    if not MPI.Is_finalized():
        MPI.Finalize()
    return rank, size, data


def run_demo(demo_fn, world_size):
    rank, size, data = mpi_program()
    print("Final Data From ", data.shape, data[0])
    demo_fn(rank, size, data)


if __name__ == "__main__":
    run_demo(demo_basic, 4)

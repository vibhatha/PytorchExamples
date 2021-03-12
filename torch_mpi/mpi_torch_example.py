import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import socket
from torch.multiprocessing import Process
import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI
from torch.nn.parallel import DistributedDataParallel as DDP


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size):
    print("Running basic DDP example on rank ", rank)

    # create model and move it to GPU with id rank
    model = ToyModel()
    ddp_model = DDP(model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5)
    loss_fn(outputs, labels).backward()
    optimizer.step()



def run(rank, size, hostname):
    print("I am ", rank, "of ",size, "in", hostname)
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])


def init_processes(rank, size, hostname, fn, backend='tcp'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=rank, world_size=size)
    if not MPI.Is_initialized():
        MPI.Init()
    fn(rank, size)

def cleanup():
    dist.destroy_process_group()
    if MPI.Is_finalized():
        MPI.Finalize()


if __name__ == "__main__":
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    hostname = socket.gethostname()
    init_processes(world_rank, world_size, hostname, demo_basic, backend='mpi')
    cleanup()

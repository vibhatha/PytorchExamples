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
This is an MPI-based DDP program with NCCL backend
MPI spawns the processes and provide the rank for the 
DDP program. Here the MPI program provides unique dataset
from each process corresponding to the GPU process. 

This kind of a programming model can support N:N way parallelism.
MPI program (data engineering program) and Deep learning program
running on equal number of processes. 

'''

NUM_SAMPLES = 5
INPUT_FEATURES = 5
OUT_FEATURES = 5
HIDDEN_FEATURES = 2


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(INPUT_FEATURES, HIDDEN_FEATURES)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(HIDDEN_FEATURES, OUT_FEATURES)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size, data):
    print(f"Running basic DDP example on rank {rank} : {data}.")
    setup(rank, world_size)
    data_tensor = torch.from_numpy(data)
    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(data_tensor)
    labels = torch.randn(NUM_SAMPLES, OUT_FEATURES).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()
    if rank == 0:
        print(model)
    cleanup()


def mpi_program():
    if not MPI.Is_initialized():
        MPI.Init()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    data = np.ones(NUM_SAMPLES * INPUT_FEATURES, dtype=np.float32).reshape(NUM_SAMPLES, INPUT_FEATURES) * rank
    print("MPI Program ", rank, data.shape)
    if not MPI.Is_finalized():
        MPI.Finalize()
    return rank, size, data


def run_demo(demo_fn, world_size):
    rank, size, data = mpi_program()
    print("Final Data From ", data)
    demo_fn(rank, size, data)


def data_gen(rank):
    return np.ones(NUM_SAMPLES * INPUT_FEATURES, dtype=np.float32).reshape(NUM_SAMPLES, INPUT_FEATURES) * rank


def send_to_groups(rank, data, map_groups):
    destination_group = map_groups[rank]
    '''
    Design an algorithm to do sending from N processes to M processes where N > M
    '''


def process_mapper(num_mpi_procs, num_gpu_procs):
    corresponding_mpi_ranks = [i for i in range(num_gpu_procs)]
    all_mpi_ranks = [i for i in range(num_mpi_procs)]

    def get_process_groups(processes, num_groups):
        """Yield successive n-sized chunks from lst."""
        num_groups = max(1, num_groups)
        return [processes[i:i + num_groups] for i in range(0, len(processes), num_groups)]

    return get_process_groups(all_mpi_ranks, num_gpu_procs)


def run_dl():
    n_gpus = torch.cuda.device_count()
    if n_gpus < 4:
        print(f"Requires at least 8 GPUs to run, but got {n_gpus}.")
    else:
        run_demo(demo_basic, 4)


if __name__ == "__main__":
    process_mapper(16, 4)


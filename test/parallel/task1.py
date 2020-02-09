import numpy as np

import mpi4py
mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI

MPI.Init()

comm = MPI.COMM_WORLD

world_rank = comm.Get_rank()
world_size = comm.Get_size()

print("Rank {}, World Size {}".format(world_rank, world_size))

recv_data = np.array([0, 0, 0, 0], dtype="i")

if world_rank == 0:
    input = np.array([1, 2, 3, 4], dtype="i")
    dtype = MPI.INT
    dest = 3
    print("Rank {} Sending {} to Rank {}".format(world_rank, input, dest))
    comm.Send([input, dtype], dest=dest, tag=0)

if world_rank == 1:
    comm.Recv([recv_data, MPI.INT], source=2, tag=0)


print("Rank {} Data Received {}".format(world_rank, recv_data))

MPI.Finalize()

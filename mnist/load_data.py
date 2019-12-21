from random import Random
import torch
from torchvision import datasets, transforms
from mpi4py import MPI

"""
Torchvision is not a part of the torch version obtained when compiled from source. 
In order to keep RHEL7 requirements satisfied and support heavy weight data pre-processing
in the Big data frameworks, we decouple the data loading part from the Deep Learning workload. 
This is a proof of concept. 
The data is loaded using Pytorch APIs (TorchVision). 
Then the data is converted into numpy and passed to data partition programme that is used in 
distributed mode in Data Parallel Pytorch programmes. This is a proof of concept, showing the 
way it can be done. 
Instead of Torch distribution here we use MPI distributed APIs get world_rank and world_size. 
"""


class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


""" Partitioning MNIST """


def partition_dataset(size, rank, dataset):
    if (rank == 0):
        print("Data Loading")

    print(type(dataset))

    bsz = int(128 / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(rank)


    train_set = torch.utils.data.DataLoader(partition,
                                            batch_size=bsz,
                                            shuffle=True)

    print("Data Points Per Rank {} of Size {}".format(len(train_set.dataset), size))

    return train_set, bsz



train_set = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
test_set = datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))

train_set_array = train_set.data.numpy()
test_set_array = test_set.data.numpy()

print(train_set_array.shape)
print(test_set_array.shape)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

partition_dataset(size, rank, train_set_array)

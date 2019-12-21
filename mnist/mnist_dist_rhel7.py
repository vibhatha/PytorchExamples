from __future__ import print_function

from math import ceil
from random import Random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


""" Dataset partitioning helper """


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


def partition_numpy_dataset():
    print("Data Loading")
    dataset = np.load("datasets/train_data.npy")
    targets = np.load("datasets/train_target.npy")
    print(type(dataset))
    size = dist.get_world_size()
    bsz = int(128 / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]
    print("Partition Sizes {}".format(partition_sizes))
    partition_data = DataPartitioner(dataset, partition_sizes)
    partition_data = partition_data.use(dist.get_rank())
    train_set_data = torch.utils.data.DataLoader(partition_data,
                                                 batch_size=bsz,
                                                 shuffle=False)
    partition_target = DataPartitioner(targets, partition_sizes)
    partition_target = partition_target.use(dist.get_rank())
    train_set_target = torch.utils.data.DataLoader(partition_target,
                                                   batch_size=bsz,
                                                   shuffle=False)

    return train_set_data, train_set_target, bsz


""" Gradient averaging. """


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


""" Distributed Synchronous SGD Example """


def run(rank, size):
    if (rank == 0):
        print("Run Fn")

    torch.manual_seed(1234)
    train_set_data, train_set_target, bsz = partition_numpy_dataset()

    print("Data Points Per Rank {} of Size {}".format(len(train_set_data.dataset), size))
    model = Net()
    optimizer = optim.SGD(model.parameters(),
                          lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set_data.dataset) / float(bsz))
    if (rank == 0):
        print("Started Training")
    total_data = len(train_set_data)
    epochs = 10
    total_steps = epochs * total_data
    for epoch in range(10):
        epoch_loss = 0.0
        count = 0
        for data, target in zip(train_set_data, train_set_target):
            data = np.reshape(data, (data.shape[0], 1, data.shape[1], data.shape[2]))/ 128.0
            # print(
            #     "Data Size {}({},{}) of Rank {} : target {}, {}".format(data.shape, (data[0].numpy().dtype), type(data),
            #                                                             rank, target, len(target)))
            #print(data[0], target[0])
            count = count + 1
            result = '{0:.4g}'.format((count / float(total_steps)) * 100.0)
            print("Progress {}% \r".format(result), end='\r')
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            #print(epoch_loss)
            loss.backward()
            average_gradients(model)
            optimizer.step()
        if (rank == 0):
            print('Rank ', dist.get_rank(), ', epoch ',
                  epoch, ': ', epoch_loss / num_batches)


def init_processes(rank, size, fn, backend='tcp'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    print(world_rank, world_size)
    init_processes(world_rank, world_size, run, backend='mpi')
